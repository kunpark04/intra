"""Remote orchestrator for the Probe 1 1m TZ-fix audit. Separate from the
15m/1h audit (`_run_probe1_audit_remote.py`) because 1m is substantially
bigger: 16,384 combos on ~2M training bars, likely multi-hour wall-clock.

Modes:
  launch  — git-sync remote, launch 1m ET re-sweep under detached screen,
            wrapper also runs the stratified_recount_1m.py afterward
            (CT vs ET comparison directly on remote).

  poll    — screen-ls + log tail + free-mem + top process.

  pull    — SFTP the ET 1m parquet + manifest + comparison JSON back.
            Note: the CT parquet is 6.2 GB and stays remote.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST     = "195.88.25.157"
USER     = "root"
PASSWORD = "J@maicanP0wer123"
REMOTE_ROOT = "/root/intra"
JOB_NAME   = "probe1_1m_et_audit"
SCREEN_SESSION = JOB_NAME

REPO = Path(__file__).resolve().parents[1]

ET_SWEEP_OUTPUT  = "data/ml/probe1_audit/ml_dataset_v11_1m_et.parquet"
RECOUNT_SCRIPT   = "tasks/_probe1_stratified_recount_1m.py"
RECOUNT_JSON     = "data/ml/probe1_audit/stratified_recount_1m.json"
LOG_PATH         = "logs/probe1_1m_et_audit.log"
WRAPPER_PATH     = "run_probe1_1m_et_audit.sh"


def _connect() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return c


def _run(c: paramiko.SSHClient, cmd: str, timeout: int | None = 120) -> tuple[str, str, int]:
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def _git_sync(c: paramiko.SSHClient) -> None:
    print(f"[orch1m] git-syncing remote ...", flush=True)
    out, _, rc = _run(
        c,
        f"cd {REMOTE_ROOT} && git fetch origin master && "
        f"git reset --hard origin/master && git log -1 --oneline",
        timeout=60,
    )
    print(out, flush=True)
    if rc != 0:
        raise RuntimeError(f"git sync failed rc={rc}")


def _write_wrapper(c: paramiko.SSHClient) -> None:
    print(f"[orch1m] writing wrapper at {WRAPPER_PATH} ...", flush=True)
    wrapper = (
        "#!/bin/bash\n"
        "set -e\n"
        f"cd {REMOTE_ROOT}\n"
        "mkdir -p data/ml/probe1_audit logs\n"
        "echo \"[wrapper1m] start $(date -u +%FT%TZ)\"\n"
        "\n"
        "echo '[wrapper1m] ===== 1m ET re-sweep (16384 combos) ====='\n"
        "systemd-run --scope \\\n"
        "  -p MemoryMax=8G -p CPUQuota=280% \\\n"
        "  python3 scripts/param_sweep.py \\\n"
        "    --range-mode v11 \\\n"
        "    --timeframe 1min \\\n"
        "    --combinations 16384 \\\n"
        "    --seed 0 \\\n"
        "    --eval-partition train \\\n"
        "    --workers 3 \\\n"
        "    --bar-hour-tz ET \\\n"
        f"    --output {ET_SWEEP_OUTPUT}\n"
        "echo \"[wrapper1m] 1m ET sweep done $(date -u +%FT%TZ)\"\n"
        "\n"
        "echo '[wrapper1m] ===== stratified recount (CT vs ET on 1m) ====='\n"
        "systemd-run --scope \\\n"
        "  -p MemoryMax=6G -p CPUQuota=200% \\\n"
        f"  python3 {RECOUNT_SCRIPT}\n"
        "echo \"[wrapper1m] finish $(date -u +%FT%TZ)\"\n"
    )
    sftp = c.open_sftp()
    try:
        with sftp.file(f"{REMOTE_ROOT}/{WRAPPER_PATH}", "w") as f:
            f.write(wrapper)
        sftp.chmod(f"{REMOTE_ROOT}/{WRAPPER_PATH}", 0o755)
    finally:
        sftp.close()
    print(f"[orch1m] wrapper written.", flush=True)


def _launch(c: paramiko.SSHClient) -> None:
    print(f"[orch1m] launching screen '{SCREEN_SESSION}' ...", flush=True)
    out, _, _ = _run(
        c,
        f"cd {REMOTE_ROOT} && mkdir -p logs && rm -f {LOG_PATH} && "
        f"screen -dmS {SCREEN_SESSION} bash -c "
        f"\"./{WRAPPER_PATH} > {LOG_PATH} 2>&1\" && sleep 5",
        timeout=30,
    )
    print(out, flush=True)
    out, _, _ = _run(c, f"screen -ls; echo '---LOG---'; tail -30 {REMOTE_ROOT}/{LOG_PATH}")
    print(out, flush=True)


def cmd_launch() -> None:
    c = _connect()
    try:
        _git_sync(c)
        _write_wrapper(c)
        _launch(c)
        print("", flush=True)
        print("=" * 68, flush=True)
        print(" 1m LAUNCH COMPLETE", flush=True)
        print("=" * 68, flush=True)
        print(f" screen: {SCREEN_SESSION}", flush=True)
        print(f" log:    {LOG_PATH} (on remote)", flush=True)
        print(f" wall-clock: ~1-3h estimate (16384 combos × 3 workers on 1m data)", flush=True)
        print(f" poll:   python3 tasks/_run_probe1_1m_audit_remote.py --mode poll", flush=True)
        print(f" pull:   python3 tasks/_run_probe1_1m_audit_remote.py --mode pull", flush=True)
    finally:
        c.close()


def cmd_poll() -> None:
    c = _connect()
    try:
        cmd = (
            "screen -ls; echo '---LOG---'; "
            f"tail -50 {REMOTE_ROOT}/{LOG_PATH}; "
            "echo '---MEM---'; free -h | head -2; "
            "echo '---PROC---'; "
            "ps -eo pid,rss,pcpu,etime,cmd --sort=-rss | head -6"
        )
        out, _, _ = _run(c, cmd, timeout=30)
        print(out, flush=True)
    finally:
        c.close()


def cmd_pull() -> None:
    c = _connect()
    try:
        local_dir = REPO / "data" / "ml" / "probe1_audit"
        local_dir.mkdir(parents=True, exist_ok=True)
        sftp = c.open_sftp()
        try:
            for rel in (
                RECOUNT_JSON,
                ET_SWEEP_OUTPUT.replace(".parquet", "_manifest.json"),
                ET_SWEEP_OUTPUT,  # big, pull last
            ):
                remote_path = f"{REMOTE_ROOT}/{rel}"
                local_path = REPO / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[orch1m] pull {remote_path} -> {local_path}", flush=True)
                try:
                    sftp.get(remote_path, str(local_path))
                    print(f"[orch1m]   ok", flush=True)
                except FileNotFoundError:
                    print(f"[orch1m]   MISSING on remote (may not be done yet)", flush=True)
        finally:
            sftp.close()
    finally:
        c.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["launch", "poll", "pull"], required=True)
    args = ap.parse_args()
    if args.mode == "launch":
        cmd_launch()
    elif args.mode == "poll":
        cmd_poll()
    elif args.mode == "pull":
        cmd_pull()


if __name__ == "__main__":
    main()
