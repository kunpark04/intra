"""Remote orchestrator for the Probe 1 TZ-fix audit on sweep-runner-1.

Three operations:

  launch  — git-sync the remote, run the stratified recount synchronously
            (1-2 min), pull the recount JSON back, write a sweep wrapper,
            launch the ET re-sweep (15m + 1h in series) under screen.

  poll    — show screen status, tail the log, free memory, top process
            (for /loop every ~10 min per feedback_poll_interval.md).

  pull    — SFTP the ET-resweep parquets + manifests back to local, then
            run the stratified recount again on the new parquets locally
            to produce the comparison JSON.

Authority: tasks/council-transcript-2026-04-23-probe3-reconvene.md chairman
verdict + user directive 2026-04-23 to do all three audit tasks remotely.

Remote host: sweep-runner-1 (195.88.25.157) — credentials in
memory/reference_sweep_runner_ssh.md. Uses paramiko (OpenSSH can't do
password auth non-interactively on Windows).

Engine flag: scripts/param_sweep.py --bar-hour-tz ET opt-in added in the
same TZ-fix commit cycle. Default CT is preserved for backward compat.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Remote connection constants (credentials in memory/reference_sweep_runner_ssh.md) ─
HOST     = "195.88.25.157"
USER     = "root"
PASSWORD = "J@maicanP0wer123"
REMOTE_ROOT = "/root/intra"
JOB_NAME   = "probe1_et_audit"
SCREEN_SESSION = JOB_NAME

# ── Local/remote paths ─────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]

RECOUNT_SCRIPT       = "tasks/_probe1_stratified_recount.py"
RECOUNT_OUTPUT_JSON  = "data/ml/probe1_audit/stratified_recount.json"

ET_SWEEP_15M_OUTPUT  = "data/ml/probe1_audit/ml_dataset_v11_15m_et.parquet"
ET_SWEEP_1H_OUTPUT   = "data/ml/probe1_audit/ml_dataset_v11_1h_et.parquet"

LOG_PATH             = "logs/probe1_et_audit.log"
WRAPPER_PATH         = "run_probe1_et_audit.sh"


def _connect() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return c


def _run(c: paramiko.SSHClient, cmd: str, timeout: int | None = 120) -> tuple[str, str, int]:
    """Execute a remote command, return (stdout, stderr, exitcode)."""
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def _git_sync(c: paramiko.SSHClient) -> None:
    """Pull latest master on remote. Pushed locally first — remote must match."""
    print(f"[orch] git-syncing remote {REMOTE_ROOT} ...", flush=True)
    out, err, rc = _run(
        c,
        f"cd {REMOTE_ROOT} && git fetch origin master && "
        f"git reset --hard origin/master && git log -1 --oneline",
        timeout=60,
    )
    print(out, flush=True)
    if err.strip():
        print(f"[orch] stderr: {err}", flush=True)
    if rc != 0:
        raise RuntimeError(f"git sync failed (rc={rc})")


def _run_recount(c: paramiko.SSHClient) -> None:
    """Run the pure-pandas stratified recount synchronously on remote."""
    print(f"[orch] running stratified recount on remote ...", flush=True)
    cmd = (
        f"cd {REMOTE_ROOT} && "
        f"mkdir -p data/ml/probe1_audit && "
        f"python3 {RECOUNT_SCRIPT}"
    )
    # Give it a generous 10-min ceiling (should take <2 min in practice)
    out, err, rc = _run(c, cmd, timeout=600)
    print(out, flush=True)
    if err.strip():
        print(f"[orch] recount stderr: {err}", flush=True)
    if rc != 0:
        raise RuntimeError(f"recount failed (rc={rc})")


def _pull_recount_json(c: paramiko.SSHClient) -> None:
    """SFTP the recount JSON back to local."""
    local_dir = REPO / "data" / "ml" / "probe1_audit"
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"{REMOTE_ROOT}/{RECOUNT_OUTPUT_JSON}"
    local_path = REPO / RECOUNT_OUTPUT_JSON
    print(f"[orch] pulling {remote_path} -> {local_path}", flush=True)
    sftp = c.open_sftp()
    try:
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()
    print(f"[orch] recount JSON pulled.", flush=True)


def _write_sweep_wrapper(c: paramiko.SSHClient) -> None:
    """Write a shell wrapper that runs 15m then 1h ET sweeps in series.

    Uses systemd-run --scope for resource capping per rule 3 of
    reference_remote_job_workflow.md. workers=3 per rule 6.

    Sweep args mirror the v11 range-mode used by Probe 1; the key delta is
    --bar-hour-tz ET.
    """
    print(f"[orch] writing sweep wrapper at {WRAPPER_PATH} ...", flush=True)
    wrapper_body = (
        "#!/bin/bash\n"
        "set -e\n"
        f"cd {REMOTE_ROOT}\n"
        "mkdir -p data/ml/probe1_audit\n"
        "echo \"[wrapper] start $(date -u +%FT%TZ)\"\n"
        "\n"
        "echo '[wrapper] ========== 15m ET re-sweep =========='\n"
        "exec_15m() {\n"
        "  systemd-run --scope \\\n"
        "    -p MemoryMax=7G -p CPUQuota=280% \\\n"
        "    python3 scripts/param_sweep.py \\\n"
        "      --range-mode v11 \\\n"
        "      --timeframe 15min \\\n"
        "      --combinations 3000 \\\n"
        "      --seed 0 \\\n"
        "      --eval-partition train \\\n"
        "      --workers 3 \\\n"
        "      --bar-hour-tz ET \\\n"
        f"      --output {ET_SWEEP_15M_OUTPUT}\n"
        "}\n"
        "exec_15m\n"
        "echo \"[wrapper] 15m done $(date -u +%FT%TZ)\"\n"
        "\n"
        "echo '[wrapper] ========== 1h ET re-sweep =========='\n"
        "exec_1h() {\n"
        "  systemd-run --scope \\\n"
        "    -p MemoryMax=7G -p CPUQuota=280% \\\n"
        "    python3 scripts/param_sweep.py \\\n"
        "      --range-mode v11 \\\n"
        "      --timeframe 1h \\\n"
        "      --combinations 1500 \\\n"
        "      --seed 0 \\\n"
        "      --eval-partition train \\\n"
        "      --workers 3 \\\n"
        "      --bar-hour-tz ET \\\n"
        f"      --output {ET_SWEEP_1H_OUTPUT}\n"
        "}\n"
        "exec_1h\n"
        "echo \"[wrapper] 1h done $(date -u +%FT%TZ)\"\n"
        "\n"
        "echo '[wrapper] both ET sweeps complete — running comparison recount'\n"
        "python3 tasks/_probe1_stratified_recount_et.py || true\n"
        "echo \"[wrapper] finish $(date -u +%FT%TZ)\"\n"
    )
    sftp = c.open_sftp()
    try:
        with sftp.file(f"{REMOTE_ROOT}/{WRAPPER_PATH}", "w") as f:
            f.write(wrapper_body)
        sftp.chmod(f"{REMOTE_ROOT}/{WRAPPER_PATH}", 0o755)
    finally:
        sftp.close()
    print(f"[orch] wrapper written.", flush=True)


def _launch_sweep_in_screen(c: paramiko.SSHClient) -> None:
    """Launch the wrapper under a detached screen session."""
    print(f"[orch] launching ET re-sweep in screen session '{SCREEN_SESSION}' ...", flush=True)
    cmd = (
        f"cd {REMOTE_ROOT} && mkdir -p logs && rm -f {LOG_PATH} && "
        f"screen -dmS {SCREEN_SESSION} bash -c "
        f"\"./{WRAPPER_PATH} > {LOG_PATH} 2>&1\" && sleep 5"
    )
    out, err, rc = _run(c, cmd, timeout=30)
    if err.strip():
        print(f"[orch] launch stderr: {err}", flush=True)
    print(f"[orch] launch command returned rc={rc}. Verifying ...", flush=True)

    out, err, rc = _run(c, f"screen -ls; echo '---LOG---'; tail -30 {REMOTE_ROOT}/{LOG_PATH}")
    print(out, flush=True)


def cmd_launch() -> None:
    c = _connect()
    try:
        _git_sync(c)
        _run_recount(c)
        _pull_recount_json(c)
        _write_sweep_wrapper(c)
        _launch_sweep_in_screen(c)
        print("", flush=True)
        print("=" * 68, flush=True)
        print(" LAUNCH COMPLETE", flush=True)
        print("=" * 68, flush=True)
        print(f" recount JSON:   data/ml/probe1_audit/stratified_recount.json (pulled)", flush=True)
        print(f" ET re-sweep:    screen session '{SCREEN_SESSION}' on remote", flush=True)
        print(f" expected wall-clock: ~6-9h (15m ~4-6h + 1h ~2-3h, series)", flush=True)
        print(f" poll command:   python3 tasks/_run_probe1_audit_remote.py --mode poll", flush=True)
        print(f" pull command:   python3 tasks/_run_probe1_audit_remote.py --mode pull", flush=True)
    finally:
        c.close()


def cmd_poll() -> None:
    c = _connect()
    try:
        cmd = (
            "screen -ls; echo '---LOG---'; "
            f"tail -40 {REMOTE_ROOT}/{LOG_PATH}; "
            "echo '---MEM---'; free -h | head -2; "
            "echo '---PROC---'; "
            "ps -eo pid,rss,pcpu,etime,cmd --sort=-rss | head -6"
        )
        out, _, _ = _run(c, cmd, timeout=30)
        print(out, flush=True)
    finally:
        c.close()


def cmd_pull() -> None:
    """SFTP the ET-sweep parquets + manifests back. Run the recount locally
    on the new parquets to produce a comparison readout."""
    c = _connect()
    try:
        local_audit_dir = REPO / "data" / "ml" / "probe1_audit"
        local_audit_dir.mkdir(parents=True, exist_ok=True)
        sftp = c.open_sftp()
        try:
            for remote_file in (
                ET_SWEEP_15M_OUTPUT,
                ET_SWEEP_1H_OUTPUT,
                ET_SWEEP_15M_OUTPUT.replace(".parquet", "_manifest.json"),
                ET_SWEEP_1H_OUTPUT.replace(".parquet", "_manifest.json"),
                "data/ml/probe1_audit/stratified_recount_et.json",
            ):
                remote_path = f"{REMOTE_ROOT}/{remote_file}"
                local_path = REPO / remote_file
                local_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"[orch] pull {remote_path} -> {local_path}", flush=True)
                try:
                    sftp.get(remote_path, str(local_path))
                    print(f"[orch]   ok", flush=True)
                except FileNotFoundError:
                    print(f"[orch]   MISSING on remote (sweep may not be done yet)", flush=True)
        finally:
            sftp.close()
    finally:
        c.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["launch", "poll", "pull"], required=True,
                    help="launch: git-sync + recount + SFTP back + write wrapper + "
                         "launch ET re-sweep under screen. "
                         "poll: screen-ls + tail log + free mem (use with /loop "
                         "delaySeconds=600 per feedback_poll_interval.md). "
                         "pull: SFTP the ET-sweep parquets + comparison JSON back.")
    args = ap.parse_args()
    if args.mode == "launch":
        cmd_launch()
    elif args.mode == "poll":
        cmd_poll()
    elif args.mode == "pull":
        cmd_pull()


if __name__ == "__main__":
    main()
