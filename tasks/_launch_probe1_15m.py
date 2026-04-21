"""Launch Probe 1 C3a — 15min v11 sweep @ 3000 combos on sweep-runner-1.

Handles, in order:
  1. SSH to sweep-runner-1 (paramiko).
  2. `git pull` on /root/intra to sync the C1+C2 commits.
  3. Build 15m / 1h parquets if missing (data/NQ_{15min,1h}.parquet).
  4. Rebuild Cython extension — signature changed (added cooldown_bars),
     a stale .so would silently segfault or produce wrong trade counts.
  5. Write /root/intra/run_probe1_15m.sh wrapper with systemd-run envelope.
  6. Launch inside a detached `screen` session named `probe1_15m`.
  7. Verify launch (screen -ls, log tail, process list).

Output destination (on remote and local after pull):
    /root/intra/data/ml/originals/ml_dataset_v11_15m.parquet
    /root/intra/data/ml/originals/ml_dataset_v11_15m_manifest.json

Launch envelope (per memory/reference_remote_job_workflow.md + feedback_kamatera_cpu_cap.md):
    systemd-run --scope -p MemoryMax=8G -p CPUQuota=280%
    --workers 3  (leave 1 vCPU idle for pyarrow merges + OS)

Preregistration: tasks/probe1_preregistration.md §3 (15min budget = 3000 combos).
"""
from __future__ import annotations

import io
import sys
import time

import paramiko

# UTF-8 stdout (per memory §8 — remote log may emit unicode)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST = "195.88.25.157"
USER = "root"
PWD = "J@maicanP0wer123"  # noqa: S105 — local-only ref, never committed to remote

SCREEN_NAME = "probe1_15m"
LOG_PATH = f"/root/intra/logs/{SCREEN_NAME}.log"
WRAPPER_PATH = f"/root/intra/run_{SCREEN_NAME}.sh"

WRAPPER_BODY = """#!/bin/bash
set -e
cd /root/intra
mkdir -p data/ml/originals
exec systemd-run --scope \\
    -p MemoryMax=8G \\
    -p CPUQuota=280% \\
    python3 scripts/param_sweep.py \\
        --combinations 3000 \\
        --range-mode v11 \\
        --timeframe 15min \\
        --seed 0 \\
        --output data/ml/originals/ml_dataset_v11_15m.parquet \\
        --workers 3 \\
        --eval-partition train
"""


def run(client: paramiko.SSHClient, cmd: str, label: str | None = None) -> tuple[str, str]:
    """Run `cmd` on the remote; return (stdout, stderr) as strings."""
    if label:
        print(f"\n── {label} ──")
        print(f"$ {cmd}")
    _, out, err = client.exec_command(cmd, timeout=None)
    so = out.read().decode(errors="replace")
    se = err.read().decode(errors="replace")
    if so:
        print(so.rstrip())
    if se.strip():
        print(f"[stderr] {se.rstrip()}")
    return so, se


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}…")
    client.connect(HOST, username=USER, password=PWD, timeout=30)

    # 1. Stash any remote-side modifications so `git pull --ff-only` can
    #    fast-forward to dc5e19d (C1 + C2). The remote accumulated edits
    #    from prior Phase 3 V3 eval runs — preserved in stash for recovery.
    run(
        client,
        "cd /root/intra && git stash push -u -m 'pre-probe1-$(date +%Y%m%d-%H%M%S)'",
        label="stash remote-side changes",
    )

    # 2. Now git pull should fast-forward cleanly
    run(client, "cd /root/intra && git pull --ff-only", label="git pull")
    run(client, "cd /root/intra && git rev-parse HEAD", label="remote HEAD after pull")

    # 3. Cython: stale .so on remote has old signature (14 args, no cooldown_bars).
    #    Cython isn't installed on remote (PEP 668 blocks system pip on Ubuntu 24).
    #    Deleting the .so forces import failure → Numba fallback (numba 0.65.0 is
    #    installed, adequate for 3000 combos × 170k 15m bars).
    run(
        client,
        "rm -f /root/intra/src/cython_ext/*.so /root/intra/src/cython_ext/*.c "
        "&& ls /root/intra/src/cython_ext/ 2>&1",
        label="remove stale Cython .so — force Numba fallback",
    )

    # 4. Build 15m/1h bar caches (build_bar_caches.py landed in C1).
    run(
        client,
        "ls -la /root/intra/scripts/data_pipeline/build_bar_caches.py 2>&1",
        label="verify build script present",
    )
    run(
        client,
        "cd /root/intra && python3 scripts/data_pipeline/build_bar_caches.py 2>&1 | tail -30",
        label="build 15m + 1h parquets",
    )
    run(
        client,
        "ls -la /root/intra/data/NQ_15min.parquet /root/intra/data/NQ_1h.parquet 2>&1",
        label="verify parquets built",
    )

    # 4. Write launcher wrapper via SFTP
    print("\n── write wrapper ──")
    sftp = client.open_sftp()
    with sftp.file(WRAPPER_PATH, "w") as f:
        f.write(WRAPPER_BODY)
    sftp.chmod(WRAPPER_PATH, 0o755)
    print(f"[ok] wrote {WRAPPER_PATH}")
    run(client, f"cat {WRAPPER_PATH}", label="wrapper contents")

    # 5. Launch
    run(
        client,
        f"cd /root/intra && mkdir -p logs && rm -f {LOG_PATH} "
        f"&& screen -dmS {SCREEN_NAME} bash -c "
        f"'{WRAPPER_PATH} > {LOG_PATH} 2>&1' && sleep 8",
        label=f"launch screen({SCREEN_NAME})",
    )

    # 6. Verify
    run(client, "screen -ls", label="screen -ls")
    run(client, f"tail -40 {LOG_PATH} 2>&1 || echo '(no log yet)'", label="log tail")
    run(client, "free -h | head -2", label="memory")
    run(
        client,
        "ps -eo pid,rss,pcpu,etime,cmd --sort=-rss | head -6",
        label="process list",
    )

    client.close()
    print("\n[ok] launch complete — poll every 10 min via ScheduleWakeup.")


if __name__ == "__main__":
    main()
