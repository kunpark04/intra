"""One-shot launcher for Phase 4.2 overlap-labels job on sweep-runner-1.

Steps (all via paramiko exec_command, blocking):
  1. git fetch + reset --hard origin/master, print HEAD
  2. Pre-flight ls -la on the two expected input parquets
  3. Write /root/intra/run_overlap_labels.sh wrapper (systemd-run scope)
  4. chmod +x + launch under screen name 'overlap_labels'
  5. Sleep 15s, then verify screen is Detached, head -40 log, confirm
     v3_no_gcid screen still alive.

The script is intentionally one-shot and prints every checkpoint so the
caller can render a report without re-running anything.
"""
from __future__ import annotations

import time

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
SCREEN_NAME = "overlap_labels"
LOG_PATH = "/tmp/overlap_labels.log"
WRAPPER_REMOTE = f"{REMOTE_DIR}/run_overlap_labels.sh"

WRAPPER_SH = """#!/bin/bash
set -euo pipefail
cd /root/intra
systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% \\
    python3 scripts/analysis/build_combo_overlap_labels.py \\
        --batch-size 500000 \\
        --log-every 5
echo "[overlap_labels] ALL DONE"
"""


def run(ssh: paramiko.SSHClient, cmd: str, *, label: str) -> tuple[int, str, str]:
    """Blocking exec_command, returning (exit_code, stdout, stderr)."""
    print(f"\n=== {label} ===")
    print(f"$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    so = stdout.read().decode()
    se = stderr.read().decode()
    if so:
        print(so.rstrip())
    if se:
        print(f"[stderr] {se.rstrip()}")
    print(f"[exit_code] {exit_code}")
    return exit_code, so, se


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    print(f"[connected] {HOST}")

    # --- Pre-check: overlap_labels screen must NOT already exist ---
    ec, so, _ = run(ssh, "screen -ls | grep overlap_labels || true", label="pre-check: duplicate screen")
    if "overlap_labels" in so:
        print("\n[ABORT] overlap_labels screen already exists — refusing to spawn duplicate.")
        ssh.close()
        return

    # --- Step 1: git sync ---
    ec, so, _ = run(
        ssh,
        "cd /root/intra && git fetch origin && git reset --hard origin/master && git log -1 --oneline",
        label="Step 1: git sync",
    )
    if ec != 0:
        print("\n[ABORT] git sync failed")
        ssh.close()
        return
    head_line = so.strip().splitlines()[-1] if so.strip() else ""
    print(f"[head_line] {head_line}")

    # --- Step 2: pre-flight input parquets ---
    run(
        ssh,
        "ls -la /root/intra/data/ml/ml1_results_v12/combo_features_v12.parquet",
        label="Step 2a: combo_features_v12.parquet",
    )
    run(
        ssh,
        "ls -la /root/intra/data/ml/originals/ml_dataset_v11.parquet",
        label="Step 2b: ml_dataset_v11.parquet",
    )

    # --- Step 3: write wrapper ---
    sftp = ssh.open_sftp()
    with sftp.open(WRAPPER_REMOTE, "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod(WRAPPER_REMOTE, 0o755)
    sftp.close()
    print(f"\n[wrote] {WRAPPER_REMOTE}")
    run(ssh, f"cat {WRAPPER_REMOTE}", label="Step 3: wrapper content")

    # --- Step 4: launch under screen ---
    launch_cmd = (
        f"cd /root/intra && "
        f"screen -dmS {SCREEN_NAME} bash -c "
        f"'bash {WRAPPER_REMOTE} 2>&1 | tee {LOG_PATH}'"
    )
    run(ssh, launch_cmd, label="Step 4: screen launch")

    # --- Step 5: verify after 15s ---
    print("\n[sleep] 15s before verify")
    time.sleep(15)

    run(ssh, "screen -ls", label="Step 5a: screen -ls (all sessions)")
    run(ssh, f"head -40 {LOG_PATH} 2>&1 || echo '[log missing]'", label="Step 5b: log head")
    run(ssh, "screen -ls | grep v3_no_gcid || echo '[MISSING v3_no_gcid]'", label="Step 5c: v3_no_gcid alive")

    ssh.close()
    print("\n[done] launcher exiting")


if __name__ == "__main__":
    main()
