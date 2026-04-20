"""run_v4_refit_remote.py — launch V4 refit on pre-2024 partition on sweep-runner-1.

Two-step remote job for Phase 1 of the random-K null test plan
(`C:\\Users\\kunpa\\.claude\\plans\\random-k-null-test-v12.md`):

  Step A: filter v11 parquet → ml_dataset_v11_pre2024.parquet (exit_time < 2024-01-01)
  Step B: retrain V4 booster + isotonic calibrators on that partition
          into data/ml/adaptive_rr_v4_pre2024/

Per `feedback_remote_git_sync.md`, code is not SFTP'd — the runner `git pull`s
on the remote so the working tree matches master.

Launches under screen session `v4_refit` wrapped by
`systemd-run --scope -p MemoryMax=9G -p CPUQuota=280%`, matching the
eval-notebook launcher's resource envelope.

Usage:
    python scripts/runners/run_v4_refit_remote.py
"""
from __future__ import annotations

import time

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
SCREEN_NAME = "v4_refit"
LOG_PATH = "/tmp/v4_refit.log"

WRAPPER_SH = """#!/bin/bash
# v4 refit wrapper — runs on sweep-runner-1 under screen + systemd-run scope.
set -euo pipefail
cd /root/intra

echo "[v4_refit] git pull @ $(date)"
git fetch --all
git reset --hard origin/master

echo "[v4_refit] STEP A: filter v11 parquet to pre-2024"
python3 scripts/data_pipeline/filter_v11_by_time.py \\
    --input  data/ml/originals/ml_dataset_v11.parquet \\
    --output data/ml/originals/ml_dataset_v11_pre2024.parquet \\
    --cutoff 2024-01-01

echo "[v4_refit] STEP B: train V4 on pre-2024 partition"
python3 scripts/models/adaptive_rr_model_v4.py \\
    --input-parquet data/ml/originals/ml_dataset_v11_pre2024.parquet \\
    --output-dir    data/ml/adaptive_rr_v4_pre2024 \\
    --max-rows 10000000 \\
    --target-base-trades 1200000 \\
    --n-folds 5 \\
    --rebuild-cache

echo "[v4_refit] ALL DONE $(date)"
"""


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    print(f"[launch] connected to {HOST}")

    # Write the wrapper script via SFTP
    wrapper_remote = f"{REMOTE_DIR}/run_v4_refit.sh"
    with sftp.open(wrapper_remote, "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod(wrapper_remote, 0o755)
    print(f"[launch] wrote {wrapper_remote}")

    # Kill any stale v4_refit screen session
    ssh.exec_command(f"screen -S {SCREEN_NAME} -X quit 2>/dev/null")[1].channel.recv_exit_status()
    time.sleep(1)

    cmd = (
        f"screen -dmS {SCREEN_NAME} bash -c '"
        f"systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
        f"bash {wrapper_remote} 2>&1 | tee {LOG_PATH}; exec bash'"
    )
    print(f"[launch] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    print(f"[launch] screen launch exit code: {exit_code}")

    time.sleep(2)

    # Verify session is alive
    _, so, _ = ssh.exec_command("screen -ls")
    print("[launch] screen sessions:")
    print(so.read().decode())

    print(f"\n[launch] Monitor via:")
    print(f"  ssh root@{HOST} 'tail -f {LOG_PATH}'")
    print(f"  ssh root@{HOST} 'screen -r {SCREEN_NAME}'")
    print(f"[launch] Completion marker: '[v4_refit] ALL DONE'")

    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
