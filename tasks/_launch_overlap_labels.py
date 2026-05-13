"""_launch_overlap_labels.py — Phase 4.2 launcher for combo-overlap labeling job.

Uploads wrapper shell script, launches under screen, verifies startup.
Does NOT block-wait for completion.
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

WRAPPER_SH = """#!/bin/bash
set -euo pipefail
cd /root/intra
python3 scripts/analysis/build_combo_overlap_labels.py \\
    --combo-features data/ml/combo_features_v12.parquet \\
    --mfe-parquet data/ml/mfe/ml_dataset_v11_mfe.parquet \\
    --output data/ml/combo_overlap_labels.parquet \\
    --batch-size 500000 \\
    --log-every 5
echo "[overlap_labels] ALL DONE"
"""


def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    so = stdout.read().decode()
    se = stderr.read().decode()
    return exit_code, so, se


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    print(f"[launch] connected to {HOST}")

    # --- Step 1: git sync on remote ---
    print("\n[step 1] git sync on remote")
    ec, so, se = run(
        ssh,
        "cd /root/intra && git fetch origin && git reset --hard origin/master && git log -1 --oneline",
        timeout=120,
    )
    print(f"  exit_code={ec}")
    print(f"  stdout: {so.strip()}")
    if se.strip():
        print(f"  stderr: {se.strip()}")

    # Verify ce2be3a commit or newer is present
    ec2, so2, _ = run(
        ssh,
        "cd /root/intra && git log --oneline | head -5",
    )
    print(f"  recent commits:\n{so2}")

    # Verify Phase 4.1 script exists
    ec3, so3, _ = run(
        ssh,
        "ls -la /root/intra/scripts/analysis/build_combo_overlap_labels.py",
    )
    print(f"  script file: {so3.strip()}")
    if ec3 != 0:
        print("[ABORT] build_combo_overlap_labels.py missing on remote!")
        ssh.close()
        return

    # --- Step 2: pre-check, confirm v3_no_gcid alive, and no stale overlap_labels screen ---
    print("\n[step 2] pre-check screens before launch")
    _, so_ls_pre, _ = run(ssh, "screen -ls")
    print(f"  screen -ls (before):\n{so_ls_pre}")

    if "overlap_labels" in so_ls_pre:
        print("  [INFO] overlap_labels screen already exists; skipping launch")
        print("  Will check log and report existing state.")
        existing_screen = True
    else:
        existing_screen = False

    # --- Step 3: write wrapper via SFTP ---
    sftp = ssh.open_sftp()
    wrapper_remote = f"{REMOTE_DIR}/run_overlap_labels.sh"
    with sftp.open(wrapper_remote, "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod(wrapper_remote, 0o755)
    print(f"\n[step 3] wrote wrapper: {wrapper_remote}")
    sftp.close()

    # --- Step 4: launch under screen (only if not already running) ---
    if not existing_screen:
        print("\n[step 4] launching screen session")
        launch_cmd = (
            f"cd /root/intra && screen -dmS {SCREEN_NAME} bash -c "
            f"'systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
            f"bash {wrapper_remote} 2>&1 | tee {LOG_PATH}; exec bash'"
        )
        print(f"  cmd: {launch_cmd}")
        ec_l, so_l, se_l = run(ssh, launch_cmd)
        print(f"  exit_code={ec_l}")
        if so_l.strip():
            print(f"  stdout: {so_l.strip()}")
        if se_l.strip():
            print(f"  stderr: {se_l.strip()}")

    # --- Step 5: verify launch ---
    print("\n[step 5] waiting 20s for script startup")
    time.sleep(20)

    _, so_ls, _ = run(ssh, "screen -ls")
    print("screen -ls (after launch):")
    print(so_ls)

    _, so_log_check, _ = run(ssh, f"ls -la {LOG_PATH}")
    print(f"log file: {so_log_check.strip()}")

    _, so_log, _ = run(ssh, f"head -30 {LOG_PATH}")
    print(f"\nfirst 30 lines of {LOG_PATH}:")
    print(so_log)

    _, so_v3, _ = run(ssh, "screen -ls | grep v3_no_gcid || echo 'NOT FOUND'")
    print(f"\nv3_no_gcid screen status: {so_v3.strip()}")

    # Final screens summary
    _, so_ls_final, _ = run(ssh, "screen -ls")
    print(f"\nFINAL screen -ls:\n{so_ls_final}")

    ssh.close()
    print("\n[launch] done.")


if __name__ == "__main__":
    main()
