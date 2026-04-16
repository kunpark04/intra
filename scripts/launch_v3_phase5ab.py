"""Upload and launch Phase 5A+5B on sweep-runner-1.

Step 1 (5B): export_calibrators_v3.py — fit + serialize per-combo isotonic knots
Step 2 (5A): filter_backtest_reopt_v3.py — re-optimize thresholds with two-stage calibrator

Both run in a single screen session sequentially.
"""
from __future__ import annotations
import time
from pathlib import Path
import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"

REPO = Path(__file__).resolve().parent.parent

# Files to upload.
UPLOAD_FILES = [
    "scripts/export_calibrators_v3.py",
    "scripts/filter_backtest_reopt_v3.py",
    "scripts/v3_inference.py",  # updated with two-stage functions
]


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    # Upload files.
    for rel in UPLOAD_FILES:
        local = REPO / rel
        remote = f"{REMOTE_DIR}/{rel}"
        print(f"  Uploading {rel}...", end=" ", flush=True)
        sftp.put(str(local), remote)
        print("OK")

    # Write shell wrapper that runs both phases sequentially.
    wrapper = f"""\
#!/usr/bin/env bash
set -e
cd {REMOTE_DIR}
echo "[phase5] Starting Phase 5B: export calibrators $(date)"
python3 scripts/export_calibrators_v3.py 2>&1 | tee /tmp/phase5b.log
echo "[phase5] Phase 5B complete $(date)"
echo "[phase5] Starting Phase 5A: filter backtest reopt $(date)"
python3 scripts/filter_backtest_reopt_v3.py 2>&1 | tee /tmp/phase5a.log
echo "[phase5] Phase 5A complete $(date)"
echo "[phase5] ALL DONE $(date)"
"""
    with sftp.open(f"{REMOTE_DIR}/run_phase5ab.sh", "w") as f:
        f.write(wrapper)
    sftp.chmod(f"{REMOTE_DIR}/run_phase5ab.sh", 0o755)
    print("  Wrote run_phase5ab.sh")

    # Launch in screen with resource limits.
    cmd = (
        f"screen -dmS v3_phase5ab bash -c '"
        f"systemd-run --scope -p MemoryMax=7G -p CPUQuota=400% "
        f"bash {REMOTE_DIR}/run_phase5ab.sh; exec bash'"
    )
    print(f"  Launching: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    print(f"  screen launch exit code: {exit_code}")

    # Verify screen is running.
    time.sleep(1)
    stdin, stdout, stderr = ssh.exec_command("screen -ls")
    print(stdout.read().decode())

    sftp.close()
    ssh.close()
    print("Done. Poll with: screen -r v3_phase5ab")


if __name__ == "__main__":
    main()
