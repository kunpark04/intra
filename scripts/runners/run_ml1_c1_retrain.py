"""Upload and launch ML#1 C1-target retrain on sweep-runner-1.

Two-stage pipeline (sequential in one screen session):
    1. build_full_combo_r_stats.py  — scans v2-v10 sweep parquets
    2. ml1_surrogate_c1.py          — retrains LightGBM with C1 target

Outputs written to remote data/ml/full_combo_r_stats.parquet and
data/ml/ml1_results_c1/; the SFTP pull happens in a follow-up step
once the screen session reports ALL DONE.
"""
from __future__ import annotations
import time
from pathlib import Path
import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"

REPO = Path(__file__).resolve().parent.parent.parent

UPLOAD_FILES = [
    "scripts/analysis/build_full_combo_r_stats.py",
    "scripts/models/ml1_surrogate_c1.py",
]


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    ssh.exec_command(
        f"mkdir -p {REMOTE_DIR}/scripts/analysis {REMOTE_DIR}/scripts/models "
        f"{REMOTE_DIR}/data/ml/ml1_results_c1"
    )[1].channel.recv_exit_status()

    for rel in UPLOAD_FILES:
        local = REPO / rel
        remote = f"{REMOTE_DIR}/{rel}"
        print(f"  Uploading {rel}...", end=" ", flush=True)
        sftp.put(str(local), remote)
        print("OK")

    wrapper = f"""\
#!/usr/bin/env bash
set -e
cd {REMOTE_DIR}
echo "[ml1c1] Starting $(date)"
echo "[ml1c1] Stage 1: build_full_combo_r_stats.py"
python3 scripts/analysis/build_full_combo_r_stats.py 2>&1 | tee /tmp/ml1c1_stage1.log
echo "[ml1c1] Stage 2: ml1_surrogate_c1.py"
python3 scripts/models/ml1_surrogate_c1.py 2>&1 | tee /tmp/ml1c1_stage2.log
echo "[ml1c1] ALL DONE $(date)"
"""
    with sftp.open(f"{REMOTE_DIR}/run_ml1_c1.sh", "w") as f:
        f.write(wrapper)
    sftp.chmod(f"{REMOTE_DIR}/run_ml1_c1.sh", 0o755)
    print("  Wrote run_ml1_c1.sh")

    cmd = (
        f"screen -dmS ml1_c1 bash -c '"
        f"systemd-run --scope -p MemoryMax=7G -p CPUQuota=280% "
        f"bash {REMOTE_DIR}/run_ml1_c1.sh; exec bash'"
    )
    print(f"  Launching: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    print(f"  screen launch exit code: {exit_code}")

    time.sleep(1)
    stdin, stdout, stderr = ssh.exec_command("screen -ls")
    print(stdout.read().decode())

    sftp.close()
    ssh.close()
    print("Done. Poll with: screen -r ml1_c1  |  tail -f /tmp/ml1c1_stage1.log")


if __name__ == "__main__":
    main()
