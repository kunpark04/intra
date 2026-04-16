"""Upload and launch Phase 5D (B16 final held-out eval) on sweep-runner-1.

Output artifact: data/ml/adaptive_rr_v3/b16_final_eval.json
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

UPLOAD_FILES = [
    "scripts/final_holdout_eval_v3.py",
    "scripts/inference_v3.py",
]


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

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
echo "[phase5d] Starting B16 final held-out eval $(date)"
python3 scripts/final_holdout_eval_v3.py 2>&1 | tee /tmp/phase5d.log
echo "[phase5d] ALL DONE $(date)"
"""
    with sftp.open(f"{REMOTE_DIR}/run_phase5d.sh", "w") as f:
        f.write(wrapper)
    sftp.chmod(f"{REMOTE_DIR}/run_phase5d.sh", 0o755)
    print("  Wrote run_phase5d.sh")

    cmd = (
        f"screen -dmS v3_phase5d bash -c '"
        f"systemd-run --scope -p MemoryMax=7G -p CPUQuota=400% "
        f"bash {REMOTE_DIR}/run_phase5d.sh; exec bash'"
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
    print("Done. Poll with: screen -r v3_phase5d")


if __name__ == "__main__":
    main()
