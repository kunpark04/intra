"""Upload and launch Phase 5C (portfolio sim) on sweep-runner-1.

Runs scripts/evaluation/portfolio_sim_v3.py in a screen session with resource limits.
Output artifact: data/ml/adaptive_rr_v3/portfolio_sim_v3.json
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
    "scripts/evaluation/portfolio_sim_v3.py",
    "scripts/models/inference_v3.py",
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
echo "[phase5c] Starting portfolio sim $(date)"
python3 scripts/evaluation/portfolio_sim_v3.py 2>&1 | tee /tmp/phase5c.log
echo "[phase5c] ALL DONE $(date)"
"""
    with sftp.open(f"{REMOTE_DIR}/run_phase5c.sh", "w") as f:
        f.write(wrapper)
    sftp.chmod(f"{REMOTE_DIR}/run_phase5c.sh", 0o755)
    print("  Wrote run_phase5c.sh")

    cmd = (
        f"screen -dmS v3_phase5c bash -c '"
        f"systemd-run --scope -p MemoryMax=7G -p CPUQuota=400% "
        f"bash {REMOTE_DIR}/run_phase5c.sh; exec bash'"
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
    print("Done. Poll with: screen -r v3_phase5c")


if __name__ == "__main__":
    main()
