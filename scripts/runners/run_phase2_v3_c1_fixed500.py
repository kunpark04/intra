"""Upload and launch Phase 2 V3+C1+Fixed$500 OOS eval on sweep-runner-1.

Runs the V3 ML#2 production stack against the C1-selected top-10 combos
(from evaluation/top_strategies.json) on the held-out TEST partition,
with fixed $500 sizing as the primary policy (plus fixed5/kelly comps).

Output: data/ml/adaptive_rr_v3/final_holdout_eval_v3_c1_fixed500.json
"""
from __future__ import annotations
import time
from pathlib import Path
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
REPO = Path(__file__).resolve().parent.parent.parent

UPLOAD_FILES = [
    "scripts/evaluation/final_holdout_eval_v3_c1_fixed500.py",
    "evaluation/top_strategies.json",
]


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    ssh.exec_command(
        f"mkdir -p {REMOTE_DIR}/scripts/evaluation {REMOTE_DIR}/evaluation "
        f"{REMOTE_DIR}/data/ml/adaptive_rr_v3"
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
echo "[phase2] Starting $(date)"
python3 scripts/evaluation/final_holdout_eval_v3_c1_fixed500.py 2>&1 | tee /tmp/phase2.log
echo "[phase2] ALL DONE $(date)"
"""
    with sftp.open(f"{REMOTE_DIR}/run_phase2.sh", "w") as f:
        f.write(wrapper)
    sftp.chmod(f"{REMOTE_DIR}/run_phase2.sh", 0o755)
    print("  Wrote run_phase2.sh")

    # Clean up any stale ml1_c1_s2 / phase2 screens
    for name in ("ml1_c1_s2", "phase2"):
        ssh.exec_command(
            f"screen -S {name} -X quit 2>/dev/null"
        )[1].channel.recv_exit_status()
    time.sleep(1)

    cmd = (
        f"screen -dmS phase2 bash -c '"
        f"systemd-run --scope -p MemoryMax=7G -p CPUQuota=280% "
        f"bash {REMOTE_DIR}/run_phase2.sh; exec bash'"
    )
    print(f"  Launching: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    print(f"  screen launch exit code: {exit_code}")

    time.sleep(1)
    stdin, stdout, stderr = ssh.exec_command("screen -ls")
    print(stdout.read().decode())

    sftp.close(); ssh.close()
    print("Done. Poll with: screen -r phase2  |  tail -f /tmp/phase2.log")


if __name__ == "__main__":
    main()
