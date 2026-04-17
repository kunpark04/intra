"""Upload and launch ML#1 v11 retrain on sweep-runner-1.

Three-stage pipeline (sequential in one screen session):
  1. build_combo_features_ml1_v11.py  — streams v10 parquet, computes
     friction-aware target + ~41 features → combo_features_v11.parquet
  2. ml1_surrogate_v11.py             — trains 4 LightGBM boosters
     (point + p10/p50/p90) with 5-fold random CV
  3. extract_top_combos_v11.py        — UCB ranking → top_strategies_v11.json

Reads v10 mfe parquet + manifest that already live on the remote from the
v10 sweep. Writes outputs to /root/intra/data/ml/ml1_results_v11/ and
/root/intra/evaluation/top_strategies_v11.json.

Poll with: screen -r ml1_v11  |  tail -f /tmp/ml1_v11_stage{1,2,3}.log
Pull with: python scripts/runners/_pull_ml1_v11.py
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
    "scripts/analysis/build_combo_features_ml1_v11.py",
    "scripts/models/ml1_surrogate_v11.py",
    "scripts/analysis/extract_top_combos_v11.py",
]


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    ssh.exec_command(
        f"mkdir -p {REMOTE_DIR}/scripts/analysis {REMOTE_DIR}/scripts/models "
        f"{REMOTE_DIR}/data/ml/ml1_results_v11 {REMOTE_DIR}/evaluation"
    )[1].channel.recv_exit_status()

    for rel in UPLOAD_FILES:
        local = REPO / rel
        remote = f"{REMOTE_DIR}/{rel}"
        print(f"  Uploading {rel} ({local.stat().st_size:,} B)...",
              end=" ", flush=True)
        sftp.put(str(local), remote)
        print("OK")

    wrapper = f"""\
#!/usr/bin/env bash
set -e
cd {REMOTE_DIR}
echo "[ml1v11] Starting $(date)"
python3 -m pip install --quiet --break-system-packages lightgbm scikit-learn pyarrow pandas numpy matplotlib 2>&1 | tail -5
echo "[ml1v11] Stage 1: build_combo_features_ml1_v11.py"
python3 scripts/analysis/build_combo_features_ml1_v11.py 2>&1 | tee /tmp/ml1_v11_stage1.log
echo "[ml1v11] Stage 2: ml1_surrogate_v11.py"
python3 scripts/models/ml1_surrogate_v11.py 2>&1 | tee /tmp/ml1_v11_stage2.log
echo "[ml1v11] Stage 3: extract_top_combos_v11.py"
python3 scripts/analysis/extract_top_combos_v11.py 2>&1 | tee /tmp/ml1_v11_stage3.log
echo "[ml1v11] ALL DONE $(date)"
"""
    with sftp.open(f"{REMOTE_DIR}/run_ml1_v11.sh", "w") as f:
        f.write(wrapper)
    sftp.chmod(f"{REMOTE_DIR}/run_ml1_v11.sh", 0o755)
    print("  Wrote run_ml1_v11.sh")

    ssh.exec_command("screen -S ml1_v11 -X quit 2>/dev/null")[1].channel.recv_exit_status()
    time.sleep(1)

    cmd = (
        f"screen -dmS ml1_v11 bash -c '"
        f"systemd-run --scope -p MemoryMax=8G -p CPUQuota=400% "
        f"bash {REMOTE_DIR}/run_ml1_v11.sh 2>&1 | tee /tmp/ml1_v11.log; exec bash'"
    )
    print(f"  Launching: {cmd}")
    _, stdout, _ = ssh.exec_command(cmd)
    print(f"  screen launch exit code: {stdout.channel.recv_exit_status()}")

    time.sleep(1)
    _, stdout, _ = ssh.exec_command("screen -ls")
    print(stdout.read().decode())

    sftp.close()
    ssh.close()
    print("Done. Poll: tail -f /tmp/ml1_v11_stage1.log  |  "
          "Pull: python scripts/runners/_pull_ml1_v11.py")


if __name__ == "__main__":
    main()
