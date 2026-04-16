"""Pull ml1_results_c1 artifacts from remote."""
from __future__ import annotations
from pathlib import Path
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
REPO = Path(__file__).resolve().parent.parent.parent

FILES = [
    "data/ml/full_combo_r_stats.parquet",
    "data/ml/ml1_results_c1/combo_features_c1.parquet",
    "data/ml/ml1_results_c1/cv_results.json",
    "data/ml/ml1_results_c1/feature_importance.png",
    "data/ml/ml1_results_c1/run_metadata.json",
    "data/ml/ml1_results_c1/top_combos.csv",
    "data/ml/ml1_results_c1/models/composite_score.txt",
]


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()
    for rel in FILES:
        local = REPO / rel
        remote = f"{REMOTE_DIR}/{rel}"
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} ...", end=" ", flush=True)
        sftp.get(remote, str(local))
        print(f"{local.stat().st_size:,} B")
    sftp.close(); ssh.close()


if __name__ == "__main__":
    main()
