"""Pull ML#1 v11 retrain artifacts from sweep-runner-1 via SFTP."""
from __future__ import annotations
from pathlib import Path
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
REPO = Path(__file__).resolve().parent.parent.parent

FILES = [
    "data/ml/ml1_results_v11/combo_features_v11.parquet",
    "data/ml/ml1_results_v11/cv_results.json",
    "data/ml/ml1_results_v11/feature_importance.png",
    "data/ml/ml1_results_v11/models/net_sharpe_point.txt",
    "data/ml/ml1_results_v11/models/net_sharpe_p10.txt",
    "data/ml/ml1_results_v11/models/net_sharpe_p50.txt",
    "data/ml/ml1_results_v11/models/net_sharpe_p90.txt",
    "evaluation/top_strategies_v11.json",
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
        try:
            sftp.get(remote, str(local))
            print(f"{local.stat().st_size:,} B")
        except FileNotFoundError:
            print("MISSING (stage not yet complete?)")
    sftp.close(); ssh.close()


if __name__ == "__main__":
    main()
