"""Pull executed evaluation notebooks from sweep-runner-1 via SFTP.

The git-based workflow is pending: `~/.git-credentials` on the VM is empty,
so the remote side can't `git push`. Until a PAT or SSH deploy key is
installed there, we transfer the two artifacts over SFTP and let the local
side commit + push.
"""
from __future__ import annotations
from pathlib import Path
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
REPO = Path(__file__).resolve().parent.parent.parent

FILES = [
    "evaluation/v10_topk/s1_individual.ipynb",
    "evaluation/v10_topk/s2_combined.ipynb",
    "evaluation/v10_topk/s3_mc_combined.ipynb",
    "evaluation/v10_topk/s4_individual_ml2.ipynb",
    "evaluation/v10_topk/s5_combined_ml2.ipynb",
    "evaluation/v10_topk/s6_mc_combined_ml2.ipynb",
    "evaluation/v10_topk_net/s1_individual_net.ipynb",
    "evaluation/v10_topk_net/s2_combined_net.ipynb",
    "evaluation/v10_topk_net/s3_mc_combined_net.ipynb",
    "evaluation/v10_topk_net/s4_individual_ml2_net.ipynb",
    "evaluation/v10_topk_net/s5_combined_ml2_net.ipynb",
    "evaluation/v10_topk_net/s6_mc_combined_ml2_net.ipynb",
    "evaluation/top_trade_log.xlsx",
]


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()
    for rel in FILES:
        local = REPO / rel; remote = f"{REMOTE_DIR}/{rel}"
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {rel} ...", end=" ", flush=True)
        sftp.get(remote, str(local))
        print(f"{local.stat().st_size:,} B")
    sftp.close(); ssh.close()


if __name__ == "__main__":
    main()
