"""Pull executed v4_no_gcid ship-blocker audit notebooks from sweep-runner-1.

Run after `[eval_nbs] ALL DONE` marker lands in /tmp/eval_nbs.log.
"""
from __future__ import annotations

from pathlib import Path

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
REPO = Path(__file__).resolve().parent.parent

FILES = [
    "evaluation/v12_topk_top50_raw_sharpe_v4_no_gcid/s1_individual.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_v4_no_gcid/s2_combined.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_v4_no_gcid/s3_mc_combined.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_v4_no_gcid/s4_individual_ml2.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_v4_no_gcid/s5_combined_ml2.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_v4_no_gcid/s6_mc_combined_ml2.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_net_v4_no_gcid/s1_individual_net.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_net_v4_no_gcid/s2_combined_net.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_net_v4_no_gcid/s3_mc_combined_net.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_net_v4_no_gcid/s4_individual_ml2_net.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_net_v4_no_gcid/s5_combined_ml2_net.ipynb",
    "evaluation/v12_topk_top50_raw_sharpe_net_v4_no_gcid/s6_mc_combined_ml2_net.ipynb",
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
    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
