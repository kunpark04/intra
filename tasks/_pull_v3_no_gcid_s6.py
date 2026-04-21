"""Pull executed v3_no_gcid s3-s6 notebooks + extract s6 MC verdict row."""
from __future__ import annotations
import json
from pathlib import Path
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"

REMOTE = "/root/intra/evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid"
LOCAL = Path("evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid")
NB_NAMES = [
    "s3_mc_combined_net.ipynb",
    "s4_individual_ml2_net.ipynb",
    "s5_combined_ml2_net.ipynb",
    "s6_mc_combined_ml2_net.ipynb",
]


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)
    sftp = ssh.open_sftp()

    LOCAL.mkdir(parents=True, exist_ok=True)
    for name in NB_NAMES:
        remote_p = f"{REMOTE}/{name}"
        local_p = LOCAL / name
        sftp.get(remote_p, str(local_p))
        size = local_p.stat().st_size
        print(f"[pulled] {name} -> {size:,} bytes")

    sftp.close()
    ssh.close()

    import nbformat
    s6 = nbformat.read(str(LOCAL / "s6_mc_combined_ml2_net.ipynb"), as_version=4)
    print(f"\n[s6 cell count] {len(s6.cells)}")
    # The s6-mc cell builds an MC rows DataFrame; we need its stdout output
    for i, c in enumerate(s6.cells):
        if c.cell_type == "code" and "monte_carlo" in c.source:
            print(f"\n[cell {i}] source head:\n{c.source[:300]}")
            print(f"\n[cell {i}] outputs:")
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    print(o.get("text", ""))
                elif o.get("output_type") == "execute_result":
                    data = o.get("data", {})
                    if "text/plain" in data:
                        print(data["text/plain"])
                    if "text/html" in data:
                        print("[has text/html]")


if __name__ == "__main__":
    main()
