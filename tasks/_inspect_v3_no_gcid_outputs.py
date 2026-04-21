"""Inspect what actually ran on remote — file sizes, error output, s6 MC row."""
from __future__ import annotations
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"


def sh(ssh, cmd, timeout=30):
    _, so, se = ssh.exec_command(cmd, timeout=timeout)
    return so.read().decode(errors="replace"), se.read().decode(errors="replace")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)

    out, _ = sh(ssh, "ls -la /root/intra/evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/")
    print(f"[sizes]\n{out}")

    out, _ = sh(ssh, "cat /root/intra/v3_no_gcid.log")
    print(f"[full log]\n{out}")

    # Check a few cell outputs: grep for MC rows or errors
    out, _ = sh(ssh,
        "python3 -c \"import nbformat; nb=nbformat.read("
        "'/root/intra/evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/"
        "s6_mc_combined_ml2_net.ipynb', as_version=4);"
        "print('cells=', len(nb.cells));"
        "[print(i, c.cell_type, (c.source[:80] if isinstance(c.source,str) else '') ,"
        "'outputs=', len(c.get('outputs',[]))) for i,c in enumerate(nb.cells)]\"")
    print(f"[s6 cell structure]\n{out}")

    ssh.close()


if __name__ == "__main__":
    main()
