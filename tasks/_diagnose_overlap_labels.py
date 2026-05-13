"""_diagnose_overlap_labels.py — find where combo_features_v12.parquet lives on remote."""
from __future__ import annotations

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"


def run(ssh, cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(), stderr.read().decode()


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)

    print("=== search for combo_features_v12 on remote ===")
    _, so, _ = run(ssh, "find /root/intra/data -name 'combo_features_v12*' -type f 2>/dev/null")
    print(so)

    print("=== search broader for combo_features ===")
    _, so, _ = run(ssh, "find /root/intra/data -name 'combo_features*' -type f 2>/dev/null")
    print(so)

    print("=== data/ml top-level listing ===")
    _, so, _ = run(ssh, "ls -la /root/intra/data/ml/ | head -40")
    print(so)

    print("=== data/ml/ml1_results_v12/ listing (if exists) ===")
    _, so, _ = run(ssh, "ls -la /root/intra/data/ml/ml1_results_v12/ 2>/dev/null | head -40")
    print(so)

    print("=== MFE parquet check ===")
    _, so, _ = run(ssh, "ls -la /root/intra/data/ml/mfe/ml_dataset_v11_mfe.parquet 2>/dev/null")
    print(so)

    print("=== kill overlap_labels screen ===")
    ec, so, se = run(ssh, "screen -S overlap_labels -X quit")
    print(f"kill exit={ec} stdout={so!r} stderr={se!r}")

    print("=== verify screens post-kill ===")
    _, so, _ = run(ssh, "screen -ls")
    print(so)

    ssh.close()


if __name__ == "__main__":
    main()
