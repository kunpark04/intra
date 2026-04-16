"""Quick polling helper for the ml1_c1 screen session."""
from __future__ import annotations
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    cmds = [
        ("screen -ls", "screen sessions"),
        ("ls -la /tmp/ml1c1_stage1.log /tmp/ml1c1_stage2.log 2>/dev/null",
         "log files"),
        ("tail -n 30 /tmp/ml1c1_stage1.log 2>/dev/null", "stage1 tail"),
        ("tail -n 30 /tmp/ml1c1_stage2.log 2>/dev/null", "stage2 tail"),
        ("ls -la /root/intra/data/ml/full_combo_r_stats.parquet "
         "/root/intra/data/ml/ml1_results_c1/ 2>/dev/null", "artifacts"),
        ("free -h | head -2", "memory"),
    ]
    for cmd, label in cmds:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode(errors="replace").rstrip()
        print(f"\n--- {label} ---\n{out}")
    ssh.close()


if __name__ == "__main__":
    main()
