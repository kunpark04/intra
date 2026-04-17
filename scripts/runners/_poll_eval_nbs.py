"""Quick polling helper for the eval_nbs screen session."""
from __future__ import annotations
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    cmds = [
        ("screen -ls", "screen sessions"),
        ("ls -la /tmp/eval_nbs.log 2>/dev/null", "log file"),
        ("tail -n 80 /tmp/eval_nbs.log 2>/dev/null", "eval_nbs tail"),
        ("ls -la /root/intra/evaluation/*.ipynb 2>/dev/null", "notebook artifacts"),
        ("free -h | head -2", "memory"),
    ]
    for cmd, label in cmds:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode(errors="replace").rstrip()
        print(f"\n--- {label} ---\n{out}")
    ssh.close()


if __name__ == "__main__":
    main()
