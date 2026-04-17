"""Poll sweep-runner-1 for ML#1 v11 job status."""
from __future__ import annotations
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"


def sh(ssh, cmd: str) -> str:
    _, stdout, _ = ssh.exec_command(cmd)
    return stdout.read().decode()


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    print("=== screen -ls ===")
    print(sh(ssh, "screen -ls"))
    print("=== free -h ===")
    print(sh(ssh, "free -h"))
    for stage in (1, 2, 3):
        print(f"=== /tmp/ml1_v11_stage{stage}.log (tail 30) ===")
        print(sh(ssh, f"tail -n 30 /tmp/ml1_v11_stage{stage}.log 2>/dev/null || echo '(not yet written)'"))
    print("=== /tmp/ml1_v11.log (tail 5) ===")
    print(sh(ssh, "tail -n 5 /tmp/ml1_v11.log 2>/dev/null || echo '(not yet written)'"))
    ssh.close()


if __name__ == "__main__":
    main()
