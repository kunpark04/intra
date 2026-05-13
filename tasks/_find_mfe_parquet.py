"""_find_mfe_parquet.py — locate v11 MFE parquet on remote."""
from __future__ import annotations
import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"

def run(ssh, cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(), stderr.read().decode()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=15)

print("=== data/ml/mfe listing ===")
_, so, _ = run(ssh, "ls -la /root/intra/data/ml/mfe/")
print(so)

print("=== data/ml/originals listing ===")
_, so, _ = run(ssh, "ls -la /root/intra/data/ml/originals/ | head -30")
print(so)

print("=== search for v11 parquets ===")
_, so, _ = run(ssh, "find /root/intra/data -name '*v11*' -type f 2>/dev/null | head -20")
print(so)

ssh.close()
