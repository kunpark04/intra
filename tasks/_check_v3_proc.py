"""Check remote process state."""
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"


def sh(ssh, cmd):
    _, stdout, _ = ssh.exec_command(cmd, timeout=30)
    return stdout.read().decode(errors="replace")


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=20)

print("[ps -ef for python/bash]")
print(sh(ssh, "ps -ef | grep -E 'python|v3_eval' | grep -v grep"))

print("[full log]")
print(sh(ssh, "tail -40 /root/intra/v3_eval.log"))

print("[memory]")
print(sh(ssh, "free -g"))

print("[cpu top 5]")
print(sh(ssh, "ps aux --sort=-%cpu | head -6"))

ssh.close()
