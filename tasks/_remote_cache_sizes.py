import paramiko
HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=20)
_, so, _ = ssh.exec_command("ls -laSh /root/intra/evaluation/_cache/ /root/intra/evaluation/_ml2_cache*.pkl 2>/dev/null; du -sh /root/intra/evaluation/_cache/ 2>/dev/null")
print(so.read().decode())
ssh.close()
