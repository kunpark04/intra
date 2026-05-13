"""Phase 2.3 post-launch smoke check for v3_no_gcid."""
import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=15)

commands = [
    ("screen -ls | grep v3_no_gcid", "screen session check"),
    ("tail -n 30 /tmp/v3_no_gcid.log 2>/dev/null || echo 'log not yet created'", "tail log"),
    ("ls -la /root/intra/data/ml/adaptive_rr_v3_no_gcid/ 2>/dev/null || echo 'output dir not yet created'", "output dir"),
    ("git -C /root/intra log --oneline -1", "remote HEAD"),
]

for cmd, label in commands:
    print(f"=== {label} ===")
    print(f"$ {cmd}")
    _, so, se = ssh.exec_command(cmd)
    out = so.read().decode()
    err = se.read().decode()
    print(out)
    if err.strip():
        print(f"[stderr] {err}")
    print()

ssh.close()
