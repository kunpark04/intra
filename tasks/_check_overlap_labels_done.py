"""Follow-up check: overlap_labels screen exited before our 15s verify.

Streaming rate was ~18M rows/s → full run ~6s. Confirm output parquet,
tail the log for ALL DONE marker, and re-check v3_no_gcid screen.
"""
from __future__ import annotations

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"


def run(ssh: paramiko.SSHClient, cmd: str, *, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    ec = stdout.channel.recv_exit_status()
    so = stdout.read().decode()
    se = stderr.read().decode()
    if so:
        print(so.rstrip())
    if se:
        print(f"[stderr] {se.rstrip()}")
    print(f"[exit_code] {ec}")


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)

    run(ssh, "screen -ls", label="screen -ls")
    run(ssh, "tail -40 /tmp/overlap_labels.log", label="log tail")
    run(ssh, "ls -la /root/intra/data/ml/ranker_null/", label="output dir")
    run(ssh, "screen -ls | grep v3_no_gcid || echo '[MISSING v3_no_gcid]'", label="v3_no_gcid alive")

    ssh.close()


if __name__ == "__main__":
    main()
