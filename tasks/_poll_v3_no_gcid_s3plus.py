"""One-shot poll for remote v3_no_gcid s3+ restart job.

Prints one status line with last few milestone lines, then exits:
  STATUS=RUNNING|DONE|FAILED|SCREEN_GONE
"""
from __future__ import annotations
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"


def sh(ssh, cmd, timeout=30):
    _, stdout, _ = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)
    log = sh(ssh,
        "grep -nE 'Executing|DONE|Traceback|Error|Killed|OOM|MemoryError|"
        "scope exit|FAILED|assert|Done: all' /root/intra/v3_no_gcid.log "
        "2>/dev/null | tail -10")
    wc = sh(ssh, "wc -l /root/intra/v3_no_gcid.log 2>/dev/null || echo '0 NONE'").strip()
    scr = sh(ssh, "screen -ls | grep -c v3_no_gcid").strip()
    free = sh(ssh, "free -m | grep Mem").strip()
    ssh.close()

    done = "Done: all 4 executed." in log
    failed = any(k in log for k in ("Traceback", "Killed", "MemoryError", "FAILED", "OOM"))
    if done:
        status = "DONE"
    elif failed:
        status = "FAILED"
    elif scr == "0":
        status = "SCREEN_GONE"
    else:
        status = "RUNNING"

    print(f"[wc]       {wc}")
    print(f"[screen]   count={scr}")
    print(f"[mem]      {free}")
    print(f"[events]\n{log}")
    print(f"STATUS={status}")


if __name__ == "__main__":
    main()
