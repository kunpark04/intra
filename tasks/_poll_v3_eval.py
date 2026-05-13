"""Poll remote v3_eval screen job."""
from __future__ import annotations
import sys, time
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"


def sh(ssh, cmd, timeout=30):
    _, stdout, _ = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace")


def one_check():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)
    wc = sh(ssh, "wc -l /root/intra/v3_eval.log 2>/dev/null || echo '0 NONE'").strip()
    tail = sh(ssh,
        "grep -nE 'Executing|Done:|DONE|Traceback|Error|Killed|OOM|MemoryError|scope exit|FAILED|assert' "
        "/root/intra/v3_eval.log 2>/dev/null | tail -25")
    scr = sh(ssh, "screen -ls | grep -c v3_eval").strip()
    ssh.close()
    return wc, tail, scr


if __name__ == "__main__":
    wc, tail, scr = one_check()
    print(f"[wc] {wc}")
    print(f"[screen_count] {scr}")
    print(f"[tail]\n{tail}")
    done = "Done: all 12 executed." in tail
    failed = any(k in tail for k in ("Traceback", "Killed", "OOM", "MemoryError", "FAILED"))
    if done:
        print("STATUS=DONE")
    elif failed:
        print("STATUS=FAILED")
    elif scr == "0":
        print("STATUS=SCREEN_GONE")
    else:
        print("STATUS=RUNNING")
