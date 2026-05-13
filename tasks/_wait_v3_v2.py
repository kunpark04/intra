"""Persistent-connection wait for v3_eval completion."""
from __future__ import annotations
import time, sys, os
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
MAX_WAIT = 50 * 60
POLL_INTERVAL = 60


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20,
                banner_timeout=30, auth_timeout=30)
    # Use keepalive
    transport = ssh.get_transport()
    transport.set_keepalive(30)

    start = time.time()
    while time.time() - start < MAX_WAIT:
        try:
            _, so, _ = ssh.exec_command(
                "wc -l /root/intra/v3_eval.log 2>/dev/null | awk '{print $1}'; "
                "grep -cE 'Done: all 12|DONE ' /root/intra/v3_eval.log 2>/dev/null; "
                "grep -cE 'Traceback|Killed|OOM|MemoryError|FAILED' /root/intra/v3_eval.log 2>/dev/null; "
                "screen -ls | grep -c v3_eval; "
                "grep -E 'DONE|Executing' /root/intra/v3_eval.log | tail -1",
                timeout=30,
            )
            out = so.read().decode(errors="replace").strip().split("\n")
            if len(out) >= 5:
                wc, done_count, fail_count, scr, last = out[0], out[1], out[2], out[3], out[4]
            else:
                wc, done_count, fail_count, scr, last = "?", "0", "0", "?", ""
            elapsed = int(time.time() - start)
            print(f"[{elapsed}s] wc={wc} done={done_count} fail={fail_count} screen={scr}",
                  flush=True)
            print(f"    last: {last}", flush=True)
            if "Done: all 12" in " ".join(out):
                print("FINAL=DONE", flush=True)
                return 0
            if int(fail_count) > 0:
                print("FINAL=FAILED", flush=True)
                return 2
            if scr == "0" and int(done_count) < 12:
                print("FINAL=SCREEN_GONE", flush=True)
                return 3
        except Exception as e:
            print(f"[{int(time.time()-start)}s] EXC: {type(e).__name__}: {e}", flush=True)
            try:
                ssh.close()
            except Exception:
                pass
            time.sleep(5)
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, username=USER, password=PASS, timeout=20)
            ssh.get_transport().set_keepalive(30)
        time.sleep(POLL_INTERVAL)

    print("FINAL=TIMEOUT", flush=True)
    return 4


if __name__ == "__main__":
    sys.exit(main())
