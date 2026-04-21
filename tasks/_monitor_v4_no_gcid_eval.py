"""Stream filtered log lines from the remote v4_no_gcid ship-blocker audit.

Emits one stdout line per relevant event (notebook start / done / error /
ALL DONE / OOM / Traceback / Killed). Exits when ALL DONE appears or the
screen session dies without a success marker.
"""
from __future__ import annotations

import re
import sys
import time

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
LOG = "/tmp/eval_nbs.log"
SCREEN = "eval_nbs"

# Alternation MUST cover every terminal state — silence is not success.
PATTERN = re.compile(
    r"=== Executing|Done ->|ALL DONE|Traceback|Error|FAILED|Killed|OOM|"
    r"\[eval_nbs\]|BrokenPipe|CUDA out|KeyError|FileNotFoundError|ImportError|"
    r"permission denied",
    re.IGNORECASE,
)

POLL = 30  # seconds between remote log reads


def _connect() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    return ssh


def main() -> int:
    last_size = 0
    seen_all_done = False
    no_change_iters = 0
    start = time.time()

    while True:
        try:
            ssh = _connect()
            # First check screen is still alive
            _, so, _ = ssh.exec_command(f"screen -ls | grep -c {SCREEN} || true")
            alive = so.read().decode().strip()

            # Read log tail only — start from last_size to avoid re-emitting
            _, so, _ = ssh.exec_command(f"wc -c < {LOG} 2>/dev/null || echo 0")
            cur_size = int(so.read().decode().strip() or 0)

            if cur_size > last_size:
                _, so, _ = ssh.exec_command(
                    f"tail -c +{last_size + 1} {LOG}"
                )
                new_bytes = so.read().decode(errors="replace")
                for line in new_bytes.splitlines():
                    if PATTERN.search(line):
                        print(line, flush=True)
                        if "ALL DONE" in line:
                            seen_all_done = True
                last_size = cur_size
                no_change_iters = 0
            else:
                no_change_iters += 1

            ssh.close()

            if seen_all_done:
                print(f"[monitor] SUCCESS after {int(time.time() - start)}s")
                return 0

            # If screen is gone AND no ALL DONE yet, job died silently.
            if alive == "0" and not seen_all_done and no_change_iters >= 2:
                print(f"[monitor] screen {SCREEN} GONE without ALL DONE — job crashed?")
                return 2

        except Exception as e:
            print(f"[monitor] transient SSH error: {e}", flush=True)

        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
