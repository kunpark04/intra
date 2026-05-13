"""Blocking-wait poller for v3_no_gcid detached screen session on sweep-runner-1.

Reuses a single paramiko SSH connection, polls every 120s, and returns
one of SUCCESS | FAILURE | TIMEOUT. Writes result JSON to stdout and
detailed logs to the file path given by --out.

Launch time (UTC epoch) of the remote job: 2026-04-21 01:24:57Z
Hard timeout: 35 minutes from launch.
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
LOG_PATH = "/tmp/v3_no_gcid.log"
SCREEN_NAME = "v3_no_gcid"
DONE_MARKER = "[v3_no_gcid] ALL DONE"
FAILURE_PATTERNS = ("Traceback", "Killed", "OOM-killed", "OOMKilled")

# 2026-04-21 01:24:57 UTC -> unix epoch
# datetime(2026,4,21,1,24,57, tzinfo=timezone.utc).timestamp()
LAUNCH_EPOCH = int(datetime(2026, 4, 21, 1, 24, 57, tzinfo=timezone.utc).timestamp())
TIMEOUT_SEC = 35 * 60
POLL_SEC = 120


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 30) -> tuple[str, str, int]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30, banner_timeout=30)

    poll_n = 0
    final_tail = ""
    final_screen_ls = ""
    verdict = "UNKNOWN"
    completion_utc = None

    try:
        while True:
            poll_n += 1
            # Run the three commands together; cheap vs 3 round-trips.
            out, _, _ = run(
                client,
                f"tail -n 30 {LOG_PATH} 2>&1; echo '===SCREEN==='; screen -ls 2>&1 | grep -F {SCREEN_NAME} || echo 'no-screen'; echo '===EPOCH==='; date -u +%s",
            )
            parts = out.split("===SCREEN===")
            tail = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            screen_block, _, epoch_block = rest.partition("===EPOCH===")
            try:
                remote_epoch = int(epoch_block.strip().splitlines()[-1])
            except Exception:
                remote_epoch = int(time.time())

            elapsed = remote_epoch - LAUNCH_EPOCH
            final_tail = tail
            final_screen_ls = screen_block.strip()

            # Print a compact progress line to stdout for observability.
            print(
                f"[poll {poll_n}] elapsed={elapsed}s screen={'alive' if SCREEN_NAME in screen_block else 'dead'} tail_len={len(tail)}",
                file=sys.stderr,
                flush=True,
            )

            # Success check first.
            if DONE_MARKER in tail:
                verdict = "SUCCESS"
                completion_utc = datetime.now(timezone.utc).isoformat()
                break

            # Failure check.
            hit = next((p for p in FAILURE_PATTERNS if p in tail), None)
            screen_dead = SCREEN_NAME not in screen_block
            if hit:
                verdict = f"FAILURE (pattern: {hit})"
                break
            if screen_dead and elapsed > 60:
                # Screen gone without ALL DONE marker in the last 30 lines.
                verdict = "FAILURE (screen session gone, no ALL DONE)"
                break

            if elapsed > TIMEOUT_SEC:
                verdict = "TIMEOUT"
                break

            time.sleep(POLL_SEC)

    finally:
        try:
            client.close()
        except Exception:
            pass

    result = {
        "verdict": verdict,
        "polls": poll_n,
        "completion_utc": completion_utc,
        "elapsed_at_break_sec": elapsed if "elapsed" in locals() else None,
        "final_tail": final_tail,
        "final_screen_ls": final_screen_ls,
    }
    print(json.dumps(result, indent=2))
    return 0 if verdict == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
