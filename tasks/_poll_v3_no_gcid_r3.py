"""Replacement-3 poller for v3_no_gcid screen. 90-min budget, 180s cadence.

Prints snapshot per poll; exits on ALL DONE / failure / timeout.
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
REMOTE_PID = "206420"
ART_DIR = "/root/intra/data/ml/adaptive_rr_v3_no_gcid/"
DONE_MARKER = "[v3_no_gcid] ALL DONE"
FAILURE_PATTERNS = ("Traceback", "OOM-killed", "MemoryError", "Killed")
POLL_SEC = 180
# Budget starts when this script starts.
MAX_WALL_SEC = 90 * 60


def run(client, cmd, timeout=30):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors="replace"), err.read().decode(errors="replace")


def poll_once(client, poll_n):
    cmds = {
        "screen": f"screen -ls | grep -F {SCREEN_NAME} || echo NO_SCREEN",
        "tail": f"tail -40 {LOG_PATH}",
        "mem": "free -m | head -2",
        "ps": f"ps -p {REMOTE_PID} -o pid,stat,pcpu,pmem,etime,comm 2>/dev/null || echo PID_DEAD",
        "art": f"ls -la {ART_DIR} 2>/dev/null | tail -15",
        "epoch": "date -u +%s",
    }
    snap = {}
    for k, v in cmds.items():
        out, _ = run(client, v)
        snap[k] = out
    return snap


def classify(snap):
    tail = snap["tail"]
    screen_alive = SCREEN_NAME in snap["screen"]
    if DONE_MARKER in tail:
        return "SUCCESS"
    for p in FAILURE_PATTERNS:
        if p in tail:
            return f"FAILURE ({p})"
    if not screen_alive:
        return "FAILURE (screen gone)"
    return "RUNNING"


def main():
    start = time.time()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30, banner_timeout=30)

    poll_n = 0
    verdict = "UNKNOWN"
    last_snap = None
    milestones = []
    prev_v10_seen = False
    prev_fold_lines = set()

    try:
        while True:
            poll_n += 1
            elapsed = time.time() - start
            snap = poll_once(client, poll_n)
            last_snap = snap
            state = classify(snap)
            tail = snap["tail"]

            # Milestone detection
            if "Streaming v10" in tail and "trades," in tail.split("Streaming v10")[-1][:200]:
                if not prev_v10_seen:
                    prev_v10_seen = True
                    milestones.append((poll_n, elapsed, "v10 done"))
            for line in tail.splitlines():
                if "fold" in line.lower() and "training" in line.lower():
                    if line not in prev_fold_lines:
                        prev_fold_lines.add(line)
                        milestones.append((poll_n, elapsed, line.strip()))

            # Compact summary line
            pid_alive = "PID_DEAD" not in snap["ps"]
            print(
                f"[poll {poll_n}] wall={elapsed:.0f}s state={state} pid_alive={pid_alive} screen_alive={SCREEN_NAME in snap['screen']}",
                flush=True,
            )
            # Print last two lines of tail for visibility
            last2 = "\n".join(tail.rstrip().splitlines()[-3:])
            print("  tail>", last2.replace("\n", "\n  tail> "), flush=True)
            print("  mem>", snap["mem"].splitlines()[-1].strip() if snap["mem"].strip() else "", flush=True)
            print("  ps>", snap["ps"].strip().splitlines()[-1] if snap["ps"].strip() else "", flush=True)
            art_lines = snap["art"].strip().splitlines()
            if len(art_lines) > 3:
                print(f"  art> {len(art_lines)-3} files present", flush=True)

            if state == "SUCCESS":
                verdict = "SUCCESS"
                break
            if state.startswith("FAILURE"):
                verdict = state
                break
            if elapsed > MAX_WALL_SEC:
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
        "wall_sec": time.time() - start,
        "milestones": milestones,
        "final_tail": last_snap["tail"] if last_snap else "",
        "final_screen": last_snap["screen"] if last_snap else "",
        "final_art": last_snap["art"] if last_snap else "",
        "final_mem": last_snap["mem"] if last_snap else "",
        "final_ps": last_snap["ps"] if last_snap else "",
    }
    print("===RESULT_JSON===")
    print(json.dumps(result, indent=2))
    return 0 if verdict == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
