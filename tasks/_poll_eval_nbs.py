"""Poller for Phase 3 V3 ship-blocker eval_nbs screen.

- 45-min budget, 150s cadence, up to 18 polls.
- Emits compact status per poll.
- Exits on ALL DONE / eval_nbs done / failure signal / screen gone / timeout.
"""
from __future__ import annotations
import json
import sys
import time

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
LOG_PATH = "/tmp/eval_nbs.log"
SCREEN_NAME = "eval_nbs"
REMOTE_NB_DIR = "/root/intra/evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/"
DONE_MARKERS = ("ALL DONE", "eval_nbs done")
FAILURE_PATTERNS = ("Traceback", "Killed", "OOM", "MemoryError", "assert ", "AssertionError")
ERROR_LEVEL = ("ERROR", "FAILED")  # case-sensitive, full-word-ish
POLL_SEC = 150
MAX_WALL_SEC = 45 * 60


def run(client, cmd, timeout=30):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors="replace"), err.read().decode(errors="replace")


def poll_once(client):
    cmds = {
        "screen": f"screen -ls | grep -F {SCREEN_NAME} || echo NO_SCREEN",
        "tail": f"tail -30 {LOG_PATH}",
        "wc": f"wc -l {LOG_PATH} 2>/dev/null",
        "mem": "free -h | head -2",
        "disk": "df -h /root | tail -1",
        "nb_list": f"ls -la {REMOTE_NB_DIR} 2>/dev/null",
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
    for m in DONE_MARKERS:
        if m in tail:
            return "SUCCESS"
    for p in FAILURE_PATTERNS:
        if p in tail:
            return f"FAILURE ({p.strip()})"
    # Case-sensitive ERROR/FAILED tokens at line starts (avoid false positives on log-level colors)
    for line in tail.splitlines():
        stripped = line.strip()
        for tok in ERROR_LEVEL:
            if stripped.startswith(tok) or f" {tok} " in stripped or f"[{tok}]" in stripped:
                return f"FAILURE ({tok})"
    if not screen_alive:
        return "FAILURE (screen gone)"
    return "RUNNING"


def current_nb(tail):
    """Heuristic: which of the 6 notebooks is currently executing."""
    nbs = [
        "s1_individual_net",
        "s2_combined_net",
        "s3_mc_combined_net",
        "s4_individual_ml2_net",
        "s5_combined_ml2_net",
        "s6_mc_combined_ml2_net",
    ]
    latest = None
    for line in tail.splitlines():
        for nb in nbs:
            if nb in line:
                latest = nb
    return latest or "?"


def main():
    start = time.time()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30, banner_timeout=30)

    poll_n = 0
    verdict = "UNKNOWN"
    last_snap = None

    try:
        while True:
            poll_n += 1
            elapsed = time.time() - start
            snap = poll_once(client)
            last_snap = snap
            state = classify(snap)
            tail = snap["tail"]
            nb_now = current_nb(tail)

            screen_alive = SCREEN_NAME in snap["screen"]
            wc_line = snap["wc"].strip().split()[0] if snap["wc"].strip() else "?"
            mem_line = snap["mem"].splitlines()[-1].strip() if snap["mem"].strip() else ""
            print(
                f"[poll {poll_n}] wall={elapsed:.0f}s state={state} nb={nb_now} log_lines={wc_line} screen={'yes' if screen_alive else 'no'}",
                flush=True,
            )
            last3 = "\n".join(tail.rstrip().splitlines()[-3:])
            for ln in last3.splitlines():
                print(f"  tail> {ln}", flush=True)
            print(f"  mem> {mem_line}", flush=True)

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
        "wall_sec": round(time.time() - start, 1),
        "final_tail": last_snap["tail"] if last_snap else "",
        "final_screen": last_snap["screen"] if last_snap else "",
        "final_nb_list": last_snap["nb_list"] if last_snap else "",
        "final_mem": last_snap["mem"] if last_snap else "",
        "final_disk": last_snap["disk"] if last_snap else "",
    }
    print("===RESULT_JSON===", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if verdict == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
