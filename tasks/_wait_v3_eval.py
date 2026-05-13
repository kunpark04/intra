"""Block until v3_eval completes, fails, or times out (45 min)."""
from __future__ import annotations
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _poll_v3_eval import one_check

MAX_WAIT = 45 * 60
POLL_INTERVAL = 60

start = time.time()
while time.time() - start < MAX_WAIT:
    wc, tail, scr = one_check()
    elapsed = int(time.time() - start)
    print(f"[{elapsed}s] wc={wc.split()[0]} screen={scr}", flush=True)
    last_lines = tail.strip().split("\n")[-3:]
    for ln in last_lines:
        print(f"    {ln}", flush=True)
    done = "Done: all 12 executed." in tail
    failed = any(k in tail for k in ("Traceback", "Killed", "OOM", "MemoryError", "FAILED"))
    if done:
        print("FINAL=DONE")
        print(tail)
        sys.exit(0)
    if failed:
        print("FINAL=FAILED")
        print(tail)
        sys.exit(2)
    if scr == "0":
        print("FINAL=SCREEN_GONE")
        print(tail)
        sys.exit(3)
    time.sleep(POLL_INTERVAL)

print("FINAL=TIMEOUT")
sys.exit(4)
