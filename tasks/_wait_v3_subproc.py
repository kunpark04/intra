"""Subprocess-based poll loop — each poll is a fresh paramiko connection in a new process."""
from __future__ import annotations
import subprocess, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
POLL_SCRIPT = os.path.join(HERE, "_poll_v3_eval.py")
MAX_WAIT = 55 * 60
POLL_INTERVAL = 90

start = time.time()
while time.time() - start < MAX_WAIT:
    try:
        r = subprocess.run(
            [sys.executable, POLL_SCRIPT],
            capture_output=True, text=True, timeout=45,
        )
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = "[TIMEOUT in subprocess]\n"
    elapsed = int(time.time() - start)
    last = out.strip().split("\n")[-1] if out.strip() else "(empty)"
    for ln in out.strip().split("\n"):
        if "Executing" in ln or "DONE" in ln or "STATUS" in ln:
            print(f"[{elapsed}s] {ln}", flush=True)
    if "STATUS=DONE" in out:
        print("FINAL=DONE", flush=True)
        sys.exit(0)
    if "STATUS=FAILED" in out:
        print("FINAL=FAILED", flush=True)
        sys.exit(2)
    if "STATUS=SCREEN_GONE" in out:
        print("FINAL=SCREEN_GONE", flush=True)
        sys.exit(3)
    time.sleep(POLL_INTERVAL)

print("FINAL=TIMEOUT", flush=True)
sys.exit(4)
