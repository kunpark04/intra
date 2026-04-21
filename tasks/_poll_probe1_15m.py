"""Poll Probe 1 C3a (probe1_15m) sweep progress.

Reads per memory/reference_remote_job_workflow.md §6:
  - screen -ls        (session alive?)
  - log tail          (progress bar, errors)
  - memory            (watch for OOM)
  - process list      (%CPU, RSS, ELAPSED)
  - chunks dir size   (throughput signal)
  - manifest          (combo-level completion count)
"""
from __future__ import annotations

import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST = "195.88.25.157"
USER = "root"
PWD = "J@maicanP0wer123"

LOG_PATH = "/root/intra/logs/probe1_15m.log"
MANIFEST = "/root/intra/data/ml/originals/ml_dataset_v11_15m_manifest.json"
CHUNKS_GLOB = "/root/intra/data/ml/originals/ml_dataset_v11_15m_chunks"


def run(c, cmd, label):
    print(f"\n── {label} ──")
    _, o, e = c.exec_command(cmd, timeout=30)
    so = o.read().decode(errors="replace").rstrip()
    se = e.read().decode(errors="replace").rstrip()
    if so:
        print(so)
    if se:
        print(f"[stderr] {se}")


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=30)

    run(c, "screen -ls | grep probe1_15m || echo '[warn] probe1_15m session missing'", "screen status")
    run(c, f"tail -40 {LOG_PATH}", "log tail")
    run(c, "free -h | head -2", "memory")
    run(
        c,
        "ps -eo pid,rss,pcpu,etime,cmd --sort=-rss "
        "| grep param_sweep | grep -v grep | head -5",
        "sweep processes",
    )
    run(c, f"ls -la {CHUNKS_GLOB}/ 2>&1 | wc -l", "chunk file count")
    run(c, f"ls -la {CHUNKS_GLOB}/ 2>&1 | tail -5", "chunk files tail")
    run(
        c,
        f"python3 -c \"import json; "
        f"m=json.load(open('{MANIFEST}')); "
        f"from collections import Counter; "
        f"print('entries:', len(m)); "
        f"print('status:', dict(Counter(v.get('status','?') for v in m.values())))\" "
        f"2>&1 || echo '(no manifest yet)'",
        "manifest summary",
    )
    run(c, "df -h /root | head -2", "disk free")

    c.close()


if __name__ == "__main__":
    main()
