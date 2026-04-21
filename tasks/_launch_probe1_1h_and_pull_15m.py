"""Launch Probe 1 C3b (1h v11 sweep @ 1500 combos) and concurrently pull
the C3a 15m artifacts from sweep-runner-1.

Preregistration: tasks/probe1_preregistration.md §3 (1h budget = 1500 combos).
Launch envelope mirrors C3a: systemd-run MemoryMax=8G CPUQuota=280%, workers=3.

Concurrency: SFTP is a separate TCP channel; pulling 65 MB of 15m parquet
while the 1h sweep runs on the server is safe (disk and network aren't
saturated, sweep is CPU-bound).
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST = "195.88.25.157"
USER = "root"
PWD = "J@maicanP0wer123"

SCREEN_NAME = "probe1_1h"
LOG_PATH = f"/root/intra/logs/{SCREEN_NAME}.log"
WRAPPER_PATH = f"/root/intra/run_{SCREEN_NAME}.sh"

WRAPPER_BODY = """#!/bin/bash
set -e
cd /root/intra
mkdir -p data/ml/originals
exec systemd-run --scope \\
    -p MemoryMax=8G \\
    -p CPUQuota=280% \\
    python3 scripts/param_sweep.py \\
        --combinations 1500 \\
        --range-mode v11 \\
        --timeframe 1h \\
        --seed 0 \\
        --output data/ml/originals/ml_dataset_v11_1h.parquet \\
        --workers 3 \\
        --eval-partition train
"""

PULL_FILES = [
    ("/root/intra/data/ml/originals/ml_dataset_v11_15m.parquet",
     "data/ml/originals/ml_dataset_v11_15m.parquet"),
    ("/root/intra/data/ml/originals/ml_dataset_v11_15m_manifest.json",
     "data/ml/originals/ml_dataset_v11_15m_manifest.json"),
]


def run(c, cmd, label):
    print(f"\n── {label} ──")
    _, o, e = c.exec_command(cmd, timeout=60)
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

    # 1. Write + launch C3b 1h sweep
    sftp = c.open_sftp()
    with sftp.file(WRAPPER_PATH, "w") as f:
        f.write(WRAPPER_BODY)
    sftp.chmod(WRAPPER_PATH, 0o755)
    print(f"[ok] wrote {WRAPPER_PATH}")

    run(c, f"cat {WRAPPER_PATH}", "1h wrapper contents")
    run(
        c,
        f"cd /root/intra && mkdir -p logs && rm -f {LOG_PATH} "
        f"&& screen -dmS {SCREEN_NAME} bash -c "
        f"'{WRAPPER_PATH} > {LOG_PATH} 2>&1' && sleep 6",
        f"launch screen({SCREEN_NAME})",
    )
    run(c, f"screen -ls | grep {SCREEN_NAME} || echo '[warn] missing'", "1h screen check")
    run(c, f"tail -20 {LOG_PATH}", "1h log tail (early)")

    # 2. Concurrently pull 15m artifacts (sweep is CPU-bound; SFTP uses
    #    a separate channel, safe to do while sweep runs).
    print("\n── pull 15m artifacts ──")
    for remote_path, local_path in PULL_FILES:
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        sftp.get(remote_path, str(local))
        t1 = time.time()
        size_mb = local.stat().st_size / 1e6
        print(f"[pull] {remote_path} -> {local}  ({size_mb:.1f} MB in {t1-t0:.1f}s)")

    c.close()
    print("\n[ok] C3b launched + C3a artifacts local.")


if __name__ == "__main__":
    main()
