"""Poll Probe 1 C3b (1h sweep) and pull artifacts if complete.

Behavior:
  - Connect to sweep-runner-1.
  - Check screen session + log tail + manifest progress.
  - If completion is confirmed (final parquet + merged manifest present),
    SFTP-pull both artifacts to data/ml/originals/.
  - If still running, report progress and exit 1 so caller knows to poll again.

Completion heuristics:
  1. Screen session `probe1_1h` no longer listed.
  2. `ml_dataset_v11_1h.parquet` exists on remote with non-zero size.
  3. Manifest at `ml_dataset_v11_1h_manifest.json` has 1500 entries.
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
REMOTE_PARQUET = "/root/intra/data/ml/originals/ml_dataset_v11_1h.parquet"
REMOTE_MANIFEST = "/root/intra/data/ml/originals/ml_dataset_v11_1h_manifest.json"
LOCAL_PARQUET = Path("data/ml/originals/ml_dataset_v11_1h.parquet")
LOCAL_MANIFEST = Path("data/ml/originals/ml_dataset_v11_1h_manifest.json")


def exec_cmd(c: paramiko.SSHClient, cmd: str, timeout: int = 30) -> str:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    so = o.read().decode(errors="replace").rstrip()
    se = e.read().decode(errors="replace").rstrip()
    if se:
        print(f"[stderr] {se}")
    return so


def run(c, cmd, label):
    print(f"\n── {label} ──")
    so = exec_cmd(c, cmd)
    if so:
        print(so)
    return so


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=30)

    screen_out = run(c, f"screen -ls | grep {SCREEN_NAME} || echo '[none]'", "screen status")
    run(c, f"tail -30 {LOG_PATH}", "log tail")
    run(c, "free -h | head -2", "memory")
    run(
        c,
        "ps -eo pid,rss,pcpu,etime,cmd --sort=-rss "
        "| grep param_sweep | grep -v grep | head -5 || echo '[no sweep procs]'",
        "sweep procs",
    )
    parquet_stat = run(
        c,
        f"ls -la {REMOTE_PARQUET} 2>&1 || echo '[missing]'",
        "parquet stat",
    )
    manifest_count = run(
        c,
        f"python3 -c \"import json; m=json.load(open('{REMOTE_MANIFEST}')); "
        f"print(len(m) if isinstance(m,(list,dict)) else 0)\" 2>&1 || echo '[?]'",
        "manifest entry count",
    )

    sweep_running = "[none]" not in screen_out
    parquet_present = "[missing]" not in parquet_stat and "No such file" not in parquet_stat
    try:
        manifest_n = int(manifest_count.strip())
    except ValueError:
        manifest_n = 0

    print(f"\n[state] screen_alive={sweep_running}  "
          f"parquet_present={parquet_present}  manifest_n={manifest_n}/1500")

    if sweep_running or not parquet_present or manifest_n < 1500:
        print("\n[wait] C3b not finished — not pulling yet.")
        c.close()
        return 1

    # Pull artifacts
    print("\n── pull 1h artifacts ──")
    sftp = c.open_sftp()
    LOCAL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    for remote, local in ((REMOTE_PARQUET, LOCAL_PARQUET),
                          (REMOTE_MANIFEST, LOCAL_MANIFEST)):
        t0 = time.time()
        sftp.get(remote, str(local))
        dt = time.time() - t0
        mb = local.stat().st_size / 1e6
        print(f"[pull] {remote} -> {local}  ({mb:.1f} MB in {dt:.1f}s)")

    c.close()
    print("\n[ok] C3b artifacts local.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
