"""Inspect sweep-runner-1 state before the Probe 1 launch.

Collects: git HEAD + branch, git status of conflicting paths, whether
data/NQ_1min.csv exists, whether Cython is installed, which screen
sessions are actually attached to live processes, and disk free.
"""
from __future__ import annotations

import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST = "195.88.25.157"
USER = "root"
PWD = "J@maicanP0wer123"


def run(c: paramiko.SSHClient, cmd: str, label: str) -> None:
    print(f"\n── {label} ──")
    print(f"$ {cmd}")
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

    run(c, "cd /root/intra && git rev-parse HEAD", "git HEAD")
    run(c, "cd /root/intra && git branch --show-current", "branch")
    run(c, "cd /root/intra && git status --short | head -20", "git status")
    run(
        c,
        "cd /root/intra && git diff --stat "
        "evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/ | head -10",
        "blocking-file diff-stat",
    )
    run(
        c,
        "cd /root/intra && git log -1 --format='%h %s' "
        "evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/",
        "blocking-file last commit",
    )
    run(c, "ls -la /root/intra/data/NQ_1min.csv 2>&1 | head -3", "NQ_1min.csv")
    run(c, "ls -la /root/intra/data/NQ_15min.parquet /root/intra/data/NQ_1h.parquet 2>&1 | head -3", "15m/1h parquets")
    run(c, "ls -la /root/intra/data/ml/originals/ml_dataset_v11*.parquet 2>&1 | head -20", "existing v11 outputs")
    run(c, "python3 -c 'import Cython; print(Cython.__version__)' 2>&1", "Cython version")
    run(c, "python3 -c 'import numba; print(numba.__version__)' 2>&1", "Numba version")
    run(c, "python3 -c 'import numpy; print(numpy.__version__)' 2>&1", "numpy version")
    run(
        c,
        "ls -la /root/intra/scripts/data_pipeline/ 2>&1 | head -10",
        "scripts/data_pipeline contents",
    )
    run(c, "screen -ls", "live screen sessions")
    run(c, "ps -eo pid,rss,pcpu,etime,cmd --sort=-rss | head -10", "top processes")
    run(c, "df -h /root | head -2", "disk free")
    run(c, "uname -a", "uname")

    c.close()


if __name__ == "__main__":
    main()
