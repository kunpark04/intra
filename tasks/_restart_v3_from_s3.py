"""Restart v3_eval from s3 onward with reduced n_sims to avoid OOM."""
from __future__ import annotations
import time
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"

# New exec_py: resume from s3, plus patch n_sims to 3000 to stay under 9G cap
EXEC_PY = """\
import sys
import nbformat
from nbclient import NotebookClient
from pathlib import Path
from datetime import datetime, timezone

DIRS = [
    ('evaluation/v12_topk_top50_raw_sharpe_v3', ['s3', 's4', 's5', 's6']),
    ('evaluation/v12_topk_top50_raw_sharpe_net_v3', ['s1', 's2', 's3', 's4', 's5', 's6']),
]

notebooks = []
for d, order in DIRS:
    for prefix in order:
        matches = sorted(Path(d).glob(f'{prefix}_*.ipynb'))
        notebooks.extend(matches)

for p in notebooks:
    ts = datetime.now(timezone.utc).strftime('%FT%TZ')
    print(f'[v3_eval] Executing {p} at {ts}', flush=True)
    nb = nbformat.read(str(p), as_version=4)
    client = NotebookClient(
        nb, timeout=3600, kernel_name='python3',
        resources={'metadata': {'path': str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(nb, str(p))
    print(f'[v3_eval] DONE {p}', flush=True)
print('[v3_eval] Done: all 10 executed.', flush=True)
"""

# Patch: lower default n_sims in the three MC helper funcs in _top_perf_common.py
PATCH_SED = r"""
sed -i 's/def plot_mc_pnl(df, policy, title_prefix, years_span, n_sims=10_000):/def plot_mc_pnl(df, policy, title_prefix, years_span, n_sims=3_000):/' /root/intra/scripts/evaluation/_top_perf_common.py
sed -i 's/def plot_mc_sharpe(df, policy, title_prefix, years_span, n_sims=10_000):/def plot_mc_sharpe(df, policy, title_prefix, years_span, n_sims=3_000):/' /root/intra/scripts/evaluation/_top_perf_common.py
sed -i 's/def plot_mc_dd(df, policy, title_prefix, years_span, n_sims=10_000):/def plot_mc_dd(df, policy, title_prefix, years_span, n_sims=3_000):/' /root/intra/scripts/evaluation/_top_perf_common.py
"""

WRAPPER_SH = """#!/bin/bash
set -e
cd /root/intra
export PYTHONPATH=/root/intra:$PYTHONPATH
python3 /root/intra/run_v3_eval_exec.py
"""


def sh(ssh, cmd, timeout=60):
    _, so, se = ssh.exec_command(cmd, timeout=timeout)
    return so.read().decode(errors="replace"), se.read().decode(errors="replace")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)

    # Cleanup stale screen
    sh(ssh, "screen -S v3_eval -X quit 2>/dev/null; rm -f /root/intra/v3_eval.log")

    # Patch _top_perf_common.py MC defaults
    out, err = sh(ssh, PATCH_SED)
    print(f"[patch] stdout={out!r} stderr={err!r}")
    out, _ = sh(ssh, "grep 'def plot_mc_' /root/intra/scripts/evaluation/_top_perf_common.py")
    print(f"[patch verify]\n{out}")

    # Upload fresh wrapper + exec
    sftp = ssh.open_sftp()
    with sftp.open("/root/intra/run_v3_eval.sh", "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod("/root/intra/run_v3_eval.sh", 0o755)
    with sftp.open("/root/intra/run_v3_eval_exec.py", "w") as f:
        f.write(EXEC_PY)
    sftp.close()
    print("[wrote resume scripts]")

    time.sleep(1)
    cmd = (
        "screen -dmS v3_eval bash -c '"
        "systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
        "bash /root/intra/run_v3_eval.sh 2>&1 | tee /root/intra/v3_eval.log; exec bash'"
    )
    sh(ssh, cmd)
    time.sleep(4)
    out, _ = sh(ssh, "screen -ls | grep v3_eval || echo NOSCREEN")
    print(f"[screen]\n{out}")
    time.sleep(8)
    out, _ = sh(ssh, "head -20 /root/intra/v3_eval.log")
    print(f"[head log]\n{out}")
    ssh.close()


if __name__ == "__main__":
    main()
