"""Restart v3_eval from s3 onward, fresh kernel per notebook in isolated subprocess
to avoid cross-notebook memory accumulation. Also patches monte_carlo default
n_sims to 3_000 to survive 9G memory cap on 24k-trade combined portfolios."""
from __future__ import annotations
import time
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"

# Sed patches to reduce n_sims defaults across both modules
PATCH_SED = r"""
# Patch _top_perf_common.py helper default n_sims 10_000 -> 3_000
sed -i 's/def plot_mc_pnl(df, policy, title_prefix, years_span, n_sims=10_000):/def plot_mc_pnl(df, policy, title_prefix, years_span, n_sims=3_000):/' /root/intra/scripts/evaluation/_top_perf_common.py
sed -i 's/def plot_mc_sharpe(df, policy, title_prefix, years_span, n_sims=10_000):/def plot_mc_sharpe(df, policy, title_prefix, years_span, n_sims=3_000):/' /root/intra/scripts/evaluation/_top_perf_common.py
sed -i 's/def plot_mc_dd(df, policy, title_prefix, years_span, n_sims=10_000):/def plot_mc_dd(df, policy, title_prefix, years_span, n_sims=3_000):/' /root/intra/scripts/evaluation/_top_perf_common.py
# Patch src/reporting.py monte_carlo default n_sims 10_000 -> 3_000
sed -i 's/    n_sims: int = 10_000,/    n_sims: int = 3_000,/g' /root/intra/src/reporting.py
"""

# Execute one notebook per subprocess, so each kernel has a clean heap
WRAPPER_SH = """#!/bin/bash
set -e
cd /root/intra
export PYTHONPATH=/root/intra:$PYTHONPATH

NOTEBOOKS=(
  evaluation/v12_topk_top50_raw_sharpe_v3/s3_mc_combined.ipynb
  evaluation/v12_topk_top50_raw_sharpe_v3/s4_individual_ml2.ipynb
  evaluation/v12_topk_top50_raw_sharpe_v3/s5_combined_ml2.ipynb
  evaluation/v12_topk_top50_raw_sharpe_v3/s6_mc_combined_ml2.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3/s1_individual_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3/s2_combined_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3/s3_mc_combined_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3/s4_individual_ml2_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3/s5_combined_ml2_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3/s6_mc_combined_ml2_net.ipynb
)

for nb in "${NOTEBOOKS[@]}"; do
    echo "[v3_eval] Executing $nb at $(date -u +%FT%TZ)"
    python3 -c "
import nbformat
from nbclient import NotebookClient
from pathlib import Path
p = Path('$nb')
nb = nbformat.read(str(p), as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name='python3',
    resources={'metadata': {'path': str(Path.cwd())}})
client.execute()
nbformat.write(nb, str(p))
print('[ok]', p)
"
    echo "[v3_eval] DONE $nb"
done
echo "[v3_eval] Done: all 10 executed."
"""


def sh(ssh, cmd, timeout=60):
    _, so, se = ssh.exec_command(cmd, timeout=timeout)
    return so.read().decode(errors="replace"), se.read().decode(errors="replace")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)

    sh(ssh, "screen -S v3_eval -X quit 2>/dev/null; rm -f /root/intra/v3_eval.log")

    out, err = sh(ssh, PATCH_SED)
    print(f"[patch] stdout={out!r} stderr={err!r}")
    out, _ = sh(ssh, "grep 'def plot_mc_' /root/intra/scripts/evaluation/_top_perf_common.py; "
                      "grep 'n_sims: int' /root/intra/src/reporting.py")
    print(f"[patch verify]\n{out}")

    sftp = ssh.open_sftp()
    with sftp.open("/root/intra/run_v3_eval.sh", "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod("/root/intra/run_v3_eval.sh", 0o755)
    sftp.close()
    print("[wrote wrapper]")

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
