"""Upload eval notebook + xlsx builder to sweep-runner-1 and execute them.

Uploads the perf .ipynb (built locally via _build_v2_notebooks.py), the xlsx
builder, and the notebook-side helpers (composed_strategy_runner, eval script,
top strategies JSON, full src/ tree) into /root/intra/.

Remote runner (`run_eval_nbs.sh`) executes the perf notebook in-place via
nbclient, then runs build_trade_log_xlsx.py to produce
`evaluation/top_trade_log.xlsx`. Poll with `_poll_eval_nbs.py` and pull with
`_pull_eval_nbs.py`.
"""
from __future__ import annotations
import time
from pathlib import Path
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
REPO = Path(__file__).resolve().parent.parent.parent

UPLOAD_FILES = [
    "evaluation/v10_topk/s1_individual.ipynb",
    "evaluation/v10_topk/s2_combined.ipynb",
    "evaluation/v10_topk/s3_mc_combined.ipynb",
    "evaluation/v10_topk/s4_individual_ml2.ipynb",
    "evaluation/v10_topk/s5_combined_ml2.ipynb",
    "evaluation/v10_topk/s6_mc_combined_ml2.ipynb",
    "evaluation/v10_topk_net/s1_individual_net.ipynb",
    "evaluation/v10_topk_net/s2_combined_net.ipynb",
    "evaluation/v10_topk_net/s3_mc_combined_net.ipynb",
    "evaluation/v10_topk_net/s4_individual_ml2_net.ipynb",
    "evaluation/v10_topk_net/s5_combined_ml2_net.ipynb",
    "evaluation/v10_topk_net/s6_mc_combined_ml2_net.ipynb",
    "evaluation/v11_topk/s1_individual.ipynb",
    "evaluation/v11_topk/s2_combined.ipynb",
    "evaluation/v11_topk/s3_mc_combined.ipynb",
    "evaluation/v11_topk/s4_individual_ml2.ipynb",
    "evaluation/v11_topk/s5_combined_ml2.ipynb",
    "evaluation/v11_topk/s6_mc_combined_ml2.ipynb",
    "evaluation/v11_topk_net/s1_individual_net.ipynb",
    "evaluation/v11_topk_net/s2_combined_net.ipynb",
    "evaluation/v11_topk_net/s3_mc_combined_net.ipynb",
    "evaluation/v11_topk_net/s4_individual_ml2_net.ipynb",
    "evaluation/v11_topk_net/s5_combined_ml2_net.ipynb",
    "evaluation/v11_topk_net/s6_mc_combined_ml2_net.ipynb",
    "evaluation/top_strategies.json",
    "evaluation/top_strategies_v11.json",
    "scripts/evaluation/_top_perf_common.py",
    "scripts/evaluation/composed_strategy_runner.py",
    "scripts/evaluation/final_holdout_eval_v3_c1_fixed500.py",
    "scripts/evaluation/build_trade_log_xlsx.py",
    "src/__init__.py",
    "src/config.py",
    "src/data_loader.py",
    "src/backtest.py",
    "src/strategy.py",
    "src/risk.py",
    "src/reporting.py",
    "src/scoring.py",
    "src/io_paths.py",
    "src/indicators/__init__.py",
    "src/indicators/ema.py",
    "src/indicators/atr.py",
    "src/indicators/zscore.py",
    "src/indicators/zscore_variants.py",
    "src/indicators/pipeline.py",
]

# The notebook runner as a standalone Python file uploaded to remote.
EXEC_PY = """\
import nbformat
from nbclient import NotebookClient
from pathlib import Path

NOTEBOOKS = [
    'evaluation/v10_topk/s1_individual.ipynb',
    'evaluation/v10_topk/s2_combined.ipynb',
    'evaluation/v10_topk/s3_mc_combined.ipynb',
    'evaluation/v10_topk/s4_individual_ml2.ipynb',
    'evaluation/v10_topk/s5_combined_ml2.ipynb',
    'evaluation/v10_topk/s6_mc_combined_ml2.ipynb',
    'evaluation/v10_topk_net/s1_individual_net.ipynb',
    'evaluation/v10_topk_net/s2_combined_net.ipynb',
    'evaluation/v10_topk_net/s3_mc_combined_net.ipynb',
    'evaluation/v10_topk_net/s4_individual_ml2_net.ipynb',
    'evaluation/v10_topk_net/s5_combined_ml2_net.ipynb',
    'evaluation/v10_topk_net/s6_mc_combined_ml2_net.ipynb',
    'evaluation/v11_topk/s1_individual.ipynb',
    'evaluation/v11_topk/s2_combined.ipynb',
    'evaluation/v11_topk/s3_mc_combined.ipynb',
    'evaluation/v11_topk/s4_individual_ml2.ipynb',
    'evaluation/v11_topk/s5_combined_ml2.ipynb',
    'evaluation/v11_topk/s6_mc_combined_ml2.ipynb',
    'evaluation/v11_topk_net/s1_individual_net.ipynb',
    'evaluation/v11_topk_net/s2_combined_net.ipynb',
    'evaluation/v11_topk_net/s3_mc_combined_net.ipynb',
    'evaluation/v11_topk_net/s4_individual_ml2_net.ipynb',
    'evaluation/v11_topk_net/s5_combined_ml2_net.ipynb',
    'evaluation/v11_topk_net/s6_mc_combined_ml2_net.ipynb',
]

for rel in NOTEBOOKS:
    p = Path(rel)
    nb = nbformat.read(str(p), as_version=4)
    print(f'=== Executing {rel} ({len(nb.cells)} cells) ===', flush=True)
    client = NotebookClient(
        nb, timeout=3600, kernel_name='python3',
        resources={'metadata': {'path': str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(nb, str(p))
    print(f'  Done -> {p}', flush=True)
"""

# Bash wrapper: install deps, register kernel, run notebook, then build xlsx.
WRAPPER_SH = """\
#!/usr/bin/env bash
set -e
cd /root/intra
echo "[eval_nbs] Starting $(date)"

python3 -m pip install --quiet --break-system-packages nbformat nbclient ipykernel matplotlib openpyxl 2>&1 | tail -5
python3 -m ipykernel install --user --name python3 --display-name "Python 3" 2>&1 | tail -3

echo "[eval_nbs] --- Executing perf notebook ---"
python3 /root/intra/run_eval_nbs_exec.py

echo "[eval_nbs] --- Building trade log xlsx ---"
python3 /root/intra/scripts/evaluation/build_trade_log_xlsx.py

echo "[eval_nbs] ALL DONE $(date)"
"""


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    ssh.exec_command(
        f"mkdir -p {REMOTE_DIR}/evaluation "
        f"{REMOTE_DIR}/evaluation/v10_topk {REMOTE_DIR}/evaluation/v10_topk_net "
        f"{REMOTE_DIR}/evaluation/v11_topk {REMOTE_DIR}/evaluation/v11_topk_net "
        f"{REMOTE_DIR}/scripts/evaluation "
        f"{REMOTE_DIR}/src {REMOTE_DIR}/src/indicators"
    )[1].channel.recv_exit_status()

    for rel in UPLOAD_FILES:
        local = REPO / rel
        remote = f"{REMOTE_DIR}/{rel}"
        print(f"  Uploading {rel} ({local.stat().st_size:,} B)...", end=" ", flush=True)
        sftp.put(str(local), remote)
        print("OK")

    with sftp.open(f"{REMOTE_DIR}/run_eval_nbs_exec.py", "w") as f:
        f.write(EXEC_PY)
    with sftp.open(f"{REMOTE_DIR}/run_eval_nbs.sh", "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod(f"{REMOTE_DIR}/run_eval_nbs.sh", 0o755)
    print("  Wrote run_eval_nbs_exec.py + run_eval_nbs.sh")

    # Kill stale screen if any
    ssh.exec_command("screen -S eval_nbs -X quit 2>/dev/null")[1].channel.recv_exit_status()
    time.sleep(1)

    cmd = (
        "screen -dmS eval_nbs bash -c '"
        "systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
        "bash /root/intra/run_eval_nbs.sh 2>&1 | tee /tmp/eval_nbs.log; exec bash'"
    )
    print(f"  Launching: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    print(f"  screen launch exit code: {exit_code}")

    time.sleep(1)
    stdin, stdout, stderr = ssh.exec_command("screen -ls")
    print(stdout.read().decode())

    sftp.close(); ssh.close()
    print("Done. Poll with: screen -r eval_nbs  |  tail -f /tmp/eval_nbs.log")


if __name__ == "__main__":
    main()
