"""Upload v10_9264_rr2.ipynb to sweep-runner-1 and execute it in-place.

Assumes the main eval upload has already placed src/, scripts/evaluation/,
and top_strategies.json on the remote (most of the code it needs is already
there). Only the new notebook + any changed helpers are re-uploaded.

Poll remote job with: screen -r v10_9264 or tail -f /tmp/v10_9264.log
Pull result with: python scripts/runners/_pull_v10_9264_rr2.py
"""
from __future__ import annotations
import time
from pathlib import Path
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
REPO = Path(__file__).resolve().parent.parent.parent

UPLOAD_FILES = [
    "evaluation/v10_9264/v10_9264_rr2.ipynb",
    "evaluation/top_strategies.json",
    "scripts/evaluation/_top_perf_common.py",
    "scripts/evaluation/composed_strategy_runner.py",
    "scripts/evaluation/final_holdout_eval_v3_c1_fixed500.py",
    "src/__init__.py", "src/config.py", "src/data_loader.py",
    "src/backtest.py", "src/strategy.py", "src/risk.py",
    "src/reporting.py", "src/scoring.py", "src/io_paths.py",
    "src/indicators/__init__.py", "src/indicators/ema.py",
    "src/indicators/atr.py", "src/indicators/zscore.py",
    "src/indicators/zscore_variants.py", "src/indicators/pipeline.py",
]

EXEC_PY = """\
import nbformat
from nbclient import NotebookClient
from pathlib import Path

p = Path('evaluation/v10_9264/v10_9264_rr2.ipynb')
nb = nbformat.read(str(p), as_version=4)
print(f'=== Executing {p} ({len(nb.cells)} cells) ===', flush=True)
client = NotebookClient(
    nb, timeout=3600, kernel_name='python3',
    resources={'metadata': {'path': str(Path.cwd())}},
)
client.execute()
nbformat.write(nb, str(p))
print(f'  Done -> {p}', flush=True)
"""

WRAPPER_SH = """\
#!/usr/bin/env bash
set -e
cd /root/intra
echo "[v10_9264] Starting $(date)"
python3 -m pip install --quiet --break-system-packages nbformat nbclient ipykernel matplotlib 2>&1 | tail -5
python3 -m ipykernel install --user --name python3 --display-name "Python 3" 2>&1 | tail -3
python3 /root/intra/run_v10_9264_rr2_exec.py
echo "[v10_9264] ALL DONE $(date)"
"""


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    ssh.exec_command(
        f"mkdir -p {REMOTE_DIR}/evaluation/v10_9264 {REMOTE_DIR}/scripts/evaluation "
        f"{REMOTE_DIR}/src {REMOTE_DIR}/src/indicators"
    )[1].channel.recv_exit_status()

    for rel in UPLOAD_FILES:
        local = REPO / rel
        remote = f"{REMOTE_DIR}/{rel}"
        print(f"  Uploading {rel} ({local.stat().st_size:,} B)...", end=" ", flush=True)
        sftp.put(str(local), remote)
        print("OK")

    with sftp.open(f"{REMOTE_DIR}/run_v10_9264_rr2_exec.py", "w") as f:
        f.write(EXEC_PY)
    with sftp.open(f"{REMOTE_DIR}/run_v10_9264_rr2.sh", "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod(f"{REMOTE_DIR}/run_v10_9264_rr2.sh", 0o755)
    print("  Wrote run_v10_9264_rr2_exec.py + run_v10_9264_rr2.sh")

    ssh.exec_command("screen -S v10_9264 -X quit 2>/dev/null")[1].channel.recv_exit_status()
    time.sleep(1)

    cmd = (
        "screen -dmS v10_9264 bash -c '"
        "systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
        "bash /root/intra/run_v10_9264_rr2.sh 2>&1 | tee /tmp/v10_9264.log; exec bash'"
    )
    print(f"  Launching: {cmd}")
    _, stdout, _ = ssh.exec_command(cmd)
    print(f"  screen launch exit code: {stdout.channel.recv_exit_status()}")

    time.sleep(1)
    _, stdout, _ = ssh.exec_command("screen -ls")
    print(stdout.read().decode())

    sftp.close(); ssh.close()
    print("Done. Poll: tail -f /tmp/v10_9264.log  |  Pull: python scripts/runners/_pull_v10_9264_rr2.py")


if __name__ == "__main__":
    main()
