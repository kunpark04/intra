"""Launch the v11 friction-aware parameter sweep on sweep-runner-1.

v11 differs from v10 in two ways:
  1. _run_backtest_light subtracts $5/contract round-trip friction from every
     trade's net PnL (COST_PER_CONTRACT_RT in _FIXED).
  2. Stop-distance sampling excludes friction-toxic regions:
     stop_fixed_pts [15, 100], atr_multiplier [1.5, 6.0], swing_lookback [5, 30).
  Qualitative diversity (z-score formulation, filters, dual-window, exit flags)
  is identical to v10.

Output: /root/intra/data/ml/originals/ml_dataset_v11.parquet (+ manifest).

Remote uses `git pull` to sync scripts/param_sweep.py; data/NQ_1min.csv and
the src/ tree are assumed present from earlier sweeps (they live on the same
VM).

~30k combos, --workers 0 (auto = 3 on 4-vCPU runner), expected ~1-2 days.

Poll with: screen -r v11_sweep  |  tail -f /tmp/v11_sweep.log
"""
from __future__ import annotations
import time
from pathlib import Path
import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"

REPO = Path(__file__).resolve().parent.parent.parent

N_COMBOS = 30000
SEED     = 0
OUTPUT   = "data/ml/originals/ml_dataset_v11.parquet"


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)

    # Remote has unrelated uncommitted modifications in src/reporting.py from
    # earlier jobs — avoid `git reset --hard`. Just SFTP-patch the single
    # changed file for this sweep (scripts/param_sweep.py).
    print("  SFTP upload scripts/param_sweep.py ...")
    sftp = ssh.open_sftp()
    local = REPO / "scripts/param_sweep.py"
    sftp.put(str(local), f"{REMOTE_DIR}/scripts/param_sweep.py")
    print(f"  Uploaded {local.stat().st_size:,} B")

    ssh.exec_command(
        f"mkdir -p {REMOTE_DIR}/data/ml/originals"
    )[1].channel.recv_exit_status()

    wrapper = f"""\
#!/usr/bin/env bash
set -e
cd {REMOTE_DIR}
echo "[v11_sweep] Starting $(date)"
python3 -m pip install --quiet --break-system-packages pyarrow pandas numpy numba 2>&1 | tail -5
echo "[v11_sweep] Launching param_sweep.py --range-mode v11 --combinations {N_COMBOS}"
python3 scripts/param_sweep.py \\
    --combinations {N_COMBOS} \\
    --range-mode v11 \\
    --seed {SEED} \\
    --workers 0 \\
    --output {OUTPUT} 2>&1 | tee /tmp/v11_sweep_run.log
echo "[v11_sweep] ALL DONE $(date)"
"""
    with sftp.open(f"{REMOTE_DIR}/run_v11_sweep.sh", "w") as f:
        f.write(wrapper)
    sftp.chmod(f"{REMOTE_DIR}/run_v11_sweep.sh", 0o755)
    sftp.close()
    print("  Wrote run_v11_sweep.sh")

    ssh.exec_command("screen -S v11_sweep -X quit 2>/dev/null")[1].channel.recv_exit_status()
    time.sleep(1)

    cmd = (
        f"screen -dmS v11_sweep bash -c '"
        f"systemd-run --scope -p MemoryMax=8G -p CPUQuota=280% "
        f"bash {REMOTE_DIR}/run_v11_sweep.sh 2>&1 | tee /tmp/v11_sweep.log; exec bash'"
    )
    print(f"  Launching: {cmd}")
    _, stdout, _ = ssh.exec_command(cmd)
    print(f"  screen launch exit code: {stdout.channel.recv_exit_status()}")

    time.sleep(2)
    _, stdout, _ = ssh.exec_command("screen -ls")
    print(stdout.read().decode())

    ssh.close()
    print("Done. Poll: tail -f /tmp/v11_sweep.log  |  "
          "Artifacts: /root/intra/data/ml/originals/ml_dataset_v11.parquet")


if __name__ == "__main__":
    main()
