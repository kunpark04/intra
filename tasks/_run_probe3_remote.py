"""Run Probe 3 (combo-865 session/exit + robustness) end-to-end on sweep-runner-1.

Workflow:
  1. SSH to sweep-runner-1 (paramiko) + stash any remote-side edits.
  2. git fetch + checkout origin/master (picks up Phase C1 script bundle).
  3. Remove stale Cython .so (force Numba fallback, same as probe1/probe2).
  4. Verify 15m + 1h bar caches + Probe 2 1h parquet exist.
  5. Write /root/intra/run_probe3.sh wrapper.
  6. Launch in detached screen `probe3_combo865`.
  7. Poll every 60 s until screen session terminates or 90-min timeout.
  8. SFTP-pull all gate JSONs + parquets + readout.json to local.
  9. Tail the remote log and print the last 80 lines locally.

Output artifacts pulled locally to data/ml/probe3/:
  regime_halves.json
  param_nbhd.json, param_nbhd/{combos.json, grid_sidecar.json, combos.parquet}
  15m_nc.json, nc_15m/{EX_*_combos.json, EX_*.parquet}
  1h_ritual.json, ritual_1h/{EX_*_combos.json, EX_*.parquet}
  readout.json

Wall-clock budget per preregistration §7: ~35 min compute.
Abort-threshold: 90 min (trigger investigation, not a FAIL of the gate).

Preregistration: tasks/probe3_preregistration.md (signed commit 8636167).
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import paramiko

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST = "195.88.25.157"
USER = "root"
PWD  = "J@maicanP0wer123"  # noqa: S105 — local-only ref

SCREEN_NAME  = "probe3_combo865"
LOG_PATH     = f"/root/intra/logs/{SCREEN_NAME}.log"
WRAPPER_PATH = f"/root/intra/run_{SCREEN_NAME}.sh"

WRAPPER_BODY = f"""#!/bin/bash
set -e
cd /root/intra

echo "=== Probe 3 start $(date -u) ==="
git rev-parse HEAD

mkdir -p data/ml/probe3 logs

# --- §4.1 regime halves (local re-use of Probe 2 parquet; no engine run) ---
echo ""
echo "=== §4.1 regime halves ==="
t0=$(date +%s)
python3 tasks/_probe3_regime_halves.py
echo "§4.1 wall-clock: $(($(date +%s) - t0)) s"

# --- §4.2 parameter neighborhood (27 combos × 1h test) ---
echo ""
echo "=== §4.2 parameter neighborhood ==="
t0=$(date +%s)
systemd-run --scope \\
    -p MemoryMax=8G \\
    -p CPUQuota=280% \\
    python3 tasks/_probe3_param_nbhd.py
echo "§4.2 wall-clock: $(($(date +%s) - t0)) s"

# --- §4.3 15m negative control (4 EX runs × 15m test) ---
echo ""
echo "=== §4.3 15m negative control ==="
t0=$(date +%s)
systemd-run --scope \\
    -p MemoryMax=8G \\
    -p CPUQuota=280% \\
    python3 tasks/_probe3_15m_nc.py
echo "§4.3 wall-clock: $(($(date +%s) - t0)) s"

# --- §4.4 1h session/exit ritual (4 EX runs × 1h test) ---
echo ""
echo "=== §4.4 1h session/exit ritual ==="
t0=$(date +%s)
systemd-run --scope \\
    -p MemoryMax=8G \\
    -p CPUQuota=280% \\
    python3 tasks/_probe3_1h_ritual.py
echo "§4.4 wall-clock: $(($(date +%s) - t0)) s"

# --- Aggregate readout ---
echo ""
echo "=== aggregate readout ==="
python3 tasks/_probe3_readout.py

echo ""
echo "=== artifacts ==="
ls -la data/ml/probe3/
echo ""
echo "=== Probe 3 end $(date -u) ==="
"""


def run(c: paramiko.SSHClient, cmd: str, label: str | None = None,
        timeout: int = 120) -> tuple[str, str]:
    if label:
        print(f"\n-- {label} --")
        print(f"$ {cmd}")
    _, out, err = c.exec_command(cmd, timeout=timeout)
    so = out.read().decode(errors="replace")
    se = err.read().decode(errors="replace")
    if so:
        print(so.rstrip())
    if se.strip():
        print(f"[stderr] {se.rstrip()}")
    return so, se


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    client.connect(HOST, username=USER, password=PWD, timeout=30)

    # 1. Stash remote changes + fetch + checkout latest master
    run(client, "cd /root/intra && git stash push -u -m 'pre-probe3-'$(date +%s) 2>&1 | head -3 || true",
        "stash remote edits")
    run(client, "cd /root/intra && git fetch origin master 2>&1 | tail -5", "git fetch")
    run(client, "cd /root/intra && git checkout master 2>&1 | head -3 && git reset --hard origin/master 2>&1 | head -3",
        "align master with origin/master")
    run(client, "cd /root/intra && git rev-parse HEAD", "remote HEAD after checkout")

    # 2. Cython .so — remove so Numba fallback engages (probe1/probe2 pattern)
    run(client,
        "rm -f /root/intra/src/cython_ext/*.so /root/intra/src/cython_ext/*.c && echo ok",
        "clear Cython .so")

    # 3. Verify caches + Probe 2 parquet present
    run(client,
        "ls -la /root/intra/data/NQ_15min.parquet /root/intra/data/NQ_1h.parquet "
        "/root/intra/data/ml/probe2/combo865_1h_test.parquet 2>&1",
        "verify bar caches + probe2 parquet present")

    # 4. Write wrapper via SFTP
    print("\n-- write wrapper --")
    sftp = client.open_sftp()
    with sftp.file(WRAPPER_PATH, "w") as f:
        f.write(WRAPPER_BODY)
    sftp.chmod(WRAPPER_PATH, 0o755)
    print(f"[ok] wrote {WRAPPER_PATH}")

    # 5. Launch in screen
    run(client,
        f"cd /root/intra && mkdir -p logs && rm -f {LOG_PATH} "
        f"&& screen -dmS {SCREEN_NAME} bash -c '{WRAPPER_PATH} > {LOG_PATH} 2>&1' && sleep 5",
        f"launch screen({SCREEN_NAME})")
    run(client, "screen -ls 2>&1 || true", "screen -ls (confirm detached)")

    # 6. Poll every 60 s up to 90 min (Probe 3 budget ~35 min + buffer).
    print("\n-- polling --")
    max_wait = 90 * 60
    elapsed = 0
    poll_interval = 60
    while elapsed < max_wait:
        so, _ = run(client,
                    f"screen -ls 2>/dev/null | grep -c {SCREEN_NAME} || true",
                    None, timeout=30)
        alive = so.strip() and so.strip() != "0"
        run(client, f"tail -4 {LOG_PATH} 2>/dev/null || echo '(no log yet)'",
            f"t+{elapsed}s — tail")
        if not alive:
            print(f"\n[ok] screen session terminated after ~{elapsed} s")
            break
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        print(f"\n[WARN] 90-min abort threshold reached. "
              f"Investigate manually (screen is still running).")

    # 7. Show full log
    run(client, f"tail -120 {LOG_PATH} 2>&1", "full log tail")

    # 8. Pull artifacts
    print("\n-- pulling artifacts --")
    local_base = Path("data/ml/probe3")
    local_base.mkdir(parents=True, exist_ok=True)
    (local_base / "param_nbhd").mkdir(exist_ok=True)
    (local_base / "nc_15m").mkdir(exist_ok=True)
    (local_base / "ritual_1h").mkdir(exist_ok=True)

    files_to_pull = [
        "data/ml/probe3/regime_halves.json",
        "data/ml/probe3/param_nbhd.json",
        "data/ml/probe3/15m_nc.json",
        "data/ml/probe3/1h_ritual.json",
        "data/ml/probe3/readout.json",
        # param neighborhood sidecars
        "data/ml/probe3/param_nbhd/combos.json",
        "data/ml/probe3/param_nbhd/grid_sidecar.json",
        "data/ml/probe3/param_nbhd/combos.parquet",
        # 15m NC EX parquets + combo JSONs
        "data/ml/probe3/nc_15m/EX_0_native_combos.json",
        "data/ml/probe3/nc_15m/EX_0_native.parquet",
        "data/ml/probe3/nc_15m/EX_1_maxhold_60h_combos.json",
        "data/ml/probe3/nc_15m/EX_1_maxhold_60h.parquet",
        "data/ml/probe3/nc_15m/EX_2_TOD_1500ET_combos.json",
        "data/ml/probe3/nc_15m/EX_2_TOD_1500ET.parquet",
        "data/ml/probe3/nc_15m/EX_3_breakeven_after_1R_combos.json",
        "data/ml/probe3/nc_15m/EX_3_breakeven_after_1R.parquet",
        # 1h ritual EX parquets + combo JSONs
        "data/ml/probe3/ritual_1h/EX_0_native_combos.json",
        "data/ml/probe3/ritual_1h/EX_0_native.parquet",
        "data/ml/probe3/ritual_1h/EX_1_maxhold_60h_combos.json",
        "data/ml/probe3/ritual_1h/EX_1_maxhold_60h.parquet",
        "data/ml/probe3/ritual_1h/EX_2_TOD_1500ET_combos.json",
        "data/ml/probe3/ritual_1h/EX_2_TOD_1500ET.parquet",
        "data/ml/probe3/ritual_1h/EX_3_breakeven_after_1R_combos.json",
        "data/ml/probe3/ritual_1h/EX_3_breakeven_after_1R.parquet",
    ]
    for rel in files_to_pull:
        remote_p = f"/root/intra/{rel}"
        local_p  = rel
        try:
            st = sftp.stat(remote_p)
            Path(local_p).parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_p, local_p)
            print(f"[ok] pulled {local_p}  ({st.st_size:,} bytes)")
        except FileNotFoundError:
            print(f"[miss] {remote_p}  (not found on remote)")
        except Exception as e:
            print(f"[err] {remote_p}: {e!r}")

    client.close()
    print("\n[done] probe 3 remote run complete — next: review "
          "data/ml/probe3/readout.json and draft tasks/probe3_verdict.md")


if __name__ == "__main__":
    main()
