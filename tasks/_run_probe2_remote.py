"""Run Probe 2 (combo-865 isolation) end-to-end on sweep-runner-1.

Workflow:
  1. SSH to sweep-runner-1 (paramiko) + stash any remote-side edits.
  2. git fetch + checkout the Probe 2 preregistration commit (a49f370).
  3. Remove stale Cython .so (force Numba fallback, same as probe1).
  4. Verify 15m + 1h bar caches exist (built during probe1).
  5. Write /root/intra/run_probe2.sh wrapper.
  6. Launch in detached screen `probe2_combo865`.
  7. Poll every 30 s until screen session terminates or 60-min timeout.
  8. SFTP-pull the two output parquets + manifests to local.
  9. Tail the remote log and print the last 60 lines locally.

Output parquets:
  /root/intra/data/ml/probe2/combo865_15m_test.parquet
  /root/intra/data/ml/probe2/combo865_1h_test.parquet
        -> data/ml/probe2/ locally

Wall-clock budget per the preregistration §6: ~15 min total.
Abort-threshold: 60 min (trigger investigation, not a FAIL of the gate).

Preregistration: tasks/probe2_preregistration.md (signed commit a49f370).
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
PWD = "J@maicanP0wer123"  # noqa: S105 — local-only ref

SCREEN_NAME = "probe2_combo865"
LOG_PATH = f"/root/intra/logs/{SCREEN_NAME}.log"
WRAPPER_PATH = f"/root/intra/run_{SCREEN_NAME}.sh"

SIGNED_COMMIT = "a49f370"

WRAPPER_BODY = f"""#!/bin/bash
set -e
cd /root/intra

echo "=== Probe 2 start $(date -u) ==="
echo "target commit: {SIGNED_COMMIT}"
git rev-parse HEAD

mkdir -p data/ml/probe2

# 15m test — combo 865 ONLY, via --start-combo/--end-combo range
echo ""
echo "=== 15m test partition ==="
exec_start_15m=$(date +%s)
systemd-run --scope \\
    -p MemoryMax=8G \\
    -p CPUQuota=280% \\
    python3 scripts/param_sweep.py \\
        --combinations 3000 \\
        --start-combo 865 --end-combo 866 \\
        --seed 0 --range-mode v11 \\
        --timeframe 15min --eval-partition test \\
        --output data/ml/probe2/combo865_15m_test.parquet \\
        --workers 1
exec_end_15m=$(date +%s)
echo "15m wall-clock: $((exec_end_15m - exec_start_15m)) s"

# 1h test — combo 865 ONLY
echo ""
echo "=== 1h test partition ==="
exec_start_1h=$(date +%s)
systemd-run --scope \\
    -p MemoryMax=8G \\
    -p CPUQuota=280% \\
    python3 scripts/param_sweep.py \\
        --combinations 1500 \\
        --start-combo 865 --end-combo 866 \\
        --seed 0 --range-mode v11 \\
        --timeframe 1h --eval-partition test \\
        --output data/ml/probe2/combo865_1h_test.parquet \\
        --workers 1
exec_end_1h=$(date +%s)
echo "1h wall-clock: $((exec_end_1h - exec_start_1h)) s"

echo ""
echo "=== artifacts ==="
ls -la data/ml/probe2/
echo ""
echo "=== Probe 2 end $(date -u) ==="
"""


def run(c: paramiko.SSHClient, cmd: str, label: str | None = None, timeout: int = 120) -> tuple[str, str]:
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

    # 1. Stash remote changes + fetch + checkout signed commit
    run(client, "cd /root/intra && git stash push -u -m 'pre-probe2-'$(date +%s) 2>&1 | head -3 || true", "stash remote edits")
    run(client, "cd /root/intra && git fetch origin master 2>&1 | tail -5", "git fetch")
    run(client, f"cd /root/intra && git checkout {SIGNED_COMMIT} 2>&1 | head -5", "checkout signed commit")
    run(client, "cd /root/intra && git rev-parse HEAD", "remote HEAD after checkout")

    # 2. Cython .so — remove so Numba fallback engages (probe1 pattern)
    run(client, "rm -f /root/intra/src/cython_ext/*.so /root/intra/src/cython_ext/*.c && echo ok", "clear Cython .so")

    # 3. Verify bar caches exist (built in probe1)
    run(
        client,
        "ls -la /root/intra/data/NQ_15min.parquet /root/intra/data/NQ_1h.parquet 2>&1",
        "verify bar caches present",
    )

    # 4. Write wrapper via SFTP
    print("\n-- write wrapper --")
    sftp = client.open_sftp()
    with sftp.file(WRAPPER_PATH, "w") as f:
        f.write(WRAPPER_BODY)
    sftp.chmod(WRAPPER_PATH, 0o755)
    print(f"[ok] wrote {WRAPPER_PATH}")

    # 5. Launch in screen
    run(
        client,
        f"cd /root/intra && mkdir -p logs && rm -f {LOG_PATH} "
        f"&& screen -dmS {SCREEN_NAME} bash -c '{WRAPPER_PATH} > {LOG_PATH} 2>&1' && sleep 5",
        f"launch screen({SCREEN_NAME})",
    )
    run(client, "screen -ls 2>&1 || true", "screen -ls (confirm detached)")

    # 6. Poll every 30 s. Screen exits when the script completes.
    print("\n-- polling --")
    max_wait = 60 * 60  # 60 min abort threshold
    elapsed = 0
    poll_interval = 30
    while elapsed < max_wait:
        so, _ = run(
            client,
            f"screen -ls 2>/dev/null | grep -c {SCREEN_NAME} || true",
            None,  # silent
            timeout=30,
        )
        alive = so.strip() and so.strip() != "0"
        # Tail last 4 lines so we can see progress markers
        run(client, f"tail -4 {LOG_PATH} 2>/dev/null || echo '(no log yet)'", f"t+{elapsed}s — tail")
        if not alive:
            print(f"\n[ok] screen session terminated after ~{elapsed} s")
            break
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        print(f"\n[WARN] 60-min abort threshold reached. Investigate manually (screen is still running).")

    # 7. Show full log
    run(client, f"tail -80 {LOG_PATH} 2>&1", "full log tail")

    # 8. Pull artifacts
    print("\n-- pulling artifacts --")
    local_dir = Path("data/ml/probe2")
    local_dir.mkdir(parents=True, exist_ok=True)
    files_to_pull = [
        ("/root/intra/data/ml/probe2/combo865_15m_test.parquet", "data/ml/probe2/combo865_15m_test.parquet"),
        ("/root/intra/data/ml/probe2/combo865_15m_test_manifest.json", "data/ml/probe2/combo865_15m_test_manifest.json"),
        ("/root/intra/data/ml/probe2/combo865_1h_test.parquet", "data/ml/probe2/combo865_1h_test.parquet"),
        ("/root/intra/data/ml/probe2/combo865_1h_test_manifest.json", "data/ml/probe2/combo865_1h_test_manifest.json"),
    ]
    for remote_p, local_p in files_to_pull:
        try:
            st = sftp.stat(remote_p)
            sftp.get(remote_p, local_p)
            print(f"[ok] pulled {local_p}  ({st.st_size:,} bytes)")
        except FileNotFoundError:
            print(f"[miss] {remote_p}  (not found on remote)")
        except Exception as e:
            print(f"[err] {remote_p}: {e!r}")

    client.close()
    print("\n[done] probe 2 remote run complete — next: python tasks/_probe2_readout.py")


if __name__ == "__main__":
    main()
