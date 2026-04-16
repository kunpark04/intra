"""Launch Phase 3: three V3 filter backtests serially on sweep-runner-1.

Uploads inference_v3.py + the three filter_backtest_*_v3.py scripts,
then writes a serial wrapper that runs per_combo -> percentile ->
surrogate under systemd-run, each under MemoryMax=5G CPUQuota=400%.
"""
from __future__ import annotations
import io, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import paramiko

REPO = Path(__file__).resolve().parents[2]
HOST = "195.88.25.157"
USER = "root"
PWD = "J@maicanP0wer123"

UPLOADS = [
    "scripts/models/inference_v3.py",
    "scripts/backtests/filter_backtest_per_combo_v3.py",
    "scripts/backtests/filter_backtest_percentile_v3.py",
    "scripts/backtests/filter_backtest_surrogate_v3.py",
]

WRAPPER = """#!/bin/bash
cd /root/intra
set -u
for job in per_combo percentile surrogate; do
  echo "=== $(date -Is) starting $job ==="
  systemd-run --scope -p MemoryMax=5G -p CPUQuota=400% \\
      python3 scripts/backtests/filter_backtest_${job}_v3.py \\
      > logs/v3_${job}.log 2>&1
  rc=$?
  echo "=== $(date -Is) $job exit=$rc ==="
  if [ $rc -ne 0 ]; then
    echo "FAILED: $job (exit $rc); stopping chain" >&2
    exit $rc
  fi
done
echo "=== $(date -Is) all three complete ==="
"""


def main() -> None:
    """Upload script + inputs to sweep-runner-1 via paramiko and launch Phase 3 three-filter V3 comparison.

    Opens an SSH connection, SFTPs the latest local copy of the target
    script(s) and any required artifacts, starts a detached screen session
    with `systemd-run` resource caps, then exits. Polling is handled by the
    calling `/loop` skill.
    """
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=20)
    sftp = c.open_sftp()

    for rel in UPLOADS:
        local = REPO / rel
        remote = f"/root/intra/{rel}"
        sftp.put(str(local), remote)
        print(f"uploaded {rel}")

    with sftp.file("/root/intra/run_v3_phase3.sh", "w") as f:
        f.write(WRAPPER)
    sftp.chmod("/root/intra/run_v3_phase3.sh", 0o755)
    print("wrapper installed")

    launch = (
        "cd /root/intra && mkdir -p logs && "
        "rm -f logs/v3_phase3.log logs/v3_per_combo.log "
        "logs/v3_percentile.log logs/v3_surrogate.log && "
        "screen -dmS v3_phase3 bash -c "
        '"./run_v3_phase3.sh > logs/v3_phase3.log 2>&1" && sleep 5'
    )
    _, o, e = c.exec_command(launch)
    print(o.read().decode(errors="replace"))
    err = e.read().decode(errors="replace")
    if err:
        print("stderr:", err)

    time.sleep(3)
    _, o, _ = c.exec_command(
        "screen -ls; echo ---LOG---; tail -30 logs/v3_phase3.log; "
        "echo ---PER_COMBO---; tail -20 logs/v3_per_combo.log 2>/dev/null || echo '(not yet)'"
    )
    print(o.read().decode(errors="replace"))
    c.close()


if __name__ == "__main__":
    main()
