"""Launch Phase 4c — new default + expanding window comparison."""
from __future__ import annotations
import io, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import paramiko

REPO = Path(__file__).resolve().parents[2]
HOST = "195.88.25.157"
USER = "root"
PWD = "J@maicanP0wer123"

UPLOADS = ["scripts/calibration/recal_window_compare_v3.py"]

WRAPPER = """#!/bin/bash
cd /root/intra
set -u
echo "=== $(date -Is) starting phase4c ==="
systemd-run --scope -p MemoryMax=7G -p CPUQuota=400% \\
    python3 scripts/calibration/recal_window_compare_v3.py \\
    > logs/v3_phase4c_run.log 2>&1
rc=$?
echo "=== $(date -Is) phase4c exit=$rc ==="
exit $rc
"""


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=20)
    sftp = c.open_sftp()
    for rel in UPLOADS:
        sftp.put(str(REPO / rel), f"/root/intra/{rel}")
        print(f"uploaded {rel}")
    with sftp.file("/root/intra/run_v3_phase4c.sh", "w") as f:
        f.write(WRAPPER)
    sftp.chmod("/root/intra/run_v3_phase4c.sh", 0o755)

    launch = (
        "cd /root/intra && mkdir -p logs && "
        "rm -f logs/v3_phase4c.log logs/v3_phase4c_run.log && "
        "screen -dmS v3_phase4c bash -c "
        '"./run_v3_phase4c.sh > logs/v3_phase4c.log 2>&1" && sleep 5'
    )
    _, o, _ = c.exec_command(launch)
    print(o.read().decode(errors="replace"))

    time.sleep(5)
    _, o, _ = c.exec_command(
        "screen -ls; echo ---RUN---; tail -30 logs/v3_phase4c_run.log 2>/dev/null || echo '(not yet)'"
    )
    print(o.read().decode(errors="replace"))
    c.close()


if __name__ == "__main__":
    main()
