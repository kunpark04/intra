"""Launch Phase 4b — three robustness tests in one remote run."""
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
    "scripts/calibration/recal_robustness_v3.py",
]

WRAPPER = """#!/bin/bash
cd /root/intra
set -u
echo "=== $(date -Is) starting phase4b ==="
systemd-run --scope -p MemoryMax=7G -p CPUQuota=280% \\
    python3 scripts/calibration/recal_robustness_v3.py \\
    > logs/v3_recal_robustness_run.log 2>&1
rc=$?
echo "=== $(date -Is) phase4b exit=$rc ==="
exit $rc
"""


def main() -> None:
    """Upload script + inputs to sweep-runner-1 via paramiko and launch Phase 4b recalibration robustness tests.

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
        sftp.put(str(REPO / rel), f"/root/intra/{rel}")
        print(f"uploaded {rel}")

    with sftp.file("/root/intra/run_v3_phase4b.sh", "w") as f:
        f.write(WRAPPER)
    sftp.chmod("/root/intra/run_v3_phase4b.sh", 0o755)
    print("wrapper installed")

    launch = (
        "cd /root/intra && mkdir -p logs && "
        "rm -f logs/v3_phase4b.log logs/v3_recal_robustness_run.log && "
        "screen -dmS v3_phase4b bash -c "
        '"./run_v3_phase4b.sh > logs/v3_phase4b.log 2>&1" && sleep 5'
    )
    _, o, _ = c.exec_command(launch)
    print(o.read().decode(errors="replace"))

    time.sleep(5)
    _, o, _ = c.exec_command(
        "screen -ls; echo ---PHASE4B---; tail -30 logs/v3_phase4b.log; "
        "echo ---RUN---; tail -30 logs/v3_recal_robustness_run.log 2>/dev/null || echo '(not yet)'"
    )
    print(o.read().decode(errors="replace"))
    c.close()


if __name__ == "__main__":
    main()
