"""Quick SSH check that the combo-agnostic V4 refit artifacts exist on
sweep-runner-1 before launching the ship-blocker audit.

Prints the directory listing + file sizes for `data/ml/adaptive_rr_v4_no_gcid/`
and a head of any metrics JSON. Exits non-zero if the booster is missing.
"""
from __future__ import annotations

import sys

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE = "/root/intra/data/ml/adaptive_rr_v4_no_gcid"


def main() -> int:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)

    _, so, _ = ssh.exec_command(f"ls -la {REMOTE} 2>&1")
    listing = so.read().decode()
    print(f"=== ls -la {REMOTE} ===")
    print(listing)

    _, so, _ = ssh.exec_command(
        f"test -f {REMOTE}/booster_v4.txt && echo HAVE_BOOSTER || echo NO_BOOSTER"
    )
    booster_status = so.read().decode().strip()
    print(f"booster: {booster_status}")

    _, so, _ = ssh.exec_command(
        f"test -f {REMOTE}/isotonic_calibrators_v4.json && echo HAVE_CAL || echo NO_CAL"
    )
    cal_status = so.read().decode().strip()
    print(f"calibrators: {cal_status}")

    _, so, _ = ssh.exec_command(f"cat {REMOTE}/metrics_v4.json 2>/dev/null | head -40")
    metrics = so.read().decode()
    if metrics:
        print(f"=== metrics_v4.json (head) ===")
        print(metrics)

    ssh.close()

    return 0 if booster_status == "HAVE_BOOSTER" and cal_status == "HAVE_CAL" else 2


if __name__ == "__main__":
    sys.exit(main())
