"""run_v4_refit_no_gcid_remote.py — combo-agnostic V4 refit on sweep-runner-1.

Phase 1 of the LLM Council 2026-04-20 Option-1 verdict (transcript:
`evaluation/council/council-transcript-20260420_134920.md`). Retrains V4
with `global_combo_id` stripped from `ALL_FEATURES` and `CATEGORICAL_COLS`
(runtime rebind via `--no-combo-id`), closing the per-combo memorization
leak flagged in the contamination audit.

No temporal filter is applied — the leak is structural (ID-as-feature),
not temporal, so training on the full v11 sweep is both sufficient and
maximally label-rich. Output: `data/ml/adaptive_rr_v4_no_gcid/`.

Per `feedback_remote_git_sync.md`, code is not SFTP'd — the runner does
a `git fetch && git reset --hard origin/master` before training so the
tree matches master.

Launches under screen session `v4_no_gcid` wrapped by
`systemd-run --scope -p MemoryMax=9G -p CPUQuota=280%`.

Usage:
    python scripts/runners/run_v4_refit_no_gcid_remote.py
"""
from __future__ import annotations

import time

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
SCREEN_NAME = "v4_no_gcid"
LOG_PATH = "/tmp/v4_no_gcid.log"

WRAPPER_SH = """#!/bin/bash
# Combo-agnostic V4 refit wrapper — runs on sweep-runner-1 under screen + systemd-run scope.
set -euo pipefail
cd /root/intra

echo "[v4_no_gcid] git pull @ $(date)"
git fetch --all
git reset --hard origin/master

echo "[v4_no_gcid] training V4 with --no-combo-id on full v11"
python3 scripts/models/adaptive_rr_model_v4.py \\
    --output-dir data/ml/adaptive_rr_v4_no_gcid \\
    --no-combo-id \\
    --max-rows 10000000 \\
    --target-base-trades 1200000 \\
    --n-folds 5 \\
    --rebuild-cache

echo "[v4_no_gcid] ALL DONE $(date)"
"""


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    print(f"[launch] connected to {HOST}")

    wrapper_remote = f"{REMOTE_DIR}/run_v4_no_gcid.sh"
    with sftp.open(wrapper_remote, "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod(wrapper_remote, 0o755)
    print(f"[launch] wrote {wrapper_remote}")

    ssh.exec_command(f"screen -S {SCREEN_NAME} -X quit 2>/dev/null")[1].channel.recv_exit_status()
    time.sleep(1)

    cmd = (
        f"screen -dmS {SCREEN_NAME} bash -c '"
        f"systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
        f"bash {wrapper_remote} 2>&1 | tee {LOG_PATH}; exec bash'"
    )
    print(f"[launch] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    print(f"[launch] screen launch exit code: {exit_code}")

    time.sleep(2)

    _, so, _ = ssh.exec_command("screen -ls")
    print("[launch] screen sessions:")
    print(so.read().decode())

    print(f"\n[launch] Monitor via:")
    print(f"  ssh root@{HOST} 'tail -f {LOG_PATH}'")
    print(f"  ssh root@{HOST} 'screen -r {SCREEN_NAME}'")
    print(f"[launch] Completion marker: '[v4_no_gcid] ALL DONE'")

    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
