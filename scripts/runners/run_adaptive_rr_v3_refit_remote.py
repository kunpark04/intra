"""run_adaptive_rr_v3_refit_remote.py — combo-agnostic V3 refit on sweep-runner-1.

Phase 2.1 of `tasks/plan_v3_audit_and_ranker_null.md`. Retrains V3 with
`global_combo_id` stripped from `ALL_FEATURES` and `CATEGORICAL_COLS`
(runtime rebind via `--no-combo-id`), closing the per-combo memorization
leak confirmed in the V4 audit (`project_v4_combo_id_leak_confirmed.md`)
and declared urgent-to-test for V3 in the LLM Council 2026-04-20 verdict.

Mirrors the V4 refit launcher (`run_v4_refit_no_gcid_remote.py`), but:

- V3's data loader is multi-version MFE (V3 defaults to versions 2..10
  via `load_mfe_parquets`). V3 has **no** `--input-parquet` flag —
  versions are selected with `--versions`. We rely on V3's defaults
  (2..10) to match the shipped V3 training set; the surgical refit only
  removes `global_combo_id`, nothing else changes.
- V3 has no `--rebuild-cache` flag (that is V4-specific — V3 reloads
  MFE parquets each run).
- Output directory: `data/ml/adaptive_rr_v3_no_gcid/`.
- Screen session + log paths swap `v4` → `v3`.

Per `feedback_remote_git_sync.md`, code is not SFTP'd — the runner does
a `git fetch && git reset --hard origin/master` before training so the
tree matches master.

Launches under screen session `v3_no_gcid` wrapped by
`systemd-run --scope -p MemoryMax=9G -p CPUQuota=280%` (Kamatera CPU
cap per `feedback_kamatera_cpu_cap.md`; LightGBM thread cap = 3 is
enforced inside `adaptive_rr_model_v3.py` per `feedback_lgbm_threads.md`).

Usage:
    python scripts/runners/run_adaptive_rr_v3_refit_remote.py
"""
from __future__ import annotations

import time

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"
SCREEN_NAME = "v3_no_gcid"
LOG_PATH = "/tmp/v3_no_gcid.log"

WRAPPER_SH = """#!/bin/bash
# Combo-agnostic V3 refit wrapper — runs on sweep-runner-1 under screen + systemd-run scope.
set -euo pipefail
cd /root/intra

echo "[v3_no_gcid] git pull @ $(date)"
git fetch --all
git reset --hard origin/master

echo "[v3_no_gcid] training V3 with --no-combo-id on MFE v2..v10 (V3 default versions)"
python3 scripts/models/adaptive_rr_model_v3.py \\
    --output-dir data/ml/adaptive_rr_v3_no_gcid \\
    --no-combo-id \\
    --max-rows 10000000 \\
    --target-base-trades 1200000 \\
    --n-folds 5

echo "[v3_no_gcid] ALL DONE $(date)"
"""


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    print(f"[launch] connected to {HOST}")

    wrapper_remote = f"{REMOTE_DIR}/run_v3_no_gcid.sh"
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
    print(f"[launch] Completion marker: '[v3_no_gcid] ALL DONE'")

    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
