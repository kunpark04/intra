"""Restart v3_no_gcid eval from s3 onward.

s1 + s2 already executed successfully on remote; only s3 OOM'd because the
s3_mc_combined_net.ipynb was missing explicit n_sims=2000 pinning on the three
plot_mc_* helpers (defaults = 10_000) and the inline monte_carlo() call.
The local s3 notebook has now been patched (NotebookEdit). This script:

1. SFTP-uploads the patched s3 notebook to remote (overwrites 4 KB stub).
2. Kills any stale eval_nbs screen.
3. Launches a fresh screen running s3 -> s6 of v3_no_gcid inside a fresh
   python subprocess PER notebook so kernel heap resets between s3 -> s4
   transitions (the prior run held all cells in one kernel and may have
   leaked the ~2 GB MC arrays into s4).
4. Wraps the loop in systemd-run --scope MemoryMax=9G CPUQuota=280% per
   CLAUDE.md / reference_remote_job_workflow policy.
"""
from __future__ import annotations
import time
import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"

LOCAL_NB = (
    "evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s3_mc_combined_net.ipynb"
)
REMOTE_NB = f"/root/intra/{LOCAL_NB}"

WRAPPER_SH = """#!/bin/bash
set -e
cd /root/intra
export PYTHONPATH=/root/intra:$PYTHONPATH

NOTEBOOKS=(
  evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s3_mc_combined_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s4_individual_ml2_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s5_combined_ml2_net.ipynb
  evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s6_mc_combined_ml2_net.ipynb
)

for nb in "${NOTEBOOKS[@]}"; do
    echo "[v3_no_gcid_eval] Executing $nb at $(date -u +%FT%TZ)"
    python3 -c "
import nbformat
from nbclient import NotebookClient
from pathlib import Path
p = Path('$nb')
nb = nbformat.read(str(p), as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name='python3',
    resources={'metadata': {'path': str(Path.cwd())}})
client.execute()
nbformat.write(nb, str(p))
print('[ok]', p)
"
    echo "[v3_no_gcid_eval] DONE $nb"
done
echo "[v3_no_gcid_eval] Done: all 4 executed."
"""


def sh(ssh, cmd, timeout=60):
    _, so, se = ssh.exec_command(cmd, timeout=timeout)
    return so.read().decode(errors="replace"), se.read().decode(errors="replace")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)

    # 0. Kill stale eval screens
    sh(ssh, "screen -S eval_nbs -X quit 2>/dev/null; screen -S v3_no_gcid -X quit 2>/dev/null; rm -f /root/intra/v3_no_gcid.log")
    time.sleep(1)

    # 1. SFTP upload patched s3 notebook
    sftp = ssh.open_sftp()
    sftp.put(LOCAL_NB, REMOTE_NB)
    remote_size = sftp.stat(REMOTE_NB).st_size
    print(f"[sftp] uploaded {LOCAL_NB} -> {REMOTE_NB} ({remote_size} bytes)")

    # 2. Verify s1/s2 still present and executed
    out, _ = sh(ssh,
        "ls -la /root/intra/evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/")
    print(f"[remote ls]\n{out}")

    # 3. Upload wrapper
    with sftp.open("/root/intra/run_v3_no_gcid.sh", "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod("/root/intra/run_v3_no_gcid.sh", 0o755)
    sftp.close()
    print("[wrote /root/intra/run_v3_no_gcid.sh]")

    # 4. Verify n_sims=2000 is present in patched s3
    out, _ = sh(ssh, f"grep -c 'n_sims=2000' {REMOTE_NB}")
    print(f"[verify patch on remote] n_sims=2000 count = {out.strip()}")

    # 5. Launch screen
    cmd = (
        "screen -dmS v3_no_gcid bash -c '"
        "systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
        "bash /root/intra/run_v3_no_gcid.sh 2>&1 | tee /root/intra/v3_no_gcid.log; exec bash'"
    )
    sh(ssh, cmd)
    time.sleep(5)

    # 6. Confirm screen alive
    out, _ = sh(ssh, "screen -ls | grep v3_no_gcid || echo NOSCREEN")
    print(f"[screen -ls]\n{out}")

    time.sleep(10)
    out, _ = sh(ssh, "head -20 /root/intra/v3_no_gcid.log 2>/dev/null || echo NOLOG")
    print(f"[head log]\n{out}")
    ssh.close()


if __name__ == "__main__":
    main()
