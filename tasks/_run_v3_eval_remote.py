"""One-off driver for Pool B V3-filter eval notebook execution on sweep-runner-1."""
from __future__ import annotations
import sys, time
import paramiko

HOST = "195.88.25.157"; USER = "root"; PASS = "J@maicanP0wer123"

EXEC_PY = """\
import sys
import nbformat
from nbclient import NotebookClient
from pathlib import Path
from datetime import datetime, timezone

DIRS = [
    'evaluation/v12_topk_top50_raw_sharpe_v3',
    'evaluation/v12_topk_top50_raw_sharpe_net_v3',
]
ORDER = ['s1', 's2', 's3', 's4', 's5', 's6']

notebooks = []
for d in DIRS:
    for prefix in ORDER:
        matches = sorted(Path(d).glob(f'{prefix}_*.ipynb'))
        notebooks.extend(matches)

for p in notebooks:
    ts = datetime.now(timezone.utc).strftime('%FT%TZ')
    print(f'[v3_eval] Executing {p} at {ts}', flush=True)
    nb = nbformat.read(str(p), as_version=4)
    client = NotebookClient(
        nb, timeout=3600, kernel_name='python3',
        resources={'metadata': {'path': str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(nb, str(p))
    print(f'[v3_eval] DONE {p}', flush=True)
print('[v3_eval] Done: all 12 executed.', flush=True)
"""

WRAPPER_SH = """#!/bin/bash
set -e
cd /root/intra
export PYTHONPATH=/root/intra:$PYTHONPATH
python3 -m pip install --quiet --break-system-packages nbformat nbclient ipykernel 2>&1 | tail -3
python3 -m ipykernel install --user --name python3 --display-name "Python 3" 2>&1 | tail -1
python3 /root/intra/run_v3_eval_exec.py
"""


def sh(ssh, cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    return out, err


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)

    # Step 2a: git pull
    out, err = sh(ssh, "cd /root/intra && git pull --ff-only origin master 2>&1 | tail -10")
    print("[git pull]")
    print(out)

    # Step 2b: confirm files
    out, _ = sh(ssh,
        "ls /root/intra/evaluation/v12_topk_top50_raw_sharpe_v3/ "
        "/root/intra/evaluation/v12_topk_top50_raw_sharpe_net_v3/")
    print("[ls eval dirs]")
    print(out)

    # Step 2c: upload wrapper + exec_py
    sftp = ssh.open_sftp()
    with sftp.open("/root/intra/run_v3_eval.sh", "w") as f:
        f.write(WRAPPER_SH)
    sftp.chmod("/root/intra/run_v3_eval.sh", 0o755)
    with sftp.open("/root/intra/run_v3_eval_exec.py", "w") as f:
        f.write(EXEC_PY)
    sftp.close()
    print("[wrote /root/intra/run_v3_eval.sh + run_v3_eval_exec.py]")

    # Kill stale screen
    sh(ssh, "screen -S v3_eval -X quit 2>/dev/null || true")
    time.sleep(1)

    # Step 2d: launch
    cmd = (
        "screen -dmS v3_eval bash -c '"
        "systemd-run --scope -p MemoryMax=9G -p CPUQuota=280% "
        "bash /root/intra/run_v3_eval.sh 2>&1 | tee /root/intra/v3_eval.log; exec bash'"
    )
    sh(ssh, cmd)
    time.sleep(5)

    # Step 2e/f: confirm
    out, _ = sh(ssh, "screen -ls | grep v3_eval || echo NOSCREEN")
    print("[screen -ls]")
    print(out)

    time.sleep(10)
    out, _ = sh(ssh, "head -20 /root/intra/v3_eval.log 2>/dev/null || echo NOLOG")
    print("[head log]")
    print(out)

    ssh.close()


if __name__ == "__main__":
    main()
