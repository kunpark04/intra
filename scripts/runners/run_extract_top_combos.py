"""Upload + run `extract_top_combos.py` on sweep-runner-1.

The extractor is fast (<5 s) so this runner executes the script inline
over SSH, then SFTPs the resulting `evaluation/top_strategies.json`
back to the local repo. No screen session or polling required.
"""
from __future__ import annotations
from pathlib import Path
import sys

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra"

REPO = Path(__file__).resolve().parent.parent.parent
LOCAL_SCRIPT = REPO / "scripts" / "analysis" / "extract_top_combos.py"
LOCAL_OUT = REPO / "evaluation" / "top_strategies.json"
REMOTE_SCRIPT = f"{REMOTE_DIR}/scripts/analysis/extract_top_combos.py"
REMOTE_OUT = f"{REMOTE_DIR}/evaluation/top_strategies.json"


def main() -> None:
    """Upload the extractor, run it on sweep-runner-1, SFTP the JSON back.

    Parses optional CLI overrides (`--top-k`, `--min-trades`, …),
    forwards them to the remote invocation, and writes the output to
    `evaluation/top_strategies.json` locally.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    extra_args = " ".join(sys.argv[1:])

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    sftp = ssh.open_sftp()

    # Ensure remote scripts/analysis/ + evaluation/ dirs exist (post-reorg).
    ssh.exec_command(
        f"mkdir -p {REMOTE_DIR}/scripts/analysis {REMOTE_DIR}/evaluation"
    )[1].channel.recv_exit_status()

    print(f"  Uploading {LOCAL_SCRIPT.relative_to(REPO)}...", end=" ", flush=True)
    sftp.put(str(LOCAL_SCRIPT), REMOTE_SCRIPT)
    print("OK")

    cmd = f"cd {REMOTE_DIR} && python3 scripts/analysis/extract_top_combos.py {extra_args}".strip()
    print(f"  Running: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    if out:
        print(out.rstrip())
    if err:
        print(err.rstrip(), file=sys.stderr)
    if exit_code != 0:
        print(f"  Extractor exited non-zero ({exit_code}); aborting.",
              file=sys.stderr)
        sftp.close()
        ssh.close()
        sys.exit(exit_code)

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {REMOTE_OUT}...", end=" ", flush=True)
    sftp.get(REMOTE_OUT, str(LOCAL_OUT))
    print("OK")

    sftp.close()
    ssh.close()
    print(f"Done. Local artifact: {LOCAL_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
