"""SFTP pull of /root/intra/data/ml/adaptive_rr_v3_no_gcid/ and /tmp/v3_no_gcid.log.

Mirrors remote dir contents to local data/ml/adaptive_rr_v3_no_gcid/. Pulls
full log to tasks/_v3_no_gcid_remote.log. Prints file list + sizes.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import paramiko

HOST = "195.88.25.157"
USER = "root"
PASS = "J@maicanP0wer123"
REMOTE_DIR = "/root/intra/data/ml/adaptive_rr_v3_no_gcid"
REMOTE_LOG = "/tmp/v3_no_gcid.log"

REPO = Path(__file__).resolve().parent.parent
LOCAL_DIR = REPO / "data" / "ml" / "adaptive_rr_v3_no_gcid"
LOCAL_LOG = REPO / "tasks" / "_v3_no_gcid_remote.log"


def main() -> int:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    t = paramiko.Transport((HOST, 22))
    t.connect(username=USER, password=PASS)
    sftp = paramiko.SFTPClient.from_transport(t)

    print(f"=== SFTP pull {REMOTE_DIR} -> {LOCAL_DIR} ===")
    try:
        entries = sftp.listdir_attr(REMOTE_DIR)
    except FileNotFoundError:
        print(f"[fail] remote dir not found: {REMOTE_DIR}")
        sftp.close(); t.close()
        return 2

    pulled = []
    for e in entries:
        remote = f"{REMOTE_DIR}/{e.filename}"
        local = LOCAL_DIR / e.filename
        print(f"  pulling {e.filename} ({e.st_size:,} bytes)")
        sftp.get(remote, str(local))
        pulled.append((e.filename, e.st_size))

    print(f"\n=== SFTP pull {REMOTE_LOG} -> {LOCAL_LOG} ===")
    sftp.get(REMOTE_LOG, str(LOCAL_LOG))
    log_size = LOCAL_LOG.stat().st_size
    print(f"  log: {log_size:,} bytes")

    sftp.close()
    t.close()

    print("\n=== SUMMARY ===")
    for fn, sz in pulled:
        print(f"  {fn}  {sz:,}")
    print(f"  + {LOCAL_LOG}  {log_size:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
