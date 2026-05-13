"""SFTP pull overlap labels parquet + remote log from sweep-runner-1."""
from __future__ import annotations
import os
import sys
import paramiko

HOST = "195.88.25.157"
USER = "root"
PASSWORD = "J@maicanP0wer123"

REMOTE_PARQUET = "/root/intra/data/ml/ranker_null/combo_overlap_labels.parquet"
REMOTE_LOG = "/tmp/overlap_labels.log"

LOCAL_PARQUET = r"C:\Users\kunpa\Downloads\Projects\intra\data\ml\ranker_null\combo_overlap_labels.parquet"
LOCAL_LOG = r"C:\Users\kunpa\Downloads\Projects\intra\tasks\_overlap_labels_remote.log"


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    try:
        sftp = client.open_sftp()
        try:
            # Confirm remote file exists and capture size
            for remote, local in [
                (REMOTE_PARQUET, LOCAL_PARQUET),
                (REMOTE_LOG, LOCAL_LOG),
            ]:
                try:
                    attrs = sftp.stat(remote)
                    print(f"[sftp] {remote}: {attrs.st_size} bytes")
                except FileNotFoundError:
                    print(f"[sftp] ERROR: {remote} not found on remote", file=sys.stderr)
                    return 2
                os.makedirs(os.path.dirname(local), exist_ok=True)
                sftp.get(remote, local)
                local_size = os.path.getsize(local)
                print(f"[sftp] pulled -> {local} ({local_size} bytes)")
        finally:
            sftp.close()
    finally:
        client.close()
    print("[sftp] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
