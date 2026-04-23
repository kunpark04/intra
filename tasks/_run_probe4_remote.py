"""Run Probe 4 (combos 1298 + 664, SES_0 only) end-to-end on sweep-runner-1.

Workflow:
  1. SSH to sweep-runner-1 (paramiko) + stash any remote-side edits.
  2. git fetch + reset --hard origin/master (picks up Probe 4 script bundle).
     Per `feedback_remote_git_sync.md`: git-pull, never SFTP-patch src/.
  3. Remove stale Cython .so so Numba fallback engages (probe1/2/3 pattern).
  4. Verify 1h bar cache + cached param dicts present.
  5. Write /root/intra/run_probe4.sh wrapper that invokes `_probe4_run_combo.py`
     twice (one per combo, SES_0 only), records per-run exit codes, then
     `_probe4_readout.py`. Wrapper does NOT use `set -e` — each engine
     invocation's exit code is captured to `data/ml/probe4/run_status.json`
     and the readout gates verdict emission on all runs succeeding. Post-hoc
     SES_1 RTH / SES_2 GLOBEX partitioning happens in the readout, not the
     engine (see C1 fix: engine session_filter_mode is UTC-hour, not ET-minute).
  6. Launch in detached screen `probe4`.
  7. Poll every 60 s until screen terminates or 30-min timeout. On timeout,
     unexpected exit, or KeyboardInterrupt, send `screen -S probe4 -X quit`
     to the remote to avoid ghost sessions.
  8. SFTP-pull per-run JSONs + SES_0 per-trade parquets + readout.json +
     run_status.json.
  9. Tail remote log and print the last 120 lines locally.

Output artifacts pulled locally to data/ml/probe4/:
  combo{1298,664}_SES_0.json
  combo{1298,664}_SES_0.parquet
  combo{1298,664}_SES_0_combos.json
  combo{1298,664}_SES_0_trades.parquet
  run_status.json
  readout.json

Each `_probe4_run_combo.py` invocation runs under
  systemd-run --scope -p MemoryMax=9G

(per `feedback_kamatera_cpu_cap.md` + the memory-cap convention shared with
 `run_eval_notebooks_remote.py`). CPUQuota is NOT set — 2 runs are short and
 sequential (~4 min total), not the long-running LightGBM workload for which
 the sustained-70% cap was introduced.

Preregistration: tasks/probe4_preregistration.md (signed commit 432fb3d).
Precedent: tasks/_run_probe3_remote.py.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import paramiko

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST = "195.88.25.157"
USER = "root"
PWD  = "J@maicanP0wer123"  # noqa: S105 — local-only ref

SCREEN_NAME  = "probe4"
LOG_PATH     = f"/root/intra/logs/{SCREEN_NAME}.log"
WRAPPER_PATH = f"/root/intra/run_{SCREEN_NAME}.sh"

# 2-run suite: one SES_0 engine call per combo. Post-hoc ET-minute session
# partitioning happens in _probe4_readout.py (engine session_filter_mode
# branches are UTC-hour, which cannot produce the prereg §4.4 ET definitions).
COMBO_IDS: list[int] = [1298, 664]

MAX_WAIT_SECONDS = 30 * 60  # §8.4 budget ~4 min + generous buffer
POLL_INTERVAL    = 60


def _build_wrapper() -> str:
    """Wrapper intentionally does NOT use `set -e`. Per-run exit codes are
    captured into run_status.json; the readout refuses to emit a verdict if
    any run failed, preserving fail-fast at the readout boundary."""
    lines: list[str] = [
        "#!/bin/bash",
        # No `set -e` — we want all runs to attempt even if an earlier one
        # fails, so the readout can surface which step(s) failed.
        "cd /root/intra",
        "",
        'echo "=== Probe 4 start $(date -u) ==="',
        "git rev-parse HEAD",
        "",
        "mkdir -p data/ml/probe4 logs",
        'echo "{" > data/ml/probe4/run_status.json',
        "",
    ]
    for i, cid in enumerate(COMBO_IDS):
        is_last = (i == len(COMBO_IDS) - 1)
        comma = "" if is_last else ","
        lines.extend([
            "",
            f'echo ""',
            f'echo "=== combo {cid}  SES_0 ==="',
            "t0=$(date +%s)",
            "systemd-run --scope \\",
            "    -p MemoryMax=9G \\",
            f"    python3 tasks/_probe4_run_combo.py --combo-id {cid}",
            f'rc_{cid}=$?',
            f'echo "combo{cid}_SES_0 exit_code: $rc_{cid}  wall-clock: $(($(date +%s) - t0)) s"',
            f'echo "  \\"combo{cid}\\": $rc_{cid}{comma}" >> data/ml/probe4/run_status.json',
        ])
    lines.extend([
        'echo "}" >> data/ml/probe4/run_status.json',
        "",
        'echo ""',
        'echo "=== run_status.json ==="',
        "cat data/ml/probe4/run_status.json",
        "",
        'echo ""',
        'echo "=== aggregate readout ==="',
        "python3 tasks/_probe4_readout.py",
        "readout_rc=$?",
        'echo "readout exit_code: $readout_rc"',
        "",
        'echo ""',
        'echo "=== artifacts ==="',
        "ls -la data/ml/probe4/",
        "",
        'echo "=== Probe 4 end $(date -u) ==="',
        "",
    ])
    return "\n".join(lines)


def run(c: paramiko.SSHClient, cmd: str, label: str | None = None,
        timeout: int = 120) -> tuple[str, str]:
    if label:
        print(f"\n-- {label} --")
        print(f"$ {cmd}")
    _, out, err = c.exec_command(cmd, timeout=timeout)
    so = out.read().decode(errors="replace")
    se = err.read().decode(errors="replace")
    if so:
        print(so.rstrip())
    if se.strip():
        print(f"[stderr] {se.rstrip()}")
    return so, se


def _kill_screen(client: paramiko.SSHClient) -> None:
    """Send quit to the probe4 screen session if it's still alive. Idempotent
    — harmless if the session already terminated."""
    try:
        run(client,
            f"screen -S {SCREEN_NAME} -X quit 2>&1 || true",
            f"kill screen({SCREEN_NAME})")
    except Exception as e:
        print(f"[warn] screen cleanup failed: {e!r}")


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    client.connect(HOST, username=USER, password=PWD, timeout=30)

    launched = False
    try:
        # 1. Stash remote-side edits + align with origin/master
        run(client,
            "cd /root/intra && git stash push -u -m 'pre-probe4-'$(date +%s) 2>&1 | head -3 || true",
            "stash remote edits")
        run(client, "cd /root/intra && git fetch origin master 2>&1 | tail -5", "git fetch")
        run(client,
            "cd /root/intra && git checkout master 2>&1 | head -3 "
            "&& git reset --hard origin/master 2>&1 | head -3",
            "align master with origin/master")
        run(client, "cd /root/intra && git rev-parse HEAD", "remote HEAD after checkout")

        # 2. Remove Cython .so — force Numba fallback (probe1/2/3 convention)
        run(client,
            "rm -f /root/intra/src/cython_ext/*.so /root/intra/src/cython_ext/*.c && echo ok",
            "clear Cython .so")

        # 3. Verify required inputs present
        run(client,
            "ls -la /root/intra/data/NQ_1h.parquet "
            "/root/intra/tasks/_probe4_param_dicts.json "
            "/root/intra/tasks/_probe4_run_combo.py "
            "/root/intra/tasks/_probe4_readout.py 2>&1",
            "verify 1h bar cache + param dicts + probe4 scripts present")

        # 4. Write wrapper via SFTP
        print("\n-- write wrapper --")
        sftp = client.open_sftp()
        wrapper_body = _build_wrapper()
        with sftp.file(WRAPPER_PATH, "w") as f:
            f.write(wrapper_body)
        sftp.chmod(WRAPPER_PATH, 0o755)
        print(f"[ok] wrote {WRAPPER_PATH}  ({len(wrapper_body)} bytes, "
              f"{len(COMBO_IDS)} runs)")

        # 5. Launch in detached screen
        run(client,
            f"cd /root/intra && mkdir -p logs && rm -f {LOG_PATH} "
            f"&& screen -dmS {SCREEN_NAME} bash -c '{WRAPPER_PATH} > {LOG_PATH} 2>&1' "
            f"&& sleep 5",
            f"launch screen({SCREEN_NAME})")
        launched = True
        run(client, "screen -ls 2>&1 || true", "screen -ls (confirm detached)")

        # 6. Poll until completion or 30-min cap
        print("\n-- polling --")
        elapsed = 0
        timed_out = True
        while elapsed < MAX_WAIT_SECONDS:
            so, _ = run(client,
                        f"screen -ls 2>/dev/null | grep -c {SCREEN_NAME} || true",
                        None, timeout=30)
            alive = so.strip() and so.strip() != "0"
            run(client,
                f"tail -4 {LOG_PATH} 2>/dev/null || echo '(no log yet)'",
                f"t+{elapsed}s — tail")
            if not alive:
                print(f"\n[ok] screen session terminated after ~{elapsed} s")
                timed_out = False
                break
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

        if timed_out:
            print(f"\n[WARN] {MAX_WAIT_SECONDS // 60}-min abort threshold reached. "
                  f"Killing ghost screen session.")
            _kill_screen(client)
            launched = False

        # 7. Tail full log for visibility
        run(client, f"tail -120 {LOG_PATH} 2>&1", "full log tail")

        # 8. Pull artifacts
        print("\n-- pulling artifacts --")
        local_base = Path("data/ml/probe4")
        local_base.mkdir(parents=True, exist_ok=True)

        files_to_pull: list[str] = [
            "data/ml/probe4/readout.json",
            "data/ml/probe4/run_status.json",
        ]
        for cid in COMBO_IDS:
            tag = f"combo{cid}_SES_0"
            files_to_pull.append(f"data/ml/probe4/{tag}.json")
            files_to_pull.append(f"data/ml/probe4/{tag}.parquet")
            files_to_pull.append(f"data/ml/probe4/{tag}_combos.json")
            files_to_pull.append(f"data/ml/probe4/{tag}_trades.parquet")

        for rel in files_to_pull:
            remote_p = f"/root/intra/{rel}"
            local_p  = rel
            try:
                st = sftp.stat(remote_p)
                Path(local_p).parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_p, local_p)
                print(f"[ok] pulled {local_p}  ({st.st_size:,} bytes)")
            except FileNotFoundError:
                print(f"[miss] {remote_p}  (not found on remote)")
            except Exception as e:
                print(f"[err] {remote_p}: {e!r}")

    except KeyboardInterrupt:
        print("\n[interrupt] KeyboardInterrupt — killing remote screen before exit.")
        if launched:
            _kill_screen(client)
        raise
    except Exception:
        # Any unexpected exception path must also clean up the screen.
        if launched:
            print("\n[error] unexpected exit — killing remote screen before re-raising.")
            _kill_screen(client)
        raise
    finally:
        client.close()

    print("\n[done] probe 4 remote run complete — next: review "
          "data/ml/probe4/readout.json and draft tasks/probe4_verdict.md")


if __name__ == "__main__":
    main()
