# Currently Running Processes

**Snapshot time**: 2026-04-14 ~05:50 UTC
**Host**: sweep-runner-1 (`195.88.25.157`)
**Purpose**: Live mapping of running processes to the Part B roadmap in
[`ml1_ml2_synthesis_roadmap.md`](ml1_ml2_synthesis_roadmap.md). Update this
file when jobs start/finish.

---

## Project Python jobs

| PID   | Started  | Elapsed | CPU  | RSS    | Script                              | Part B item |
|-------|----------|---------|------|--------|-------------------------------------|-------------|
| 27951 | 04:42    | ~1h08m  | 187% | 282 MB | `scripts/permutation_test_ml1.py`   | **B3** — Permutation test on ML#1 (shuffle target labels → retrain → confirm AUC collapses) |
| 28016 | 04:42    | ~1h08m  | 184% | 4.03 GB | `scripts/adaptive_rr_model_monotonic_v2.py`  | **B9** — Monotonic constraint on `candidate_rr` + `rr_x_atr` (wraps V2 with LightGBM `monotone_constraints=-1`) |

Both processes are `Rl` (running, multi-threaded). `b9` has not yet written
artifacts to `data/ml/adaptive_rr_b9/` — expected; the V2-with-constraint
training runs ~2000 rounds before saving model + plots.

## Non-project processes (ignore)

| PID | Process | Note |
|-----|---------|------|
| 809 | `/usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal` | Ubuntu system daemon. Sleeps for the machine's entire uptime, waiting for a shutdown signal so it can pause/resume package upgrades across reboot. 0% CPU, 12 MB RSS. **Not queued work — not related to this project.** |
| 9850, 9858, 9859, 9871, 9872, 10343, 11051, 11095, 11096, 13837, 14635, 15615 | `screen -S sweep` / `SCREEN -S sweep` | Nine detached shell sessions from the v2–v10 sweep runs on 2026-04-13. All idle — no active work inside them. Safe to leave or `screen -wipe` later. |

## Queue status

- **No third project job is queued.** Verified via `ps`, all screen sessions
  (idle shells), `crontab -l` (empty), `atq` (empty).
- Next Part B items per the roadmap's Week-1 sequence (not yet started):
  **B4** (filter on `surrogate_top_combos.csv`, ~10 min) and **B2** (adaptive
  E[R] percentile threshold, ~1 hr). These should launch once B3 + B9 finish.

---

## How to update this file

When a job starts or finishes:

```
ps -eo pid,etime,pcpu,pmem,rss,stat,cmd --sort=-pcpu | grep python
```

Then edit the table. Keep this file as the single authoritative "what's
running right now" reference — the roadmap file is the long-lived plan, this
one is the live dashboard.
