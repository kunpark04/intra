# Currently Running Processes

**Snapshot time**: 2026-04-18 ~13:58 UTC
**Host**: sweep-runner-1 (`195.88.25.157`)
**Purpose**: Live mapping of running processes to the Part B roadmap in
[`ml1_ml2_synthesis_roadmap.md`](ml1_ml2_synthesis_roadmap.md). Update this
file when jobs start/finish.

---

## Active screen sessions

| Screen | Started (UTC) | What | Status |
|---|---|---|---|
| _(none — Tier 1 complete at 14:46 UTC)_ | — | — | — |

## Completed since last snapshot (2026-04-18 09:57)

- **Tier 1 v12 variants OOS eval** (2026-04-18 14:46 UTC) — 16 net-of-cost
  notebooks across v12_topk_{k05,k10,top50}_net. Top-50 is best result
  yet: +0.54 combined Sharpe, +141% return, 81% DD. κ>0 variants failed.
  MC pair (s3/s6) skipped for top-50 after 10k×50 sim OOM on 9 GB cap.
  Full results in Phase 6.6 of `tasks/part_b_findings.md`.

- **v10+v11+v12 OOS eval notebook suite** (screen `eval_nbs`) — finished
  2026-04-18 09:35:41 UTC. 20 wall-minutes for 36 notebooks under
  `systemd-run --scope -p MemoryMax=9G -p CPUQuota=280%`. All notebooks
  pulled locally via `_pull_eval_nbs.py`. See Phase 6.5 verdict in
  `tasks/part_b_findings.md`.
- **v11 friction-aware sweep** — 2026-04-18 03:09 UTC (30k combos, 329.7 min).
- **ML#1 v11 retrain + leakage fix** — 2026-04-17 19:46 UTC.
- **ML#1 v12 parameter-only pipeline** — 2026-04-18 09:07 UTC.
- **v10_9264 post-mortem + eval publish** — 2026-04-17.
- **Phase 2 V3+C1+Fixed500 held-out eval** — 2026-04-17.

## Queue status

- **No active remote jobs.** Phase 6.5 v12 OOS eval complete and pulled.
- **Phase 6.5 verdict:** FAIL on the published success criterion
  (median net Sharpe > 0). v12 top-10 fixed$500 net: 5/10 individually
  profitable, median Sharpe −0.11, combined portfolio Sharpe +0.24
  (return +27%, DD 62%). ML#2 V3 filter rejects all v12 top-10 trades
  (0 trades under filter) — V3 was calibrated on v2–v10 economics, v11
  combo space is out-of-distribution.
- **Next decision:** either (a) accept v12 with a revised success gate
  keyed to combined-portfolio Sharpe, (b) rebuild V3 on v11 sweep data,
  or (c) escalate to v13 calendar-time walk-forward (originally queued
  as the next step if v12 failed).

---

## How to update this file

When a job starts or finishes:

```
screen -ls
ps -eo pid,etime,pcpu,pmem,rss,stat,cmd --sort=-pcpu | grep python
```

Then edit the table. Keep this file as the single authoritative "what's
running right now" reference — the roadmap file is the long-lived plan,
this one is the live dashboard.
