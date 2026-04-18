# Currently Running Processes

**Snapshot time**: 2026-04-18 ~09:16 UTC
**Host**: sweep-runner-1 (`195.88.25.157`)
**Purpose**: Live mapping of running processes to the Part B roadmap in
[`ml1_ml2_synthesis_roadmap.md`](ml1_ml2_synthesis_roadmap.md). Update this
file when jobs start/finish.

---

## Active screen sessions

| Screen | Started (UTC) | What | Status |
|---|---|---|---|
| `eval_nbs` | 2026-04-18 09:16 | v10+v11+v12 OOS eval notebook suite (36 notebooks, gross+net) under `systemd-run --scope -p MemoryMax=9G -p CPUQuota=280%` | **Active** (Phase 6.5) |

## Completed since last snapshot (2026-04-14)

- **v11 friction-aware sweep** (screen `v11_sweep`) — finished
  2026-04-18 03:09 UTC. 30,000 combos, 329.7 min wall.
  `data/ml/originals/ml_dataset_v11.parquet` (6.6 GB, 102M trade rows).
- **ML#1 v11 retrain + leakage fix** (screen `ml1_v11`) — finished
  2026-04-17 ~19:46 UTC. See `tasks/part_b_findings.md` Phase 6.3.
- **ML#1 v12 pipeline** (screen `v12_pipe`) — finished
  2026-04-18 ~09:07 UTC. OOF R² 0.93 / Spearman 0.96 on 13,814 combos.
- **v10_9264 post-mortem + eval artifacts publish** (screens `v10_9264`,
  `eval_publish`) — finished 2026-04-17.
- **Phase 2 V3+C1+Fixed500 held-out eval** (screen `phase2`) — finished
  2026-04-17.

## Queue status

- **Next blocker:** Phase 6.5 v12 OOS eval notebook run (currently
  active). Pull + commit when screen `eval_nbs` goes idle.
- **After Phase 6.5:** decide whether v12 top-10 OOS passes the
  published success criterion (median net Sharpe > 0, ≥ 5/10
  individually profitable). If yes, mark Phase 6 closed and consider
  paper-trade (B17) or v13 calendar-time walk-forward.

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
