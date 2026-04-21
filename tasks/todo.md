# Phase 5 Post-Sweep Plan — Execute Probe 1 (15m + 1h Timeframe Lift)

**Date**: 2026-04-21 UTC
**Supersedes**: `tasks/root_ablation_criterion.md` (V3-no-memory refit moot — see Phase A).
**Driven by**: Chairman sweep (2026-04-21 AM) + LLM Council on within-NQ/MNQ
fork ordering (2026-04-21 12:50 CDT).

## The binding question (resolved 2026-04-21)

> The v11 sweep falsified 1-minute Z-score mean-reversion on NQ/MNQ (gross
> ceiling 1.108, one combo of 13,814). Within the NQ/MNQ envelope, is the
> **bar timeframe** axis carrying the failure — i.e. does the same signal
> family clear a tradeable gross Sharpe bar on 15-minute or 1-hour bars,
> and does that edge survive a combo-agnostic K-fold audit?

Council verdict selected the timeframe axis over signal-family swap (Probe 1
beats Probe 2) on two arguments: (a) cheaper kill-bar — if timeframe fails,
the Z-score family is falsified family-wide; (b) Probe 2's candidate signals
(momentum/breakout) have no harness prior in this repo, so a negative Probe 2
result is ambiguous ("bad signal or bad implementation?").

---

## Phase A — Close out the ablation (COMPLETED 2026-04-21)

- [x] `tasks/root_ablation_criterion.md` marked SUPERSEDED with rationale.
- [x] `lessons.md` entry `2026-04-21 check_universe_ceiling_before_ablation`
      added (prevention rule: compute gross ceiling before designing filter
      ablations).
- [x] Task #17 marked completed-superseded (audit trail preserved).

## Phase B — LLM Council on fork ordering (COMPLETED 2026-04-21 12:50 CDT)

LLM Council fired (Probe 3 of prior plan). Five advisors, anonymized peer
review, chairman synthesis. **Recommendation: Probe 1 (timeframe lift) at
15m AND 1h in parallel. Skip 5m** — Expansionist flagged 5m as correlated
with 1m's failure mode; if 5m passes and 15m fails, the signal-to-noise win
is marginal and sweep compute doubles.

**Unanimous peer-review finding** (all 5 reviewers flagged independently):
pre-registered combo-agnostic K-fold audit is non-negotiable. AUC parity
fooled the pipeline twice (V3 + V4); no future ML#2 stack ships without a
combo-agnostic partition refit passing the ship bar.

**Chairman's "One Thing To Do First"**: write `tasks/probe1_preregistration.md`
**before any code**. That document is Phase C0 below.

Council artifacts (filed under `tasks/council/`):
- [x] `tasks/council/council-report-2026-04-21-nq-mnq-fork.html`
- [x] `tasks/council/council-transcript-2026-04-21-nq-mnq-fork.md`

---

## Phase C — Execute Probe 1

### C0 — Pre-registration (Chairman's One Thing First, ~1-2h)

Write `tasks/probe1_preregistration.md`, mirroring the
`tasks/phase5_kill_criterion.md` template (Status / Decision Rule /
irrevocable commitments / Signature block with commit hash). Must contain,
**in this order**:

1. **Combo-level K-fold protocol**
   - `sklearn.model_selection.GroupKFold(n_splits=5)` with `groups=combo_id`.
   - Refit `adaptive_rr_v3` architecture per fold **sans** `global_combo_id`
     AND `prior_wr_{10,50}` + `prior_r_ma10` + `has_history_50` (the
     `v3_no_memory` feature set from the superseded ablation doc).
   - Each fold: train on 4/5 of combos' trades, predict OOS on held-out 1/5.
   - Filter basket = union of OOF predictions above the net-E[R] gate.
   - Cross-fold metrics: mean + std of s6_net Sharpe p50, ruin,
     combo-count-in-basket.

2. **Sunset threshold** (ties go to the stricter side)
   - **Gross pre-gate**: ≥10 combos with gross Sharpe ≥ 1.3 AND trades
     ≥ bar-count-adjusted `MIN_TRADES_GATE` (500 @ 1m → ~33 @ 15m → ~8 @ 1h,
     floored at 50 for statistical significance).
   - If gross pre-gate FAILS on BOTH timeframes → **family-level sunset**;
     no K-fold audit runs; Z-score mean-reversion on NQ/MNQ is declared
     falsified across the bar-timeframe axis.
   - If gross pre-gate PASSES on at least one → proceed to K-fold on the
     passing timeframe(s).
   - **Post-K-fold ship gate**: mean(cross-fold Sharpe p50) ≥ 1.0 AND
     mean(cross-fold ruin) ≤ 20% AND ≥10 combos survive the basket in at
     least 4 of 5 folds.

3. **Walk-forward time-slice definitions** (2019-2024 regime audit)
   - Chronological split of the sweep's OOS trade log into 4-5 calendar-year
     slices.
   - Sanity probe: pre-2024 K-fold performance must not exceed 2024 K-fold
     performance by > 30% — guards against regime-dependent overfit
     surviving the combo-partition.

4. **Microstructure sweep parameter spec** (Executor's "free-ride axis")
   - Add to the v11-class sweep:
     - `entry_timing_offset ∈ {0, 1, 2}` (bars delay after signal)
     - `fill_slippage_ticks ∈ {0, 1, 2}`
     - `cooldown_after_exit_bars ∈ {0, 3, 10}`
   - Total microstructure cells = 27 — keep cardinality low to avoid
     combinatorial blow-up against the existing v11 range.

5. **Timeframes**: 15m AND 1h (parallel sweeps). 5m is **deprecated** per
   council.

Plus irrevocable commitments carried from `phase5_kill_criterion.md`:
- Results read mechanically; no post-hoc reinterpretation of thresholds.
- No selective metric switching. Gross pre-gate reads zero-friction Sharpe;
  ship gate reads s6_net Sharpe p50 at $5/contract RT.
- AUC parity is **never** sufficient evidence for a ship.

User signs the preregistration (commit hash captured) **before any code runs**.

### C1 — Bar-aggregation infrastructure (~2-3h)

New module `src/indicators/bar_resample.py` exposing
`resample_bars(df_1min: DataFrame, freq: str) -> DataFrame`:
- OHLCV aggregation: `open=first`, `high=max`, `low=min`, `close=last`,
  `volume=sum`.
- `session_break = any` within window.
- `time` = window start (left-anchored — matches `next_bar_open` fill
  semantics).
- Drop partial windows at data boundaries.

Cache to disk (one-time, for reuse across sweeps + eval notebooks):
- `data/NQ_15min.parquet`
- `data/NQ_1h.parquet`

Validation (inline tests or scratch notebook):
- Round-trip: `resample_bars(df, '1min')` == input (modulo dtype).
- Volume conservation: `sum(resampled.volume) == sum(input.volume)`.
- Hand-check: pick 4 consecutive 15m windows, verify OHLCV against raw rows.

### C2 — Sweep harness timeframe support (~1-2h)

Extend `scripts/param_sweep.py`:
- Add `--timeframe {1min,15min,1h}` CLI flag (default `1min` preserves legacy
  behaviour).
- When non-default, load from `data/NQ_{15min,1h}.parquet`.
- Adjust `MIN_TRADES_GATE` proportional to bar count, floored at 50.
- Thread the microstructure sweep parameters (C0 item 4) into
  `_sample_combos` for `--range-mode v11`.

Smoke test: 100 combos @ 15min. Assert parquet writes cleanly, trade counts
in reasonable range (50-5000 per combo), no dtype errors. Pre-sweep gate per
`CLAUDE.md` before C3.

### C3 — Parallel sweeps (remote, ~12-24h unattended)

Launch on sweep-runner-1 using `scripts/runners/run_v10_sweep_remote.py` as
template:
- **Sweep A**: `--timeframe 15min --range-mode v11 --combinations 3000`
  → `data/ml/originals/ml_dataset_v11_15m.parquet`
- **Sweep B**: `--timeframe 1h --range-mode v11 --combinations 1500`
  → `data/ml/originals/ml_dataset_v11_1h.parquet` (smaller N — hourly sweep
  has less combinatorial richness per unit of compute).
- Run under `systemd-run --scope -p MemoryMax=9G -p CPUQuota=280%`.
- Monitor every 10 min per `feedback_poll_interval.md`.

### C4 — Gross-ceiling readout (~30 min)

Adapt `scripts/analysis/_sharpe_distribution.py` (or its v11 equivalent) for
each new parquet:
- Compute zero-friction gross Sharpe per combo (trade-level annualized).
- Apply bar-count-adjusted `MIN_TRADES_GATE`.
- Count combos at gross Sharpe ≥ 1.3.

**Branch A — Sunset**: both 15m and 1h have <10 combos clearing 1.3 gross
Sharpe at the trade-count floor. Z-score mean-reversion on NQ/MNQ is
falsified across the bar-timeframe axis. Document in `lessons.md`; update
`CLAUDE.md` to deprecate the family; fresh council question for the next
fork.

**Branch B — K-fold audit**: at least one timeframe clears the pre-gate.
Run C5 on the passing timeframe(s).

### C5 — Combo-agnostic K-fold audit (~3-5h per timeframe)

Only for timeframes that cleared C4.
- Build `combo_features_v{15m,1h}.parquet` from the sweep output (reuse
  `scripts/models/ml1_surrogate_v12.py`'s feature-engineering path, adapted
  for the new parquet).
- Refit `adaptive_rr_model_v3.py --no-memory --n_folds 5 --group_col combo_id`
  (new flags; thin wrapper around existing LightGBM + isotonic pipeline).
- Run s6_net MC (n_sims=2000) on the OOF basket.
- Report cross-fold mean ± std of Sharpe p50, ruin, basket size.

Apply the ship gate mechanically (C0 item 2). Write results to
`evaluation/probe1_{15m,1h}_kfold/` matching the standard eval-notebook
layout.

### C6 — Verdict document (~1h)

Write `tasks/probe1_verdict.md`:
- Mechanical readout of C4 + C5 against the pre-registered thresholds.
- Branch decision.
- `lessons.md` entry (what worked / what didn't at the new timeframe; any
  leak carryover detected).
- Next action: either production wiring (ship the new timeframe) or scope
  the next fork.

---

## Deferred / not-now

- **5-minute bar probe** — explicitly deprecated by council. Revive only if
  both 15m and 1h clear gross pre-gate and an ablation-of-the-ablation is
  warranted.
- **Signal-family swap (Probe 2 from prior plan)** — defer until Probe 1
  resolves. If Probe 1 sunsets (Branch A), this becomes the next fork
  question (fresh council).
- **Formal friction-sensitivity curve** — still low incremental information;
  resurrect only for write-up visuals.
- **ML#2 redesign at trade-grain per-combo (Expansionist's Option E)** —
  blocked by `tasks/phase5_kill_criterion.md` irrevocable commitment #10.
  Not reopenable until a combo-agnostic edge is demonstrated elsewhere.

---

## Compute budget

| Phase | Wall-clock | Notes |
|---|---|---|
| C0 pre-registration | 1-2h human | **Blocking** — no code before sign-off |
| C1 bar aggregation | 2-3h | New `src/indicators/bar_resample.py` |
| C2 sweep harness | 1-2h + smoke | Pre-sweep gate required |
| C3 parallel sweeps | 12-24h unattended | Remote, poll 10min |
| C4 gross readout | 30 min | Branch decision |
| C5 K-fold audit | 3-5h × ≤2 timeframes | Remote |
| C6 verdict | 1h | Mechanical |
| **Total** | 1-2 days if C5 runs; ~1 day if C4 sunsets | |

---

## Recommendation

**Execute C0 now** (write `tasks/probe1_preregistration.md`) as the
Chairman's One Thing First. After user sign-off (commit-hashed), C1-C6
proceed mechanically with no further decision points until the C4 branch.
