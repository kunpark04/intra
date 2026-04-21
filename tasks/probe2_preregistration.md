# Probe 2 Pre-Registration — Combo-865 Isolation on Holdout Bars

**Date**: 2026-04-21 UTC
**Authority**: LLM Council verdict (Probe 1 Branch A fork),
`tasks/council-report-2026-04-21-probe1-branch-a-fork.html`
**Template**: Mirrors `tasks/probe1_preregistration.md`.
**Process discipline**: Thresholds committed before the experiment runs,
applied mechanically after. Ties go to the stricter (sunset) side.

---

## Status

**SIGNED 2026-04-21 UTC (commit recorded at signing ceremony, see §9).**

Probe 2 execution (combo-865 remote run on test partition + verdict) may
now proceed. The decision rule in §3 is binding regardless of which branch
the observed metrics fall into.

---

## 1. Purpose

Test whether combo 865's cross-timeframe coherence observed on the Probe 1
**training** partitions (top-10 on both 15m and 1h: gross Sharpe 1.386 on
2,786 trades and 2.098 on 747 trades respectively) is a **real edge** or a
**3000-combo-sweep coincidence**.

The probe answers one compound binary question:

- **Combo-isolation gate**: does combo 865, run with its exact v11-sweep
  parameter dict on the held-out 20% of calendar bars (post-training
  partition, 2024-10-22 → 2026-04-08), produce a net-of-friction outcome
  that simultaneously clears (a) a statistical Sharpe threshold and (b) an
  economic net-dollar floor on the project's stated capital base ($50k)?

PASS → authorize Probe 3 (Option Y session-structure sweep).
FAIL → Option Z (project sunset of the Z-score mean-reversion family on
NQ/MNQ).

This probe does not reopen the bar-timeframe axis: Probe 1 Branch A remains
terminal for intermediate timeframes (§7.6 of probe1_preregistration).
Probe 2 tests a single combo's persistence at its own two timeframes, not
the family as a whole.

---

## 2. Combo Under Test

`combo_id = 865` from the v11 sampler (`--range-mode v11 --seed 0`).

**Parameter dict** (fully specified, verified identical across
`data/ml/originals/ml_dataset_v11_15m.parquet` and
`data/ml/originals/ml_dataset_v11_1h.parquet` — 0 mismatches across 19
parameter columns):

| Parameter | Value |
|---|---|
| `z_band_k` | 2.304384 |
| `z_window` | 41 |
| `z_input` | `returns` |
| `z_anchor` | `rolling_mean` |
| `z_denom` | `parkinson` |
| `z_type` | `parametric` |
| `z_window_2` | 47 |
| `z_window_2_weight` | 0.330829 |
| `volume_zscore_window` | 47 |
| `ema_fast` | 6 |
| `ema_slow` | 48 |
| `stop_method` | `fixed` |
| `stop_fixed_pts` | 17.017749 |
| `atr_multiplier` | NaN (unused with fixed-stop) |
| `swing_lookback` | NaN (unused with fixed-stop) |
| `min_rr` | 1.847651 |
| `max_hold_bars` | 120 |
| `exit_on_opposite_signal` | False |
| `use_breakeven_stop` | False |
| `zscore_confirmation` | False |
| `volume_entry_threshold` | 0.0 |
| `vol_regime_lookback` | 0 |
| `vol_regime_min_pct` | 0.0 |
| `vol_regime_max_pct` | 1.0 |
| `session_filter_mode` | 0 |
| `tod_exit_hour` | 0 |
| `entry_timing_offset` | 1 |
| `fill_slippage_ticks` | 1 |
| `cooldown_after_exit_bars` | 3 |

These values are **frozen at signing**. Any deviation requires a written
amendment to this document pre-commit.

---

## 3. Protocol

### Partitions

- **Data source**: `data/NQ_1min.csv` (2,573,239 valid 1-minute bars).
- **Split**: chronological 80/20 at the 1-minute level.
  - **Train** (excluded from Probe 2 read): bars 0–2,058,590; 2019-01-01
    17:01 → 2024-10-22 05:05 (~5.92 years).
  - **Test** (Probe 2 read set): bars 2,058,591–2,573,238; 2024-10-22 05:06
    → 2026-04-08 20:20 (~1.48 years).
- **Bar resampling**: `src/indicators/bar_resample.py` applied to the test
  partition, left-anchored / partial-window-dropped semantics (identical
  to Probe 1).

### Timeframes tested

- **15-minute**: 15m bars built from the test partition's 1-min data.
- **1-hour**: 1h bars built from the test partition's 1-min data.

1-minute is NOT in scope for this probe; combo 865's cross-TF signal in
Probe 1 was between 15m and 1h, and the 1-minute v11 sweep indexed combos
differently (combinations = 13,814), so `combo_id = 865` at 1m would be a
different parameter dict.

### Execution

Re-invoke `scripts/param_sweep.py` with identical sampler arguments as the
training sweeps, pinned to combo 865 only, on the test partition:

```
# 15m test partition  
python3 scripts/param_sweep.py \
    --combinations 3000 \
    --start-combo 865 --end-combo 866 \
    --seed 0 --range-mode v11 \
    --timeframe 15min --eval-partition test \
    --output data/ml/probe2/combo865_15m_test.parquet --workers 1

# 1h test partition
python3 scripts/param_sweep.py \
    --combinations 1500 \
    --start-combo 865 --end-combo 866 \
    --seed 0 --range-mode v11 \
    --timeframe 1h --eval-partition test \
    --output data/ml/probe2/combo865_1h_test.parquet --workers 1
```

Each invocation runs ONE combo on test partition bars. No modifications to
`param_sweep.py` are required — `--start-combo` and `--end-combo` already
support single-combo execution, and `--eval-partition test` already points
at the held-out 20%.

### Metrics (per timeframe)

For each timeframe `tf ∈ {15m, 1h}`:

- `n_trades(tf)`: count of closed trades on the test partition.
- `net_sharpe(tf)`: annualized net-of-friction Sharpe on test-partition
  trade log.
  - Formula: `mean(net_pnl_dollars) / std(net_pnl_dollars, ddof=1) *
    sqrt(n_trades / YEARS_SPAN_TEST)`
  - `YEARS_SPAN_TEST = 1.4799` (from bar-count / (23 × 252 × 60) on test
    partition; computed at signing).
  - `net_pnl_dollars` is already net of `$5/contract RT` friction per the
    v11 sweep engine (see `CLAUDE.md` Parameter-sweep block).
- `net_dollars_per_year(tf)`: mean net P/L annualized.
  - Formula: `mean(net_pnl_dollars) * (n_trades / YEARS_SPAN_TEST)`
- Also logged for audit (not gate-bound): `gross_sharpe`, `win_rate`,
  train-vs-test ratio for each metric.

---

## 4. Gates (binding, mechanically applied)

**All three conditions must hold on AT LEAST ONE timeframe** (either 15m
or 1h, measured on the same timeframe — no cross-timeframe mixing):

1. **Statistical-Sharpe gate**: `net_sharpe(tf) >= 1.3`
2. **Statistical-sample gate**: `n_trades(tf) >= 50`
3. **Economic-floor gate**: `net_dollars_per_year(tf) >= 5000`
   (equivalent to $5,000/year on $50,000 starting equity at flat $500
   risk — 10% net-of-friction annual return on stated capital)

### Branches

**PASS — Probe 3 authorized**:
- At least one of `{15m, 1h}` satisfies gates (1) AND (2) AND (3)
  simultaneously.
- Next action: draft `tasks/probe3_preregistration.md` for the full Option
  Y session-structure sweep (3000 combos × session_filter ∈ {all,
  RTH-only, RTH-lunch-exclude, overnight-only}) on the test-passing
  timeframe. Spawn a fresh LLM Council BEFORE signing Probe 3
  (probes shouldn't chain without deliberation on each fork).

**FAIL — Option Z (project sunset)**:
- Both timeframes fail at least one of (1), (2), (3).
- Next action: write `tasks/project_sunset_verdict.md` declaring
  Z-score mean-reversion on NQ/MNQ terminally retired. Update `CLAUDE.md`
  with a sunset banner. Add a lesson to `lessons.md` on why a cross-TF
  stable combo in training does NOT translate to a post-hoc isolation test
  ( = lesson about the null-hypothesis coincidence rate in a 3000-combo
  sweep).

### Tie-breaking

- `net_sharpe = 1.29`, `n_trades = 49`, or `net_dollars_per_year = 4999`
  → FAIL. "Approximately meets" is not meets.
- Cross-timeframe mixing is NOT admissible (e.g., a 15m Sharpe pass + 1h
  economic pass = FAIL, because no single timeframe cleared all three
  gates simultaneously).

---

## 5. Irrevocable Commitments

1. **Results read mechanically.** No post-hoc reinterpretation of
   thresholds once the test-partition metrics are read.
2. **Combo 865 parameters are frozen.** The param dict in §2 is the
   complete specification. No substitution, no "combo 865-ish" alternates,
   no neighboring-param probes without a new preregistration.
3. **No post-hoc timeframe switching.** If both 15m and 1h fail, running
   on 1-minute, 30-minute, or any other timeframe is NOT admissible within
   Probe 2. §7.6 of Probe 1 remains binding for the bar-timeframe axis.
4. **No post-hoc gate relaxation.** Specifically: "passed by 1" is not
   grounds to re-run with slightly different thresholds. If 15m yields net
   Sharpe 1.29 AND economic $4999, that is a FAIL, and Option Z fires.
5. **Scope lock stands.** NQ/MNQ only (`memory/feedback_nq_mnq_scope_only.md`).
6. **Probe 3 still requires its own council.** A Probe 2 PASS does NOT
   skip the council step for Probe 3 — a session-structure probe has its
   own axis decisions (which filter set, which parameter holdover rule)
   that need deliberation.

---

## 6. Compute Budget

| Step | Wall-clock | Notes |
|---|---|---|
| Remote git-pull + env sync | 2 min | Leveraging the probe1 launcher path |
| 15m probe (1 combo × test) | 5–15 min | Single-combo; fast |
| 1h probe (1 combo × test) | 3–10 min | Single-combo; fast (fewer bars) |
| Pull artifacts + local readout | 5 min | SFTP + `_probe2_readout.py` |
| Verdict document | 30 min | `tasks/probe2_verdict.md` |
| **Total wall-clock** | **~30 min compute + 30 min writeup** | |

If either single-combo sweep exceeds **60 minutes**, abort and investigate
— something is wrong with the sweep harness or data, and the abort is NOT
a FAIL (abort is an out-of-band event that triggers a compute investigation
separate from the gate-reading discipline).

---

## 7. References

- Probe 1 verdict (trigger): `tasks/probe1_verdict.md`
- Probe 1 preregistration (precedent + §7.6 terminal): `tasks/probe1_preregistration.md`
- Council verdict (authority): `tasks/council-report-2026-04-21-probe1-branch-a-fork.html`
- Council transcript: `tasks/council-transcript-2026-04-21-probe1-branch-a-fork.md`
- Scope lock: `memory/feedback_nq_mnq_scope_only.md`
- Bar resampling module: `src/indicators/bar_resample.py`
- Sweep script: `scripts/param_sweep.py`
- Remote workflow: `memory/reference_remote_job_workflow.md`
- CPU/memory envelope: `memory/feedback_kamatera_cpu_cap.md`,
  `memory/feedback_lgbm_threads.md` (LightGBM thread cap n/a for this
  probe since Probe 2 runs the backtest engine only, no ML training).

---

## 8. Signature

By signing below, the signer commits to this document's decision rule
without post-hoc reinterpretation. §4 thresholds are mechanically applied
to the observed metrics. §5's irrevocable commitments stand.

- **Signed**: kunpa (confirmed via explicit "Conduct Probe 2 remotely"
  reply 2026-04-21 UTC)
- **Date/time of signature**: 2026-04-21 UTC
- **Commit hash at time of signature**: (populated at commit time; see
  `git log` immediately after this file lands)
