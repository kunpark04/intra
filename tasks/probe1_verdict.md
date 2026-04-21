# Probe 1 Verdict — 15m + 1h Timeframe Lift

**Date**: 2026-04-21 UTC
**Preregistration**: `tasks/probe1_preregistration.md` (signed 2026-04-21
18:11:57 UTC at commit `d0ee506`)
**Readout script**: `tasks/_probe1_gross_ceiling.py`
**Per-combo CSV dumps**: `tasks/probe1_15m_gross_sharpe.csv`,
`tasks/probe1_1h_gross_sharpe.csv`

---

## 1. Verdict (one line)

**BRANCH A — family-level sunset.** Z-score mean-reversion on NQ/MNQ is
declared falsified across the bar-timeframe axis (1min, 15min, 1h).
No K-fold audit runs.

---

## 2. Observed Metrics

Formula (mirrors `scripts/analysis/build_combo_features_ml1_v12.py:254-260`):

```
sharpe = mean(gross_pnl_dollars) / std(gross_pnl_dollars, ddof=1)
           * sqrt(n_trades / YEARS_SPAN_TRAIN)
YEARS_SPAN_TRAIN = 5.8056
MIN_TRADES_GATE  = 50
```

### Gross-ceiling readout

| Timeframe | Sweep combos | Combos w/ trades | Gated (n≥50) | max Sharpe | **N(Sharpe ≥ 1.3)** |
|---|---|---|---|---|---|
| 1min (v11 baseline, reference) | 16,384 | 13,814 | 13,814 | **1.108** | **1** |
| **15min (this probe)** | 3,000 | 2,308 | 1,306 | **1.817** | **9** |
| **1h (this probe)**   | 1,500 | 1,078 |   515 | **2.272** | **4** |

### Gated distribution (15m)

| quantile | gross Sharpe |
|---|---|
| p50 | −0.085 |
| p75 | +0.304 |
| p90 | +0.619 |
| p95 | +0.795 |
| p99 | +1.167 |
| max | +1.817 |

| threshold | count |
|---|---|
| ≥ 0.5 | 191 |
| ≥ 1.0 | 25 |
| ≥ 1.3 | **9** |
| ≥ 1.5 | 3 |
| ≥ 2.0 | 0 |

### Gated distribution (1h)

| quantile | gross Sharpe |
|---|---|
| p50 | −0.191 |
| p75 | +0.157 |
| p90 | +0.423 |
| p95 | +0.601 |
| p99 | +1.188 |
| max | +2.272 |

| threshold | count |
|---|---|
| ≥ 0.5 | 38 |
| ≥ 1.0 | 8 |
| ≥ 1.3 | **4** |
| ≥ 1.5 | 4 |
| ≥ 2.0 | 2 |

---

## 3. Mechanical Rule Application (preregistration §3)

- **Threshold**: N_1.3 ≥ 10 on either timeframe ⇒ Branch B (proceed to K-fold).
- **Observed**: N_1.3(15m) = 9, N_1.3(1h) = 4.
- **Tie-breaking (§3 line 152–153)**: "`N_1.3 = 9` on both timeframes →
  treat as Branch A (sunset)." 15m is at 9, 1h is below 9. Strictly
  worse than the explicit tie example. **Branch A fires.**
- **§7.5**: Ties go to the stricter side.
- **§7.1**: No post-hoc reinterpretation of thresholds.
- **§7.6**: FAIL is terminal for this axis. Intermediate timeframes
  (30min, 2h, 5min, etc.) are not admissible without a new preregistration
  cycle and fresh council.

---

## 4. Observations (not reinterpretations)

These are recorded for the council's next-step deliberation. They do NOT
alter the verdict.

1. **15m was one combo short of the gate.** Under the strict rule this is
   Branch A. But the ordinal movement from 1min (N_1.3 = 1) to 15m
   (N_1.3 = 9) is a **9× increase** in combos at the target Sharpe bar. The
   gross ceiling also rose from 1.108 (1min) → 1.817 (15m) → 2.272 (1h).
   The bar-timeframe axis is not *structurally dead*; it is *insufficient
   by the pre-registered bar* to justify continued spend.

2. **Cross-timeframe stable combo**: `combo_id = 865` appears in the top 10
   on BOTH timeframes (15m: 1.386 on 2,786 trades; 1h: 2.098 on 747
   trades). This is a single observation and cannot support a ship, but it
   is the first cross-timeframe coherent signal seen in the project.

3. **Best 1h combo (Sharpe 2.272 on 498 trades)** is the highest gross
   Sharpe ever observed in this project. It did not translate into a
   10-combo basket, which is what the preregistration required.

4. **Microstructure axes (entry_timing_offset, fill_slippage_ticks,
   cooldown_after_exit_bars) did not concentrate the ceiling.** The top
   combos span all three values of each axis; no obvious "free-ride"
   configuration emerged. Full microstructure coverage on 15m confirmed
   (27/27 cells populated, 69-106 combos/cell).

5. **Trade-count attrition**: 15m dropped 47% of combos below the
   trade-count gate (1306/2308 gated out of 2308 with any trades, from
   3000 sampled). 1h dropped 52% (515/1078). The family is not only
   ceiling-bound but also **sparsity-bound** at slower timeframes —
   many parameter combos produce fewer than 50 trades in 5.8 years.

---

## 5. Required Follow-Up Actions (preregistration §3 Branch A)

The preregistration specifies three next actions on Branch A. This verdict
discharges items 1 + 2 inline; item 3 (fresh council) is spawned separately.

### 5.1 — Deprecate Z-score mean-reversion family in CLAUDE.md

Update `CLAUDE.md` with a banner noting that the Z-score mean-reversion
signal family has been falsified on NQ/MNQ across the bar-timeframe axis
(1min, 15min, 1h) under the $5/contract RT friction model. Current
production is in **halted** state — V3 and V4 were revoked 2026-04-21, and
no current candidate replaces them.

### 5.2 — Lesson entry for lessons.md

Document the family-level falsification process:

> **Family-level falsification requires a spec, not a single experiment.**
> Probe 1 was the first time the project articulated a *bar-timeframe axis*
> with a pre-registered decision rule. Expanding to 15m and 1h required
> ~4 hours of remote compute (inflated by earlier bugs), and the mechanical
> rule gave a binding answer that a single-experiment readout could not.
> Do this before the next family-swap probe.

### 5.3 — Fresh LLM Council (separate task)

Spawn a council on the next fork:

- **Option X**: Signal-family swap (ORB / VWAP revert / breakout) on NQ/MNQ.
- **Option Y**: Session-structure probe (RTH-only, lunch-hour excluded,
  overnight-only) while staying on 1min + Z-score.
- **Option Z**: Project sunset (no tradeable edge found; stop investing
  in this strategy family at all).

Scope lock stands (`memory/feedback_nq_mnq_scope_only.md`): no
cross-instrument or cross-asset probes admitted on any branch.

---

## 6. Data Artifacts (preserved for audit)

- `data/ml/originals/ml_dataset_v11_15m.parquet` (64.9 MB, 847,570 rows,
  2,308 combos, 55 cols)
- `data/ml/originals/ml_dataset_v11_15m_manifest.json` (2.9 MB)
- `data/ml/originals/ml_dataset_v11_1h.parquet` (11.5 MB, 142,404 rows,
  1,078 combos, 55 cols)
- `data/ml/originals/ml_dataset_v11_1h_manifest.json` (1.4 MB)
- `tasks/probe1_15m_gross_sharpe.csv` (1,306 rows, per-combo gated Sharpe)
- `tasks/probe1_1h_gross_sharpe.csv`  (515 rows, per-combo gated Sharpe)

---

## 7. Signature

- **Verdict recorded**: 2026-04-21 UTC
- **Rule source**: `tasks/probe1_preregistration.md` §3 + §7
- **Readout script**: `tasks/_probe1_gross_ceiling.py`
- **Branch**: **A** (sunset)
- **Next council authority**: `tasks/council/council-*-probe1-branch-a-fork.{html,md}`
  (to be created in the next step)
