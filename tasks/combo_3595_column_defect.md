# `gross_pnl_dollars` column-consistency defect — combo 3595 (2026-04-23 UTC)

**Severity**: CRITICAL. Affects every project artifact that computes Sharpe or aggregate PnL from `gross_pnl_dollars` across trades within a combo.

**Surfaced by**: stats-ml-logic-reviewer audit of the Probe 1 1m ET finding, bus artifact `tasks/_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md` §Data Audit.

**Independently verified**: main-agent Bash session, 2026-04-23 UTC, against `data/ml/probe1_audit/ml_dataset_v11_1m_et.parquet` (combo 3595, 33,195 trades).

---

## The defect

For combo 3595 in the ET 1m parquet (single-generation run from commit `c06b2e0`):

| Measurement | Value | Implied contract count |
|---|---|---:|
| `stop_distance_pts` | constant 4.625 pt | — |
| `friction_dollars` | constant $1,620 (every trade) | **324** (= 1620 / $5 RT) |
| `r_multiple` back-calc via `net_pnl / (rmul × stop × $/pt)` | constant 270 on every trade | **270** |
| `gross_pnl_dollars` back-calc via reviewer formula `gpd / (rmul × stop × $/pt + $5)` | varies per trade, range [−270, +∞), median 296 | **inconsistent** |

**Expected per `scripts/param_sweep.py:1030-1038`**: all three back-calcs should return the same integer contract count, because the engine computes `contracts` once per trade and uses the same value for `gross_pnl_dollars`, `friction_dollars`, `net_pnl_dollars`, and `r_multiple` within that trade:

```python
# param_sweep.py:1030
contracts = (int(fixed_equity * cfg.RISK_PCT
                 // (stop_distance_pts * cfg.MNQ_DOLLARS_PER_POINT))
             if stop_distance_pts > 0 else 0)
gross_pnl    = (exit_price - entry_price) * side * contracts * cfg.MNQ_DOLLARS_PER_POINT
friction     = contracts * float(getattr(cfg, "COST_PER_CONTRACT_RT", 0.0))
net_pnl      = gross_pnl - friction
risk_dollars = stop_distance_pts * contracts * cfg.MNQ_DOLLARS_PER_POINT
r_multiple   = net_pnl / risk_dollars if risk_dollars > 0 else 0.0
```

On a single trade, all four outputs use the same `contracts`. The three back-calcs should therefore all equal that number. On combo 3595 they don't.

The 324 vs 270 discrepancy is a factor-of-**1.2** — exactly the ratio `0.06 / 0.05`, suggesting `friction_dollars` was computed with `RISK_PCT = 0.06` while `gross_pnl_dollars` and `r_multiple` were computed with `RISK_PCT = 0.05`. Alternative hypotheses: different `MNQ_DOLLARS_PER_POINT`, different `fixed_equity`, different `COST_PER_CONTRACT_RT`, or a mid-loop reassignment.

The `gross_pnl_dollars`-specific varying back-calc (−270 to +∞, median 296) is additionally suspicious — it suggests `gross_pnl_dollars` may have been computed trade-by-trade with a floating-point `contracts` rather than a single integer, or with some trade-varying scaling.

---

## Scope of affected artifacts

Every file that reads `gross_pnl_dollars` from an `ml_dataset_v*.parquet` and computes Sharpe or aggregate dollar-PnL across trades within a combo inherits this defect. Non-exhaustive list (grep-derived):

**Tracked readout scripts**:
- `tasks/_probe1_gross_ceiling.py` — the signed Probe 1 readout script, commit `d0ee506`. Its N_1.3(15m) = 9 and N_1.3(1h) = 4 numbers in the signed verdict inherit the defect.
- `tasks/_probe1_stratified_recount.py` — 15m/1h TZ audit at `db5e5f1`. Retracted above.
- `tasks/_probe1_stratified_recount_1m.py` — 1m TZ audit at `161485d`. Retracted above.
- Various `scripts/evaluation/*.py`, `scripts/backtests/*.py` that compute portfolio PnL from `gross_pnl_dollars`.

**Per-probe artifacts**:
- **Probe 2 combo-865 1h PASS** (Sharpe 2.895, +$124k/yr, 220 trades) — computed from `gross_pnl_dollars`. Needs re-verification under the v12 1-contract formula to confirm the aggregate Sharpe stands.
- **Probe 3 4-gate suite** — gates §4.1 (regime halves), §4.2 (param ±5% nbhd), §4.4 (session/exit ritual) all use per-trade PnL on `gross_pnl_dollars` or derived aggregates. Retraction earlier this session was for a different reason (TZ bug in session decomposition); the `gross_pnl_dollars` defect is a SECOND independent concern.
- **Probe 4 absolute gates + Welch-t** — computed from `net_pnl_dollars` which is `gross_pnl_dollars − friction_dollars`. Inherits the defect (the discrepancy propagates through the subtraction).
- **Scope D** — per-trade PnL decomposition on `net_pnl_dollars`. Inherits the defect.

**ML surrogate training sets** (v11/v12):
- **v12 `combo_features_v12.parquet`** — its `audit_full_gross_sharpe` column uses the v12 formula (`rmul × stop × $/pt`), not `gross_pnl_dollars`, so it is NOT directly affected by this defect. But other `audit_full_*` aggregates (trade counts, mean PnL) that derive from `gross_pnl_dollars` or `net_pnl_dollars` are.
- **v11 `combo_features`** — unknown; needs audit.

---

## What this means for prior "signed" verdicts

1. **Probe 1's signed N_1.3 table (1/9/4)** was computed with `gross_pnl_dollars` on the `_probe1_gross_ceiling.py` script. Under the v12 1-contract formula, the numbers become 0/2/0 — **no timeframe crosses the gate of 10 under any formula**. Branch A fires either way. So Probe 1's branch decision is robust to this defect, but the specific counts reported in the verdict are not trustworthy quantitatively.

2. **Probe 2 combo-865 1h Sharpe of 2.895** is computed on `gross_pnl_dollars`. The aggregate Sharpe value could be materially different under a 1-contract formula — needs recomputation. The PASS/FAIL outcome may or may not change.

3. **Probe 3 and Probe 4 verdicts were already retracted earlier this session** (TZ bug in session decomposition, commits `a38d3c5` and amendments). The `gross_pnl_dollars` defect is a separate reason to treat their point estimates as suspect, but the retraction was already in place for the session-labeling reason.

---

## Root-cause investigation plan (not executed yet)

To narrow down the mechanism:

1. **Load 10 trades from combo 3595** and print `(contracts_from_friction, contracts_from_rmul, contracts_from_gross, entry_price, exit_price, side, friction, gross, net)` side-by-side. Verify whether the inconsistency is within-trade (same trade's columns disagree) or across-trade (different trades used different `contracts`).
2. **Instrument `scripts/param_sweep.py:1030`** with an `assert` that `gross_pnl == (exit - entry) * side * contracts * $/pt` within 1e-6 and `friction == contracts * cost_rt`. Re-run a tiny sweep (1 combo, 10 trades). If the assert fires, the bug is in that block; if it doesn't, the bug is upstream (in how `contracts` or `fixed_equity` or `cfg.RISK_PCT` is computed or passed).
3. **Check `cfg.RISK_PCT` and `cfg.MNQ_DOLLARS_PER_POINT` at combo 3595's generation time**. If `RISK_PCT` changes across combos within the same sweep, the engine has a config-mutation bug; if constant, the issue is elsewhere.
4. **Diff the engine PnL code against an older known-good revision** (pre-v11) to see when the defect was introduced. If it predates v11, every project artifact is affected.
5. **Audit whether `fixed_equity` is static ($50k) or compounding** — per `CLAUDE.md`, the current project sizing is `fixed_dollars_500` ($500 risk per trade), but the actual engine variable is `fixed_equity × RISK_PCT`. Mismatch between these could produce the 324 vs 270 ratio observed.

This investigation is not started in this session. Priority: HIGH — affects every probe's numerical claims.

---

## References

- `tasks/_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md` §Data Audit "Internal column-consistency defect"
- `tasks/probe1_1m_et_audit_finding.md` Amendment 1 (cites this doc)
- `scripts/param_sweep.py:1030-1038` (the engine PnL block)
- `scripts/analysis/build_combo_features_ml1_v12.py:229` (the 1-contract formula that escapes this defect)
- `data/ml/probe1_audit/ml_dataset_v11_1m_et.parquet` (source of the combo 3595 observation)
