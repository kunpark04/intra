# `gross_pnl_dollars` column-consistency "defect" — RETRACTED (2026-04-24 UTC)

> 🛑 **RETRACTED.** This document claimed a CRITICAL column-consistency defect on combo 3595 where `friction_dollars`, `r_multiple`, and `gross_pnl_dollars` implied different contract counts (324, 270, and "varying"). **There is no defect.** The claim was produced by a back-calc error that assumed `COST_PER_CONTRACT_RT = $5` universally, when in fact the engine applies a per-combo slippage bump.

## Root cause of the false positive

`scripts/param_sweep.py:895-901` computes effective `COST_PER_CONTRACT_RT` per combo:

```python
_fill_slip_bump = (
    int(combo.get("fill_slippage_ticks", 0))
    * 2
    * float(_FIXED["TICK_SIZE"])
    * float(_FIXED["MNQ_DOLLARS_PER_POINT"])
)
ns.COST_PER_CONTRACT_RT = float(_FIXED["COST_PER_CONTRACT_RT"]) + _fill_slip_bump
```

For MNQ (`TICK_SIZE=0.25`, `MNQ_DOLLARS_PER_POINT=2`), each unit of `fill_slippage_ticks` adds `2 × 0.25 × 2 = $1` per round-trip. So:
- `fill_slippage_ticks=0` → cost_rt = $5 (base)
- `fill_slippage_ticks=1` → cost_rt = $6
- `fill_slippage_ticks=2` → cost_rt = $7

Combo 3595 has `fill_slippage_ticks=1`, so its effective cost_rt = $6, not $5.

With cost_rt=$6:
- `friction_dollars = contracts × $6 = 1620` → contracts = **270** (not 324)
- This matches the r_multiple-implied contracts of 270 exactly

**All four relationships hold with max error 0.000000** on combo 3595's actual trades:
- `net_pnl_dollars == gross_pnl_dollars − friction_dollars`
- `gross_pnl_dollars == (r_multiple × stop_distance_pts × 2 + cost_rt) × contracts`
- `friction_dollars == contracts × cost_rt`
- `r_multiple == net_pnl_dollars / (stop_distance_pts × contracts × 2)`

## Provenance of the error

1. stats-ml-logic-reviewer (bus artifact `tasks/_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md`) reported the "defect" in its §Data Audit, using `cost_rt=$5` in its back-calc.
2. main-agent independently reproduced the 324-vs-270 discrepancy in a Bash verification using the same $5 assumption.
3. Neither check noticed that `param_sweep.py:895-901` adds a per-combo slippage bump.
4. I committed this investigation note as `b545d93` and wrote about a "CRITICAL column-consistency defect" in Amendment 1 of the 1m finding doc (also `b545d93`).

The hallucinated defect met the `stats-ml-logic-reviewer`'s own stated "verify before claiming" principle in pattern (back-calc from observed columns) but not in substance (failed to verify cost_rt was constant at $5 across combos). Per the agent's operating principle: *"If you cannot produce the concrete evidence on demand, do not raise the flag."* The concrete evidence (back-calc) looked compelling but didn't account for a per-combo engine-side shift — a classic hidden-assumption failure mode.

## Correction to prior commits

- Commit `b545d93` claimed the column defect was real and invalidated every Sharpe computed from `gross_pnl_dollars` (Probe 2 combo-865 Sharpe 2.895, all Probe 3/4 gates, Scope D, ML surrogate features). **That claim is withdrawn.** The column is internally consistent.
- The Amendment 2 to `tasks/probe1_1m_et_audit_finding.md` (this commit cycle) addresses the cascading misstatements.

## What the false-positive cost

No lasting data damage — the column is fine, prior probe numerical claims don't inherit a column defect. But:

1. Amendment 1 used the false defect to justify blanket retraction of the 15m/1h findings. That retraction was premature and is itself partially retracted in Amendment 2.
2. The "investigation plan" at the bottom of this document's original version (instrumenting `param_sweep.py:1030` for assertion, diffing engine PnL code against older revisions) was scoped on a false premise and should not be executed.

## Lesson

When back-calculating a quantity from observed columns, enumerate every constant the back-calc depends on and verify each is literally constant across the sample. `COST_PER_CONTRACT_RT` looked constant in the config but is modified per combo at `param_sweep.py:901`. This kind of hidden per-row arithmetic is the standard failure mode for column-consistency checks. A clean back-calc would have computed cost_rt per combo from `friction/contracts_from_rmul` directly and found it exactly 6.0 — which is what `fill_slippage_ticks=1 → +$1` predicts.

## References

- `scripts/param_sweep.py:895-901` (the slippage-bump formula)
- `scripts/param_sweep.py:1030-1038` (the PnL computation block — unchanged, always was internally consistent)
- `tasks/probe1_1m_et_audit_finding.md` Amendment 2 (the correction cycle this retraction is part of)
- `tasks/_agent_bus/probe1_1m_audit_2026-04-23/stats-ml-logic-reviewer.md` §Data Audit (the review that contained the error)
