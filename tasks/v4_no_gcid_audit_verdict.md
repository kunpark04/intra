# V4 Combo-Agnostic Ship-Blocker Audit — VERDICT: FAIL

**Date**: 2026-04-20 / 2026-04-21 UTC
**Result**: 🛑 **REVOKE Pool B + V4 ship decision**
**Authority**: LLM Council Option 1 verdict (2026-04-20 13:49 CDT)
**Plan**: tasks #7–#11 of `C:\Users\kunpa\.claude\plans\sleepy-roaming-kettle.md`

## Summary

The shipped Pool B + V4 configuration — which the 2026-04-19 s6_net MC
produced at **Sharpe p50 2.13, 95% CI (+0.56, +3.64), ruin 0.02%** — was
leaking via the `global_combo_id` LightGBM categorical feature. When V4
is refit without `global_combo_id` (`adaptive_rr_v4_no_gcid/`, commit
`930bace`) and re-audited on the identical Pool B top-50 with identical
MC kernel, every ship-decision metric collapses.

## Head-to-head (s6 net, identical Pool B, identical MC kernel)

| Metric | Shipped V4 (with `global_combo_id`) | V4 combo-agnostic (`v4_no_gcid`) | Delta |
|---|---|---|---|
| `n_trades` filtered | 641 | **1,446** | +126% retention |
| `win_rate` | 33.39% (CI 29.8–36.97) | **27.17%** (CI 24.9–29.53) | **−6.22 pp, CIs disjoint** |
| `sharpe_p50` | **+2.1331** | **−0.4194** | **−2.55 Sharpe** |
| `sharpe_ci_95` | (+0.56, +3.64) | **(−2.10, +1.19)** | **crosses zero** |
| `sharpe_pos_prob` | 99.6% | **30.2%** | strategy Sharpe-negative in 70% of 10k paths |
| `dd_worst_pct` | 55.1% | **207.69%** | 3.8× worse |
| `risk_of_ruin_prob` | 0.02% | **56.37%** | 2,800× worse — majority-ruin regime |
| `trades_per_year` | 438.7 | 989.7 | +126% |

Source notebooks:
- Baseline → `evaluation/v12_topk_top50_raw_sharpe_net_v4/s6_mc_combined_ml2_net.ipynb` (commit `734d3a3`)
- Combo-agnostic → `evaluation/v12_topk_top50_raw_sharpe_net_v4_no_gcid/s6_mc_combined_ml2_net.ipynb` (this commit)

## Interpretation

### Leak direction was contraction, not expansion

Combo-agnostic V4 admits **2.25× more trades** from the same 43,846-trade
unfiltered Pool B stream (1,446 vs 641 kept). This means the original V4
was using `global_combo_id` to *exclude* combos it had seen lose in-sample —
pure per-combo overfit. Strip the ID column and the model has no basis to
reject those combos; it accepts many more trades, most of which are net
losers. Win rate drops 6.22 percentage points.

### 70%-negative Sharpe posterior is the cleanest falsifier

`sharpe_pos_prob = 0.3019` means across 10,000 IID bootstrap paths of the
filtered trade stream, only 30% produce a positive Sharpe — 70% are
negative. The shipped V4 baseline sits at 99.6%. This is a structural
collapse that cannot be rescued by alternative sizing or cost assumptions.

### Ruin probability 56% is catastrophic regime

More than half of 10k bootstrap paths exceed 50% drawdown. `dd_worst_pct
= 207.69%` indicates the cumulative $-loss in the worst path is 2.08×
starting equity — mathematically possible under `fixed_dollars_500` sizing
because losses compound without the equity-floor guard of fractional
sizing, but in practical terms the account is long-since broken.

### Retroactive implication for ML#1 top-K concentration

The 2026-04-20 full-pool ablation
(`evaluation/v12_full_pool_net_v4_2k/s6_mc_combined_ml2_net.ipynb`) showed
Sharpe 0.00 / ruin 43.45% at 24,542 trades and led to the memory
`project_ml1_topk_concentration` claiming ML#1 top-K truncation is
load-bearing. That interpretation is now incomplete: at **top-50** (the
concentrated pool), with combo-ID stripped, we observe ruin **56%** — worse
than the full-pool ablation. Top-K concentration was interacting with
V4's combo-ID memorization, not carrying the ship on its own. The
memory must be updated.

## Action items

1. ✅ **Revoke the Pool B + V4 ship decision** (this commit — `tasks/ship_decision.md` banner flipped).
2. ❓ **V3 audit deferred but still open**. `adaptive_rr_v3` contains the
   same combo-ID memorization pattern (lines 64, 67, 69 of
   `scripts/models/adaptive_rr_model_v3.py`). V3 is the declared
   production stack per CLAUDE.md. A combo-agnostic V3 refit + ship
   audit is now mandatory before any further live-capital claims. Not
   addressed in this commit.
3. 🛑 **Pool B top-50 is retired as a ship candidate** for the current
   feature set. Any re-ship attempt must first establish generalization
   (clean refit, leave-K-combos-out CV) *before* re-running MC.
4. **Paper-trading halted** per the PROVISIONAL banner's escalation rule.
5. **Update memory**:
   - `project_council_plan_stage.md` → task #9 DONE, verdict FAIL
   - `project_ml1_topk_concentration` → qualify with "top-K concentration
     interacts with V4 combo-ID leak; clean combo-agnostic behavior at
     top-50 still produces ruin 56%"
   - New memory: `project_v4_combo_id_leak_confirmed` documenting the
     falsification delta for future reference

## Reproduction

```bash
# Verify combo-agnostic artifacts on sweep-runner-1
python tasks/_check_v4_no_gcid_artifact.py

# Run the 12-notebook audit remotely (produces outputs under
# /root/intra/evaluation/v12_topk_top50_raw_sharpe_{,net_}v4_no_gcid/)
python scripts/runners/run_eval_notebooks_remote.py v12_top50_raw_sharpe_v4_no_gcid

# Stream progress (emits per-notebook start/done + terminal states)
python tasks/_monitor_v4_no_gcid_eval.py

# Pull executed notebooks after ALL DONE fires
python tasks/_pull_v4_no_gcid_eval.py
```

Total wall time on sweep-runner-1 (cold model cache, warm results_raw
cache from prior shipped-V4 run): ~19 minutes for the 12 notebooks under
9G cgroup + 280% CPUQuota.
