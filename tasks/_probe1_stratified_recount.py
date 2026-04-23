"""Probe 1 stratified recount — audit whether the 15m/1h N_1.3 counts are
TZ-sensitive.

Motivation: Probe 3 COUNCIL_RECONVENE chairman (2026-04-23 UTC) asked whether
Probe 1's family-level falsification survives the TZ fix. Engine-side
`bar_hour` (`scripts/param_sweep.py:1567`) is built from raw CSV hours = CT,
so any combo with `session_filter_mode != 0` had its filter window run on CT
hours while the preregistration's comments ("core US 9-15h") suggest ET
intent. If the passing combos (N_1.3=9 on 15m; N_1.3=4 on 1h) are dominated
by `session_filter_mode=0` combos, the bug is irrelevant — those combos ran
on all hours with no filter applied and the TZ bug cannot affect them.

This script is pure pandas over existing parquets — zero engine compute.
Runs anywhere; per the 2026-04-23 user directive, it runs remotely.

Outputs:
  data/ml/probe1_audit/stratified_recount.json

Formula (mirrors tasks/_probe1_gross_ceiling.py + build_combo_features_ml1_v12.py):
  sharpe = mean(gross_pnl_dollars) / std(gross_pnl_dollars, ddof=1)
           * sqrt(n_trades / YEARS_SPAN_TRAIN)
  YEARS_SPAN_TRAIN = 5.8056
  MIN_TRADES_GATE = 50
  SHARPE_THRESHOLD = 1.3
  PASS := (gross_sharpe >= 1.3) AND (n_trades >= 50)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "ml" / "probe1_audit"
OUT_PATH = OUT_DIR / "stratified_recount.json"

YEARS_SPAN_TRAIN = 5.8056
MIN_TRADES_GATE  = 50
SHARPE_THRESHOLD = 1.3

PARQUETS = {
    "15m": REPO / "data" / "ml" / "originals" / "ml_dataset_v11_15m.parquet",
    "1h":  REPO / "data" / "ml" / "originals" / "ml_dataset_v11_1h.parquet",
}


def compute_combo_sharpes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-combo gross-Sharpe table with session_filter_mode attached.

    Returns one row per combo_id with gross_sharpe, n_trades, and the
    combo's session_filter_mode (which is constant within each combo_id
    since it's a parameter, not a per-trade feature).
    """
    rows = []
    for combo_id, g in df.groupby("combo_id", sort=False):
        gross = g["gross_pnl_dollars"].to_numpy()
        n = len(gross)
        if n < 2:
            sharpe = 0.0
            std = 0.0
            mean = 0.0
        else:
            std = float(np.std(gross, ddof=1))
            mean = float(np.mean(gross))
            if std <= 0:
                sharpe = 0.0
            else:
                sharpe = mean / std * float(np.sqrt(n / YEARS_SPAN_TRAIN))

        rows.append({
            "combo_id":               int(combo_id),
            "n_trades":               int(n),
            "gross_sharpe":           float(sharpe),
            "mean_gross":             float(mean),
            "std_gross":              float(std),
            "session_filter_mode":    int(g["session_filter_mode"].iloc[0]),
            "entry_timing_offset":    int(g["entry_timing_offset"].iloc[0]),
            "fill_slippage_ticks":    int(g["fill_slippage_ticks"].iloc[0]),
            "cooldown_after_exit_bars": int(g["cooldown_after_exit_bars"].iloc[0]),
            "tod_exit_hour":          int(g["tod_exit_hour"].iloc[0])
                                       if "tod_exit_hour" in g.columns else 0,
        })
    return pd.DataFrame(rows).sort_values("gross_sharpe", ascending=False).reset_index(drop=True)


def stratified_counts(combo_df: pd.DataFrame) -> dict:
    """Break the N_1.3 count down by session_filter_mode."""
    gated = combo_df[combo_df["n_trades"] >= MIN_TRADES_GATE].copy()
    passing = gated[gated["gross_sharpe"] >= SHARPE_THRESHOLD].copy()

    # Overall
    result = {
        "n_combos_with_trades":       int(len(combo_df)),
        "n_combos_gated":             int(len(gated)),
        "n_pass_overall":             int(len(passing)),
        "gate_sharpe":                SHARPE_THRESHOLD,
        "gate_min_trades":            MIN_TRADES_GATE,
        "years_span_train":           YEARS_SPAN_TRAIN,
    }

    # Breakdown by session_filter_mode across the full "combos with trades" pool.
    mode_breakdown = {}
    for mode in sorted(combo_df["session_filter_mode"].unique()):
        mode_int = int(mode)
        pool = combo_df[combo_df["session_filter_mode"] == mode]
        pool_gated = pool[pool["n_trades"] >= MIN_TRADES_GATE]
        pool_pass = pool_gated[pool_gated["gross_sharpe"] >= SHARPE_THRESHOLD]
        mode_breakdown[f"mode_{mode_int}"] = {
            "n_combos_with_trades": int(len(pool)),
            "n_combos_gated":       int(len(pool_gated)),
            "n_pass":               int(len(pool_pass)),
            "pass_combo_ids":       sorted(int(x) for x in pool_pass["combo_id"].tolist()),
            "pass_sharpes":         [round(float(x), 4) for x in pool_pass["gross_sharpe"].tolist()],
        }
    result["by_session_filter_mode"] = mode_breakdown

    # Also: tod_exit_hour != 0 breakdown (another TZ-sensitive axis)
    tod_pass = passing[passing["tod_exit_hour"] != 0]
    result["pass_with_tod_exit_nonzero"] = {
        "count":       int(len(tod_pass)),
        "combo_ids":   sorted(int(x) for x in tod_pass["combo_id"].tolist()),
    }

    # Full passing-combo detail (for caller to inspect)
    result["passing_combos"] = [
        {
            "combo_id":            int(r["combo_id"]),
            "gross_sharpe":        round(float(r["gross_sharpe"]), 4),
            "n_trades":            int(r["n_trades"]),
            "session_filter_mode": int(r["session_filter_mode"]),
            "tod_exit_hour":       int(r["tod_exit_hour"]),
            "entry_timing_offset": int(r["entry_timing_offset"]),
            "fill_slippage_ticks": int(r["fill_slippage_ticks"]),
            "cooldown_after_exit_bars": int(r["cooldown_after_exit_bars"]),
        }
        for _, r in passing.iterrows()
    ]

    # Near-miss bucket: combos just below 1.3 (could cross under TZ shift)
    near_miss = gated[(gated["gross_sharpe"] >= 1.1) & (gated["gross_sharpe"] < 1.3)]
    result["near_miss_1_1_to_1_3"] = {
        "count":      int(len(near_miss)),
        "combo_ids":  sorted(int(x) for x in near_miss["combo_id"].tolist()),
        "sharpes":    [round(float(x), 4) for x in near_miss["gross_sharpe"].tolist()],
    }
    # Breakdown of near-miss by mode (same TZ question — would mode!=0 near-miss combos cross under ET?)
    nm_by_mode = {}
    for mode in sorted(combo_df["session_filter_mode"].unique()):
        mode_int = int(mode)
        nm_mode = near_miss[near_miss["session_filter_mode"] == mode_int]
        nm_by_mode[f"mode_{mode_int}"] = int(len(nm_mode))
    result["near_miss_by_session_filter_mode"] = nm_by_mode

    return result


def main() -> None:
    out = {
        "task":             "probe1_stratified_recount",
        "authority":        "tasks/council-report-2026-04-23-probe3-reconvene.html (chairman's one-thing-to-do-first)",
        "provenance":       "tasks/tz_bug_provenance_log_2026-04-23.md",
        "methodology":      "pure pandas over existing sweep parquets — zero engine compute",
        "by_timeframe":     {},
    }

    for tf, path in PARQUETS.items():
        print(f"\n{'='*68}\n  {tf.upper()}  ({path.name})\n{'='*68}", flush=True)
        if not path.exists():
            raise FileNotFoundError(f"Sweep parquet missing: {path}")
        df = pd.read_parquet(path)
        print(f"rows={len(df):,}  unique_combos={df['combo_id'].nunique():,}", flush=True)

        combo = compute_combo_sharpes(df)
        print(f"combos analysed: {len(combo):,}", flush=True)

        result = stratified_counts(combo)
        out["by_timeframe"][tf] = result

        print(f"\nOverall: {result['n_pass_overall']} / {result['n_combos_gated']} gated combos PASS "
              f"(Sharpe>={SHARPE_THRESHOLD} & n>={MIN_TRADES_GATE})", flush=True)
        print(f"\nStratified by session_filter_mode:", flush=True)
        for mode_key, mb in result["by_session_filter_mode"].items():
            print(f"  {mode_key}: gated={mb['n_combos_gated']:>4}  pass={mb['n_pass']:>3}  "
                  f"combo_ids={mb['pass_combo_ids']}", flush=True)
        print(f"\nPass combos with tod_exit_hour != 0 (another TZ-sensitive axis):", flush=True)
        print(f"  count={result['pass_with_tod_exit_nonzero']['count']}  "
              f"ids={result['pass_with_tod_exit_nonzero']['combo_ids']}", flush=True)
        print(f"\nNear-miss [1.1, 1.3) combos (could cross under TZ shift):", flush=True)
        print(f"  total={result['near_miss_1_1_to_1_3']['count']}  "
              f"by_mode={result['near_miss_by_session_filter_mode']}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[recount] wrote {OUT_PATH}", flush=True)

    # Top-line interpretation cue for the reader
    print("\n" + "=" * 68, flush=True)
    print("INTERPRETATION CUE", flush=True)
    print("=" * 68, flush=True)
    for tf, result in out["by_timeframe"].items():
        mode0_pass = result["by_session_filter_mode"]["mode_0"]["n_pass"]
        total_pass = result["n_pass_overall"]
        tz_immune_frac = mode0_pass / total_pass if total_pass > 0 else float("nan")
        print(f"  {tf}: {mode0_pass}/{total_pass} passing combos are mode=0 (TZ-immune) "
              f"= {tz_immune_frac:.1%}", flush=True)
        if total_pass > 0 and mode0_pass == total_pass:
            print(f"        -> ALL passing {tf} combos are TZ-immune. "
                  f"Probe 1 {tf} verdict is TZ-robust by construction.", flush=True)
        elif total_pass > 0 and mode0_pass > 0:
            print(f"        -> MIXED. TZ re-sweep is needed to know if mode!=0 combos "
                  f"would pass under ET semantics.", flush=True)
        elif total_pass == 0:
            print(f"        -> no passing combos (should not happen if verdict reports N>=1).", flush=True)


if __name__ == "__main__":
    main()
