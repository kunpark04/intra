"""Extract top-K combos by raw train-partition net Sharpe (no ML).

After the 2026-04-19 ablation proved the v12 UCB ranker adds no value
over random sampling on OOS Sharpe (see
`evaluation/ablation/ablation_results.json`), the empirical winner was
the trivial `sort_values("audit_full_net_sharpe").head(50)` on the
same eligible pool. This script emits that selection as a
composed_strategy_runner-compatible JSON file at
`evaluation/top_strategies_v12_raw_sharpe_top50.json`.

Schema mirrors `top_strategies_v12_top50.json` so downstream notebooks
consume it unchanged via `load_setup(top_strategies_path=...)`. Predicted
quantile fields are omitted (no booster).

Usage:
    python scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py
    python scripts/analysis/extract_top_combos_by_raw_sharpe_v12.py --top-k 100
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FEATURES_PARQUET = REPO / "data" / "ml" / "ml1_results_v12" / "combo_features_v12.parquet"
OUTPUT_JSON = REPO / "evaluation" / "top_strategies_v12_raw_sharpe_top50.json"

RANKING_COL = "audit_full_net_sharpe"

# PARAM_COLS mirrors extract_top_combos_v12.py PARAM_COLS so downstream
# composed_strategy_runner reads the same parameter block shape.
PARAM_COLS = [
    "z_band_k", "z_window", "volume_zscore_window",
    "ema_fast", "ema_slow",
    "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
    "min_rr", "exit_on_opposite_signal", "use_breakeven_stop",
    "max_hold_bars", "zscore_confirmation",
    "z_input", "z_anchor", "z_denom", "z_type",
    "z_window_2", "z_window_2_weight",
    "volume_entry_threshold",
    "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
    "session_filter_mode", "tod_exit_hour",
    "stop_fixed_pts_resolved",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    ap.add_argument("--output", type=Path, default=OUTPUT_JSON)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--min-trades", type=int, default=500)
    return ap.parse_args()


def _entry_payload(row: pd.Series) -> dict:
    params = {c: row[c] for c in PARAM_COLS if c in row.index}
    clean = {}
    for k, v in params.items():
        if pd.isna(v):
            clean[k] = None
        elif hasattr(v, "item"):
            clean[k] = v.item()
        else:
            clean[k] = v
    return {
        "global_combo_id": row["global_combo_id"],
        "combo_id": int(row["combo_id"]),
        "parameters": clean,
        "stop_fixed_pts_resolved": float(row["stop_fixed_pts_resolved"]),
        "realised": {
            "audit_full_net_sharpe": float(row["audit_full_net_sharpe"]),
            "audit_full_gross_sharpe": float(row["audit_full_gross_sharpe"]),
            "audit_n_trades": int(row["audit_n_trades"]),
            "target_robust_sharpe": float(row["target_robust_sharpe"]),
            "target_median_wf_sharpe": float(row["target_median_wf_sharpe"]),
            "target_std_wf_sharpe": float(row["target_std_wf_sharpe"]),
        },
    }


def main() -> None:
    args = _parse_args()
    print(f"[raw-sharpe-extract] loading {args.features.relative_to(REPO)}",
          flush=True)
    df = pd.read_parquet(args.features)
    print(f"[raw-sharpe-extract] {len(df)} combos total", flush=True)

    eligible = df[df["audit_n_trades"] >= args.min_trades].copy()
    print(f"[raw-sharpe-extract] n_trades >= {args.min_trades}: "
          f"{len(eligible)} eligible, "
          f"{len(df) - len(eligible)} rejected", flush=True)

    top = (eligible.sort_values(RANKING_COL, ascending=False)
           .head(args.top_k)
           .reset_index(drop=True))
    print(f"[raw-sharpe-extract] top-{args.top_k} by {RANKING_COL}: "
          f"range [{top[RANKING_COL].min():.3f} .. {top[RANKING_COL].max():.3f}]",
          flush=True)
    print(f"[raw-sharpe-extract] first 5 ids: "
          f"{list(top['global_combo_id'].head(5))}", flush=True)

    payload = {
        "source_features": str(args.features.relative_to(REPO)),
        "ranking_metric": RANKING_COL,
        "min_trades": args.min_trades,
        "top_k": args.top_k,
        "pool_sizes": {
            "total": int(len(df)),
            "eligible": int(len(eligible)),
        },
        "provenance": (
            "Selected as the winning baseline in the 2026-04-19 ablation "
            "(evaluation/ablation/ablation_results.json, Pool B). Ranks "
            "combos by the train-partition 1-contract net Sharpe already "
            "precomputed in combo_features_v12.parquet — no ML model."
        ),
        "top": [_entry_payload(r) for _, r in top.iterrows()],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[raw-sharpe-extract] wrote {args.output.relative_to(REPO)}",
          flush=True)


if __name__ == "__main__":
    main()
