"""Extract the top-K combos in each frequency bucket for final evaluation.

Loads `combo_features.parquet` (ML#1 training frame — every sweep combo with
realised metrics + hyperparameters) and the ML#1 `composite_score` LightGBM
model, predicts composite on every combo, splits combos into high-frequency
(`n_trades >= --high-freq-min`) and low-frequency (`n_trades <= --low-freq-max`)
buckets, then picks the top-K per bucket by ML#1-predicted composite.

Output: `evaluation/top_strategies.json` — one entry per combo containing
`global_combo_id`, ML prediction, realised metrics, and the complete
hyperparameter set needed to reconstruct the strategy.

Usage:
    python scripts/analysis/extract_top_combos_by_freq.py
    python scripts/analysis/extract_top_combos_by_freq.py --top-k 5 \
        --high-freq-min 1000 --low-freq-max 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. Run: pip install lightgbm",
          file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent.parent
ML1_DIR = REPO / "data" / "ml" / "ml1_results_v2filtered"
FEATURES_PARQUET = ML1_DIR / "combo_features.parquet"
COMPOSITE_MODEL = ML1_DIR / "models" / "composite_score.txt"
DEFAULT_OUT = REPO / "evaluation" / "top_strategies.json"

# Hyperparameter columns that fully specify a strategy (mirrors PARAM_COLS
# in scripts/models/ml1_surrogate.py + sweep metadata needed for replay).
PARAM_COLS = [
    "sweep_version",
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
]

REALISED_COLS = [
    "n_trades", "win_rate", "total_return_pct", "sharpe_ratio",
    "max_drawdown_pct", "profit_factor", "avg_r_multiple",
    "median_r_multiple", "composite_score",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the top-combo extractor.

    Returns:
        Namespace with `top_k`, `high_freq_min`, `low_freq_max`, `output`.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of combos to keep per frequency bucket.")
    ap.add_argument("--high-freq-min", type=int, default=1000,
                    help="n_trades threshold (inclusive) for high-freq bucket.")
    ap.add_argument("--low-freq-max", type=int, default=300,
                    help="n_trades threshold (inclusive) for low-freq bucket.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help="Output JSON path.")
    ap.add_argument("--features", type=Path, default=FEATURES_PARQUET,
                    help="combo_features.parquet path.")
    ap.add_argument("--model", type=Path, default=COMPOSITE_MODEL,
                    help="LightGBM composite-score model path.")
    return ap.parse_args()


def _to_jsonable(v):
    """Convert numpy/pandas scalars to JSON-serialisable Python types."""
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _row_to_entry(row: pd.Series) -> dict:
    """Format one combo row as the JSON entry written to `top_strategies.json`."""
    params = {c: _to_jsonable(row[c]) for c in PARAM_COLS if c in row}
    realised = {c: _to_jsonable(row[c]) for c in REALISED_COLS if c in row}
    return {
        "global_combo_id": str(row["global_combo_id"]),
        "predicted_composite": float(row["predicted_composite"]),
        "realised": realised,
        "parameters": params,
    }


def main() -> None:
    """Rank all combos by ML#1 composite, split by frequency, emit top-K per bucket.

    Loads the ML#1 training frame and saved composite LightGBM booster,
    predicts composite on every combo, filters into high- and low-frequency
    buckets, ranks each bucket, and writes the resulting JSON to
    `evaluation/top_strategies.json`.
    """
    args = parse_args()

    if not args.features.exists():
        print(f"ERROR: features parquet not found: {args.features}",
              file=sys.stderr)
        sys.exit(1)
    if not args.model.exists():
        print(f"ERROR: composite model not found: {args.model}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[extract] loading {args.features.relative_to(REPO)}")
    df = pd.read_parquet(args.features)
    print(f"[extract] {len(df):,} combos loaded, cols={len(df.columns)}")

    print(f"[extract] loading booster {args.model.relative_to(REPO)}")
    booster = lgb.Booster(model_file=str(args.model))
    feat_cols = booster.feature_name()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"ERROR: model features missing from parquet: {missing[:5]}...",
              file=sys.stderr)
        sys.exit(1)

    X = df[feat_cols].copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category")
    df["predicted_composite"] = booster.predict(X)

    hi_mask = df["n_trades"] >= args.high_freq_min
    lo_mask = df["n_trades"] <= args.low_freq_max
    hi = (df[hi_mask]
          .sort_values("predicted_composite", ascending=False)
          .head(args.top_k))
    lo = (df[lo_mask]
          .sort_values("predicted_composite", ascending=False)
          .head(args.top_k))

    print(f"[extract] high-freq pool: {int(hi_mask.sum()):,}  "
          f"low-freq pool: {int(lo_mask.sum()):,}")
    print(f"[extract] top-{args.top_k} high: "
          f"{list(hi['global_combo_id'])}")
    print(f"[extract] top-{args.top_k} low:  "
          f"{list(lo['global_combo_id'])}")

    out = {
        "source_features": str(args.features.relative_to(REPO)).replace("\\", "/"),
        "source_model": str(args.model.relative_to(REPO)).replace("\\", "/"),
        "freq_thresholds": {
            "high_freq_min_trades": args.high_freq_min,
            "low_freq_max_trades": args.low_freq_max,
        },
        "top_k": args.top_k,
        "pool_sizes": {
            "total_combos": int(len(df)),
            "high_freq_pool": int(hi_mask.sum()),
            "low_freq_pool": int(lo_mask.sum()),
        },
        "high_freq": [_row_to_entry(r) for _, r in hi.iterrows()],
        "low_freq": [_row_to_entry(r) for _, r in lo.iterrows()],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[extract] wrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
