"""Extract the top-K combos overall by ML#1-predicted composite score.

Loads `combo_features.parquet` (ML#1 training frame) and the ML#1
composite-score LightGBM model, predicts composite for every combo,
applies a `--min-trades` floor to drop statistically unreliable combos,
then picks the top-K by predicted composite.

Output: `evaluation/top_strategies.json` — one entry per combo with
`global_combo_id`, ML prediction, realised metrics, and the complete
hyperparameter set (including `stop_fixed_pts_resolved` from the sweep
manifest) needed to reconstruct the strategy.

Frequency bucketing was dropped after an audit
(`scripts/analysis/_freq_filter_audit.py`) showed that predicted composite
is uncorrelated with `n_trades` (r = -0.002) and the overall top-20
combos all live in the middle band — bucketing was actively excluding the
best combos. A `--min-trades` floor remains the only principled filter:
it rejects combos where realised metrics (and therefore the trained
labels) are noise from too-few trades.

Usage:
    python scripts/analysis/extract_top_combos.py
    python scripts/analysis/extract_top_combos.py --top-k 10 --min-trades 100
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
DEFAULT_MANIFEST_DIR = REPO / "data" / "ml" / "originals"

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
    """Parse CLI args for the top-combo extractor."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of combos to keep overall.")
    ap.add_argument("--min-trades", type=int, default=100,
                    help="Reject combos with fewer than this many trades "
                         "(realised metrics are noise below ~100).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help="Output JSON path.")
    ap.add_argument("--features", type=Path, default=FEATURES_PARQUET,
                    help="combo_features.parquet path.")
    ap.add_argument("--model", type=Path, default=COMPOSITE_MODEL,
                    help="LightGBM composite-score model path.")
    ap.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR,
                    help="Directory of sweep manifests with stop_fixed_pts_resolved.")
    return ap.parse_args()


def _load_resolved_stops(manifest_dir: Path) -> dict[str, float]:
    """Load `stop_fixed_pts_resolved` per `global_combo_id` from sweep manifests.

    The sweep stored a single resolved fixed-points value for every combo,
    even those with `stop_method` of `"atr"` or `"swing"` — pre-resolved
    from training-period medians. Replaying these combos on OOS bars must
    use this resolved value (not recompute ATR/swing on test data), or the
    strategy becomes something the ML never selected.
    """
    resolved: dict[str, float] = {}
    if not manifest_dir.exists():
        return resolved
    for v in range(2, 11):
        path = manifest_dir / f"ml_dataset_v{v}_manifest.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()):
            cid = entry.get("combo_id")
            val = entry.get("stop_fixed_pts_resolved")
            if cid is not None and val is not None:
                resolved[f"v{v}_{cid}"] = float(val)
    return resolved


def _to_jsonable(v):
    """Convert numpy/pandas scalars to JSON-serialisable Python types."""
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _row_to_entry(row: pd.Series, resolved_stops: dict[str, float]) -> dict:
    """Format one combo row as the JSON entry written to `top_strategies.json`."""
    params = {c: _to_jsonable(row[c]) for c in PARAM_COLS if c in row}
    realised = {c: _to_jsonable(row[c]) for c in REALISED_COLS if c in row}
    cid = str(row["global_combo_id"])
    if cid in resolved_stops:
        params["stop_fixed_pts_resolved"] = resolved_stops[cid]
    return {
        "global_combo_id": cid,
        "predicted_composite": float(row["predicted_composite"]),
        "realised": realised,
        "parameters": params,
    }


def main() -> None:
    """Rank all combos by ML#1 composite, apply min-trades floor, emit top-K."""
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

    resolved_stops = _load_resolved_stops(args.manifest_dir)
    print(f"[extract] loaded {len(resolved_stops):,} resolved-stop entries "
          f"from {args.manifest_dir.relative_to(REPO)}")

    eligible = df[df["n_trades"] >= args.min_trades]
    rejected = len(df) - len(eligible)
    print(f"[extract] min-trades floor (>= {args.min_trades}): "
          f"{len(eligible):,} eligible, {rejected:,} rejected")

    top = (eligible
           .sort_values("predicted_composite", ascending=False)
           .head(args.top_k))

    print(f"[extract] top-{args.top_k}: {list(top['global_combo_id'])}")

    out = {
        "source_features": str(args.features.relative_to(REPO)).replace("\\", "/"),
        "source_model": str(args.model.relative_to(REPO)).replace("\\", "/"),
        "min_trades": args.min_trades,
        "top_k": args.top_k,
        "pool_sizes": {
            "total_combos": int(len(df)),
            "eligible_combos": int(len(eligible)),
            "rejected_min_trades": int(rejected),
        },
        "top": [_row_to_entry(r, resolved_stops) for _, r in top.iterrows()],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[extract] wrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
