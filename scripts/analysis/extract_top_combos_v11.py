"""Extract top-K combos under ML#1 v11 UCB ranking.

Loads combo_features_v11.parquet + the four v11 boosters, scores every
combo, ranks by UCB = p50 + kappa * (p90 - p10) / 2 (default kappa=0),
applies a flat n_trades floor, and writes top_strategies_v11.json with a
schema identical to the legacy top_strategies.json so composed_strategy
_runner consumes it unchanged.

kappa=0 recovers pure p50-only exploit ranking. Raise kappa to reward
combos with wide predictive intervals (exploration / robustness check).

Usage:
    python scripts/analysis/extract_top_combos_v11.py
    python scripts/analysis/extract_top_combos_v11.py --top-k 10 --kappa 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parents[2]
ML1_DIR = REPO / "data" / "ml" / "ml1_results_v11"
FEATURES_PARQUET = ML1_DIR / "combo_features_v11.parquet"
MODELS_DIR = ML1_DIR / "models"
DEFAULT_OUT = REPO / "evaluation" / "top_strategies_v11.json"

CATEGORICAL_COLS = [
    "stop_method", "z_input", "z_anchor", "z_denom", "z_type",
    "session_filter_mode",
]
BOOLEAN_COLS = [
    "exit_on_opposite_signal", "use_breakeven_stop", "zscore_confirmation",
    "tight_stop_flag",
]
EXCLUDE_COLS = {"global_combo_id", "combo_id", "target_net_sharpe"}

# Schema-compatible with the legacy top_strategies.json entry format.
# composed_strategy_runner reads entry["parameters"] + stop_fixed_pts_resolved.
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
REALISED_COLS = [
    "n_trades", "gross_win_rate", "gross_sharpe",
    "target_net_sharpe", "gross_net_sharpe_gap",
    "median_stop_pts", "median_hold_bars",
    "median_mfe_pts", "median_mae_pts",
    "cost_as_pct_of_edge",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-trades", type=int, default=500)
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="UCB exploration weight (0 = exploit = p50 only)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    return ap.parse_args()


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[c for c in df.columns if c not in EXCLUDE_COLS]].copy()
    for c in BOOLEAN_COLS:
        if c in X.columns:
            X[c] = X[c].astype(float).fillna(0).astype(int)
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def _to_jsonable(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def _row_to_entry(row: pd.Series) -> dict:
    params = {c: _to_jsonable(row[c]) for c in PARAM_COLS if c in row}
    realised = {c: _to_jsonable(row[c]) for c in REALISED_COLS if c in row}
    return {
        "global_combo_id": str(row["global_combo_id"]),
        "predicted_point": float(row["pred_point"]),
        "predicted_p10": float(row["pred_p10"]),
        "predicted_p50": float(row["pred_p50"]),
        "predicted_p90": float(row["pred_p90"]),
        "ucb_score": float(row["ucb_score"]),
        "realised": realised,
        "parameters": params,
    }


def main() -> None:
    args = parse_args()
    if not args.features.exists():
        print(f"ERROR: {args.features} missing", file=sys.stderr)
        sys.exit(1)

    print(f"[v11-extract] loading {args.features.relative_to(REPO)}", flush=True)
    df = pd.read_parquet(args.features)
    print(f"[v11-extract] {len(df):,} combos", flush=True)

    X = _prepare(df)

    preds: dict[str, np.ndarray] = {}
    for name in ["net_sharpe_point", "net_sharpe_p10", "net_sharpe_p50",
                 "net_sharpe_p90"]:
        p = args.models_dir / f"{name}.txt"
        if not p.exists():
            print(f"ERROR: missing booster {p}", file=sys.stderr)
            sys.exit(1)
        mdl = lgb.Booster(model_file=str(p))
        preds[name] = mdl.predict(X)
        print(f"[v11-extract] scored {name}: "
              f"p50={np.median(preds[name]):.3f}", flush=True)

    df["pred_point"] = preds["net_sharpe_point"]
    df["pred_p10"] = preds["net_sharpe_p10"]
    df["pred_p50"] = preds["net_sharpe_p50"]
    df["pred_p90"] = preds["net_sharpe_p90"]
    df["ucb_score"] = df["pred_p50"] + args.kappa * (
        df["pred_p90"] - df["pred_p10"]) / 2.0

    eligible = df[df["n_trades"] >= args.min_trades]
    rejected = len(df) - len(eligible)
    print(f"[v11-extract] n_trades >= {args.min_trades}: "
          f"{len(eligible):,} eligible, {rejected:,} rejected", flush=True)

    top = eligible.sort_values("ucb_score", ascending=False).head(args.top_k)

    # Regression/sanity: v10_9264 should now rank in the bottom half.
    v10_9264_rank = None
    if "v10_9264" in df["global_combo_id"].values:
        sorted_all = df.sort_values("ucb_score", ascending=False)
        sorted_all = sorted_all.reset_index(drop=True)
        v10_9264_rank = int(sorted_all.index[
            sorted_all["global_combo_id"] == "v10_9264"][0])
        pct = v10_9264_rank / len(sorted_all)
        print(f"[v11-extract] v10_9264 rank: "
              f"{v10_9264_rank}/{len(sorted_all)} ({pct:.1%})", flush=True)

    print(f"[v11-extract] top-{args.top_k}: "
          f"{list(top['global_combo_id'])}", flush=True)

    out = {
        "source_features": str(args.features.relative_to(REPO)).replace("\\", "/"),
        "source_models_dir": str(args.models_dir.relative_to(REPO)).replace("\\", "/"),
        "ranking_metric": f"UCB(p50, p90, p10, kappa={args.kappa})",
        "kappa": args.kappa,
        "min_trades": args.min_trades,
        "top_k": args.top_k,
        "pool_sizes": {
            "total_combos": int(len(df)),
            "eligible_combos": int(len(eligible)),
            "rejected_min_trades": int(rejected),
        },
        "regression_checks": {
            "v10_9264_rank": v10_9264_rank,
            "v10_9264_percentile": (v10_9264_rank / len(df))
                                   if v10_9264_rank is not None else None,
        },
        "top": [_row_to_entry(r) for _, r in top.iterrows()],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[v11-extract] wrote {args.output.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
