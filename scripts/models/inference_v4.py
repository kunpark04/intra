"""V4 inference helper — V3 architecture retrained on v11 sweep.

Byte-identical to inference_v3 but imports feature constants + `add_family_a`
from `adaptive_rr_model_v4` and points at `data/ml/adaptive_rr_v4/` artifacts.

V4 deliberately ships only the pooled per-R:R isotonic calibrator — the per-
combo two-stage variant was retired in Phase 5D (null-to-negative results
across four evals) and is not rebuilt here.

Exposes:
  predict_pwin_v4(base_trade_df, rr_levels) -> (n, k) calibrated P(win)
  predict_pwin_v4_at_rr(feats, trades, combo_id, stop_pts, rr) -> (n,)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from adaptive_rr_model_v4 import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_COLS,
    FAMILY_A,
    ID_FEATURE,
    RR_FEATURE,
    add_family_a,
)

REPO = Path(__file__).resolve().parents[2]
V4_DIR = REPO / "data/ml/adaptive_rr_v4"
V4_BOOSTER = V4_DIR / "booster_v4.txt"
V4_CALIBRATORS = V4_DIR / "isotonic_calibrators_v4.json"


def _load_calibrators(path: Path = V4_CALIBRATORS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {'rr_key': (X_knots, y_knots)} for np.interp-based application."""
    raw = json.loads(path.read_text())
    return {k: (np.asarray(v["X"], dtype=np.float64),
                np.asarray(v["y"], dtype=np.float64))
            for k, v in raw.items()}


def _apply_calibrator(p_raw: np.ndarray,
                      knots: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Piecewise-linear interpolation equivalent to IsotonicRegression.predict."""
    if knots is None:
        return p_raw
    X, y = knots
    return np.interp(p_raw, X, y)


def _build_rr_frame(base: pd.DataFrame, rr: float) -> pd.DataFrame:
    """Construct the feature frame the V4 booster expects for one R:R level."""
    out = base.copy()
    out[RR_FEATURE] = np.float32(rr)
    out["abs_zscore_entry"] = np.abs(out["zscore_entry"].to_numpy())
    out["rr_x_atr"] = np.float32(rr) * out["atr_points"].to_numpy()
    for col in CATEGORICAL_COLS:
        if col in out.columns and not pd.api.types.is_categorical_dtype(out[col]):
            out[col] = out[col].astype("category")
    return out[ALL_FEATURES]


def predict_pwin_v4(
    base_trade_df: pd.DataFrame,
    rr_levels: list[float],
    booster: lgb.Booster | None = None,
    calibrators: dict | None = None,
    compute_family_a: bool = True,
) -> np.ndarray:
    """Calibrated P(win) matrix of shape (n_trades, len(rr_levels))."""
    if booster is None:
        booster = lgb.Booster(model_file=str(V4_BOOSTER))
    if calibrators is None:
        calibrators = _load_calibrators()

    df = base_trade_df.copy()
    if compute_family_a:
        df = add_family_a(df)

    n = len(df)
    out = np.empty((n, len(rr_levels)), dtype=np.float64)
    for i, rr in enumerate(rr_levels):
        frame = _build_rr_frame(df, rr)
        raw = booster.predict(frame)
        out[:, i] = _apply_calibrator(raw, calibrators.get(f"{float(rr):.2f}"))
    return out


def _derive_labels_from_trades(trades: dict,
                               stop_pts: float) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild (label_win, r_multiple) from a run_core trades dict."""
    side = np.asarray(trades["side"])
    entry = np.asarray(trades["entry_price"], dtype=np.float64)
    exit_ = np.asarray(trades["exit_price"], dtype=np.float64)
    pnl_points = (exit_ - entry) * side
    label_win = (pnl_points > 0).astype(np.int8)
    r_multiple = (pnl_points / stop_pts).astype(np.float32)
    return label_win, r_multiple


def predict_pwin_v4_at_rr(
    feats: pd.DataFrame,
    trades: dict,
    global_combo_id: str,
    stop_pts: float,
    rr: float,
    booster: lgb.Booster | None = None,
    calibrators: dict | None = None,
) -> np.ndarray:
    """V4 counterpart to predict_pwin_v3_at_rr — per-R:R calibrated P(win)."""
    if booster is None:
        booster = lgb.Booster(model_file=str(V4_BOOSTER))
    if calibrators is None:
        calibrators = _load_calibrators()

    label_win, r_multiple = _derive_labels_from_trades(trades, stop_pts)
    base = feats.copy()
    base["label_win"] = label_win
    base["r_multiple"] = r_multiple
    base["global_combo_id"] = global_combo_id

    # v4 Family A wants entry_bar_idx for chronological sort; if caller hasn't
    # supplied one, synthesise sequential bar indices (all trades in feat order).
    # CONTRACT: callers must pass `feats` in chronological (entry-bar) order —
    # this call path processes a single combo, so the synthetic 0..n-1 index
    # preserves the trade order under Family A's stable sort. A caller that
    # shuffles `feats` before inference will silently mis-compute rolling WR.
    if "entry_bar_idx" not in base.columns:
        base["entry_bar_idx"] = np.arange(len(base), dtype=np.int64)

    base = add_family_a(base)

    rr32 = np.float32(rr)
    base[RR_FEATURE] = rr32
    if "abs_zscore_entry" not in base.columns:
        base["abs_zscore_entry"] = np.abs(base["zscore_entry"].to_numpy())
    base["rr_x_atr"] = rr32 * base["atr_points"].to_numpy()
    for col in CATEGORICAL_COLS:
        if col in base.columns and not pd.api.types.is_categorical_dtype(base[col]):
            base[col] = base[col].astype("category")

    raw = booster.predict(base[ALL_FEATURES])
    return _apply_calibrator(raw, calibrators.get(f"{float(rr):.2f}"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Load booster + calibrators, print shapes.")
    args = p.parse_args()

    if args.smoke:
        b = lgb.Booster(model_file=str(V4_BOOSTER))
        cals = _load_calibrators()
        print(f"V4 booster: {b.num_trees()} trees, {len(b.feature_name())} features")
        print(f"V4 calibrators: {len(cals)} keys: {sorted(cals.keys())[:5]}...")
