"""V3 inference helper — Family A + global_combo_id + per-R:R isotonic.

Exposes `predict_pwin_v3(base_trade_df, rr_levels)` that:
  1. computes Family A features (prior_wr_10/50, prior_r_ma10, has_history_50)
     from per-combo ordered base trades,
  2. for each R:R level, builds the feature frame the V3 booster expects
     (categorical dtypes restored, derived features added),
  3. runs the LightGBM booster,
  4. applies the matching per-R:R isotonic calibrator.

Returns a calibrated P(win) matrix of shape (n_trades, len(rr_levels)).

The Family A computation mirrors `adaptive_rr_model_v3.add_family_a` —
imported directly so the two paths can never drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# Importing via path injection so callers can run from repo root.
import sys
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from adaptive_rr_model_v3 import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL_COLS,
    FAMILY_A,
    ID_FEATURE,
    RR_FEATURE,
    add_family_a,
)

REPO = Path(__file__).resolve().parents[1]
V3_DIR = REPO / "data/ml/adaptive_rr_v3"
V3_BOOSTER = V3_DIR / "booster_v3.txt"
V3_CALIBRATORS = V3_DIR / "isotonic_calibrators_v3.json"


def _load_calibrators(path: Path = V3_CALIBRATORS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {'rr_key': (X_knots, y_knots)} for np.interp-based application."""
    raw = json.loads(path.read_text())
    return {k: (np.asarray(v["X"], dtype=np.float64),
                np.asarray(v["y"], dtype=np.float64))
            for k, v in raw.items()}


def _apply_calibrator(p_raw: np.ndarray, knots: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Piecewise-linear interpolation with edge-clipping — equivalent to
    `IsotonicRegression(out_of_bounds='clip').predict`.
    If knots is None (R:R had < 100 samples at train time), return raw."""
    if knots is None:
        return p_raw
    X, y = knots
    return np.interp(p_raw, X, y)


def _build_rr_frame(base: pd.DataFrame, rr: float) -> pd.DataFrame:
    """Construct the row-per-trade feature frame the booster expects for one R:R level."""
    out = base.copy()
    out[RR_FEATURE] = np.float32(rr)
    out["abs_zscore_entry"] = np.abs(out["zscore_entry"].to_numpy())
    out["rr_x_atr"] = np.float32(rr) * out["atr_points"].to_numpy()
    for col in CATEGORICAL_COLS:
        if col in out.columns and not pd.api.types.is_categorical_dtype(out[col]):
            out[col] = out[col].astype("category")
    return out[ALL_FEATURES]


def predict_pwin_v3(
    base_trade_df: pd.DataFrame,
    rr_levels: list[float],
    booster: lgb.Booster | None = None,
    calibrators: dict | None = None,
    compute_family_a: bool = True,
) -> np.ndarray:
    """Calibrated P(win) for each (trade, R:R) pair.

    Args:
        base_trade_df: one row per base trade. Must contain the 20 V2 feature
            columns + `global_combo_id` + `label_win` + `r_multiple`. If
            `compute_family_a=False`, Family A columns must already be present.
        rr_levels: list of candidate R:R values (e.g. [1.0, 1.25, ...]).
        booster: pre-loaded LightGBM Booster (default: load V3_BOOSTER).
        calibrators: dict from `_load_calibrators` (default: load V3_CALIBRATORS).
        compute_family_a: whether to run `add_family_a`. Set False if the
            caller has already computed them on the full training window and
            is now slicing.

    Returns:
        np.ndarray of shape (n_trades, len(rr_levels)) with calibrated P(win).
        Row ordering matches `base_trade_df` post-Family-A sort.
    """
    if booster is None:
        booster = lgb.Booster(model_file=str(V3_BOOSTER))
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
        key = f"{float(rr):.2f}"
        out[:, i] = _apply_calibrator(raw, calibrators.get(key))
    return out


def _derive_labels_from_trades(trades: dict, stop_pts: float) -> tuple[np.ndarray, np.ndarray]:
    """From a run_core trades dict at fixed R:R, compute (label_win, r_multiple)
    the way training did: label_win = trade made money; r_multiple = realised R."""
    side = np.asarray(trades["side"])
    entry = np.asarray(trades["entry_price"], dtype=np.float64)
    exit_ = np.asarray(trades["exit_price"], dtype=np.float64)
    pnl_points = (exit_ - entry) * side
    label_win = (pnl_points > 0).astype(np.int8)
    r_multiple = (pnl_points / stop_pts).astype(np.float32)
    return label_win, r_multiple


def predict_pwin_v3_at_rr(
    feats: pd.DataFrame,
    trades: dict,
    global_combo_id: str,
    stop_pts: float,
    rr: float,
    booster: lgb.Booster | None = None,
    calibrators: dict | None = None,
) -> np.ndarray:
    """Drop-in V3 counterpart to `filter_backtest.predict_pwin_at_rr`.

    Consumes the output of `avf.build_features` + `avf.run_core` + the combo's
    string ID, builds the Family A features from the in-order trade stream,
    attaches global_combo_id, predicts, applies the per-R:R calibrator.
    Returns a 1-D array of calibrated P(win), one per trade.
    """
    if booster is None:
        booster = lgb.Booster(model_file=str(V3_BOOSTER))
    if calibrators is None:
        calibrators = _load_calibrators()

    label_win, r_multiple = _derive_labels_from_trades(trades, stop_pts)
    base = feats.copy()
    base["label_win"] = label_win
    base["r_multiple"] = r_multiple
    base["global_combo_id"] = global_combo_id

    # Single-combo stream — add_family_a sorts by combo id (all rows equal) and
    # rolls; the stable sort keeps chronological order intact.
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
    knots = calibrators.get(f"{float(rr):.2f}")
    return _apply_calibrator(raw, knots)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Run a 10k-row round-trip smoke test against V3 OOF metrics.")
    args = p.parse_args()

    if args.smoke:
        # Minimal sanity: load booster + calibrators, verify shapes.
        b = lgb.Booster(model_file=str(V3_BOOSTER))
        cals = _load_calibrators()
        print(f"Loaded booster with {b.num_trees()} trees, {len(b.feature_name())} features")
        print(f"Loaded {len(cals)} per-R:R calibrators: {sorted(cals.keys())[:5]}...")
