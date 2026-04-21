"""V3 combo-agnostic inference — V3 architecture minus `global_combo_id`.

Sibling of `inference_v3.py` that drives the refit produced by
`adaptive_rr_model_v3.py --no-combo-id`. The shipped V3 booster still lives
at `data/ml/adaptive_rr_v3/`; this module points at
`data/ml/adaptive_rr_v3_no_gcid/` and redefines `ALL_FEATURES` +
`CATEGORICAL_COLS` locally so module-global lookups in this file's
functions resolve to the combo-agnostic feature list.

Why a separate module instead of a flag on inference_v3: Python functions
look up module-level names via the defining module's `__dict__`; mutating
`inference_v3.ALL_FEATURES` after import would globally bias any caller
that later imports the shipped V3 path in the same process.

Keeps V3's pooled per-R:R isotonic calibrator wired in (V4 template has no
isotonic; this is the V3-specific extension). The deprecated two-stage
per-combo calibrator is intentionally NOT mirrored here — per CLAUDE.md
Phase 5D, per-combo calibration is a per-combo memorization path and is
antithetical to the combo-agnostic audit.

Exposes (mirrors `inference_v3`):
  V3_BOOSTER, V3_CALIBRATORS, V3_DIR        — new paths
  predict_pwin_v3_no_gcid(...)              — same signature as predict_pwin_v3
  predict_pwin_v3_no_gcid_at_rr(...)        — same signature as predict_pwin_v3_at_rr
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

from adaptive_rr_model_v3 import (  # noqa: E402
    ALL_FEATURES as _V3_ALL_FEATURES,
    CATEGORICAL_COLS as _V3_CATEGORICAL_COLS,
    FAMILY_A,  # noqa: F401  (re-exported for signature parity)
    ID_FEATURE,
    RR_FEATURE,
    add_family_a,
)

# Combo-agnostic feature list. `global_combo_id` is stripped both as a
# direct input and as a LightGBM categorical — the refit booster was
# trained without it, so feeding the ID column would fail a strict
# feature-count check on predict().
ALL_FEATURES = [c for c in _V3_ALL_FEATURES if c != ID_FEATURE]
CATEGORICAL_COLS = [c for c in _V3_CATEGORICAL_COLS if c != ID_FEATURE]

REPO = Path(__file__).resolve().parents[2]
V3_DIR = REPO / "data/ml/adaptive_rr_v3_no_gcid"
V3_BOOSTER = V3_DIR / "booster_v3.txt"
V3_CALIBRATORS = V3_DIR / "isotonic_calibrators_v3.json"


def _load_calibrators(path: Path = V3_CALIBRATORS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {'rr_key': (X_knots, y_knots)} for np.interp-based application."""
    raw = json.loads(path.read_text())
    return {k: (np.asarray(v["X"], dtype=np.float64),
                np.asarray(v["y"], dtype=np.float64))
            for k, v in raw.items()}


def _apply_calibrator(p_raw: np.ndarray,
                      knots: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
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


def predict_pwin_v3_no_gcid(
    base_trade_df: pd.DataFrame,
    rr_levels: list[float],
    booster: lgb.Booster | None = None,
    calibrators: dict | None = None,
    compute_family_a: bool = True,
) -> np.ndarray:
    """Calibrated P(win) for each (trade, R:R) pair, combo-agnostic V3.

    Same signature as `inference_v3.predict_pwin_v3`. Family A features are
    still computed per-combo (they depend on `global_combo_id` as a grouping
    key during training-data preparation, not as a booster feature). The
    grouping key is still required in `base_trade_df` — it is used only for
    rolling-window alignment, not fed to the booster.
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


def _derive_labels_from_trades(trades: dict,
                               stop_pts: float) -> tuple[np.ndarray, np.ndarray]:
    """From a run_core trades dict at fixed R:R, compute (label_win, r_multiple)
    the way training did: label_win = trade made money; r_multiple = realised R."""
    side = np.asarray(trades["side"])
    entry = np.asarray(trades["entry_price"], dtype=np.float64)
    exit_ = np.asarray(trades["exit_price"], dtype=np.float64)
    pnl_points = (exit_ - entry) * side
    label_win = (pnl_points > 0).astype(np.int8)
    r_multiple = (pnl_points / stop_pts).astype(np.float32)
    return label_win, r_multiple


def predict_pwin_v3_no_gcid_at_rr(
    feats: pd.DataFrame,
    trades: dict,
    global_combo_id: str,
    stop_pts: float,
    rr: float,
    booster: lgb.Booster | None = None,
    calibrators: dict | None = None,
) -> np.ndarray:
    """Combo-agnostic counterpart to `inference_v3.predict_pwin_v3_at_rr`.

    `global_combo_id` is accepted for signature parity (callers pass it
    uniformly across V3/V3-no-gcid/V4/V4-no-gcid) and is still used to
    assemble Family A rolling features (grouping key only). It is NOT
    attached to the feature frame or fed to the booster — the refit was
    trained without it.
    """
    if booster is None:
        booster = lgb.Booster(model_file=str(V3_BOOSTER))
    if calibrators is None:
        calibrators = _load_calibrators()

    label_win, r_multiple = _derive_labels_from_trades(trades, stop_pts)
    base = feats.copy()
    base["label_win"] = label_win
    base["r_multiple"] = r_multiple
    # Family A needs a grouping key to compute rolling WR; attach the combo
    # id as a column for `add_family_a` but it is stripped from ALL_FEATURES
    # at predict time.
    base["global_combo_id"] = global_combo_id

    # Single-combo stream — add_family_a sorts by combo id (all rows equal)
    # and rolls; the stable sort keeps chronological order intact.
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
                   help="Round-trip smoke test: load booster + calibrators + verify shapes.")
    args = p.parse_args()

    if args.smoke:
        b = lgb.Booster(model_file=str(V3_BOOSTER))
        cals = _load_calibrators()
        print(f"V3-no-gcid booster: {b.num_trees()} trees, {len(b.feature_name())} features")
        print(f"Feature names: {b.feature_name()}")
        print(f"Expected ALL_FEATURES: {ALL_FEATURES}")
        print(f"V3-no-gcid calibrators: {len(cals)} keys: {sorted(cals.keys())[:5]}...")
