"""V4 combo-agnostic inference — V4 architecture minus `global_combo_id`.

Sibling of `inference_v4.py` that drives the refit produced by
`adaptive_rr_model_v4.py --no-combo-id`. The shipped V4 booster still lives
at `data/ml/adaptive_rr_v4/`; this module points at
`data/ml/adaptive_rr_v4_no_gcid/` and redefines `ALL_FEATURES` +
`CATEGORICAL_COLS` locally so module-global lookups in this file's
functions resolve to the combo-agnostic feature list.

Why a separate module instead of a flag on inference_v4: Python functions
look up module-level names via the defining module's `__dict__`; mutating
`inference_v4.ALL_FEATURES` after import would globally bias any caller
that later imports the shipped V4 path in the same process.

Exposes (mirrors `inference_v4`):
  V4_BOOSTER, V4_CALIBRATORS, V4_DIR        — new paths
  predict_pwin_v4(...)                      — same signature
  predict_pwin_v4_at_rr(...)                — same signature
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
    ALL_FEATURES as _V4_ALL_FEATURES,
    CATEGORICAL_COLS as _V4_CATEGORICAL_COLS,
    FAMILY_A,  # noqa: F401  (re-exported for signature parity)
    ID_FEATURE,
    RR_FEATURE,
    add_family_a,
)

# Combo-agnostic feature list. `global_combo_id` is stripped both as a
# direct input and as a LightGBM categorical — the refit booster was
# trained without it, so feeding the ID column would fail a strict
# feature-count check on predict().
ALL_FEATURES = [c for c in _V4_ALL_FEATURES if c != ID_FEATURE]
CATEGORICAL_COLS = [c for c in _V4_CATEGORICAL_COLS if c != ID_FEATURE]

REPO = Path(__file__).resolve().parents[2]
V4_DIR = REPO / "data/ml/adaptive_rr_v4_no_gcid"
V4_BOOSTER = V4_DIR / "booster_v4.txt"
V4_CALIBRATORS = V4_DIR / "isotonic_calibrators_v4.json"


def _load_calibrators(path: Path = V4_CALIBRATORS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw = json.loads(path.read_text())
    return {k: (np.asarray(v["X"], dtype=np.float64),
                np.asarray(v["y"], dtype=np.float64))
            for k, v in raw.items()}


def _apply_calibrator(p_raw: np.ndarray,
                      knots: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if knots is None:
        return p_raw
    X, y = knots
    return np.interp(p_raw, X, y)


def _build_rr_frame(base: pd.DataFrame, rr: float) -> pd.DataFrame:
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
    """Combo-agnostic counterpart to inference_v4.predict_pwin_v4_at_rr.

    `global_combo_id` is accepted for signature parity (callers pass it
    uniformly across V3/V4/V4-no-gcid) but is NOT attached to the feature
    frame or fed to the booster — the refit was trained without it.
    """
    if booster is None:
        booster = lgb.Booster(model_file=str(V4_BOOSTER))
    if calibrators is None:
        calibrators = _load_calibrators()

    label_win, r_multiple = _derive_labels_from_trades(trades, stop_pts)
    base = feats.copy()
    base["label_win"] = label_win
    base["r_multiple"] = r_multiple
    # Family A needs a grouping key to compute rolling WR; use a constant
    # sentinel so all trades in this call share a single group (this is
    # the single-combo path, so per-combo grouping is implicit).
    base[ID_FEATURE] = global_combo_id

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
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        b = lgb.Booster(model_file=str(V4_BOOSTER))
        cals = _load_calibrators()
        print(f"V4-no-gcid booster: {b.num_trees()} trees, {len(b.feature_name())} features")
        print(f"Feature names: {b.feature_name()}")
        print(f"Expected ALL_FEATURES: {ALL_FEATURES}")
        print(f"V4-no-gcid calibrators: {len(cals)} keys: {sorted(cals.keys())[:5]}...")
