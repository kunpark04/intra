"""V4 TEST-partition trade builder — V3 parity minus the two-stage calibrator.

Parallels `final_holdout_eval_v3_c1_fixed500.py` but swaps inference_v3 for
inference_v4 and drops the per-combo two-stage path (deprecated in Phase 5D,
not rebuilt in V4).

Exposes the same `build_combo_trades_test(gcid, booster, simple_cals, _)`
contract used by `_top_perf_common._load_or_build_combos_ml2`, so that caller
can swap between V3 and V4 via a version kwarg with no other downstream
changes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec_fb = importlib.util.spec_from_file_location(
    "fb", REPO / "scripts/backtests/filter_backtest_v2.py"
)
fb = importlib.util.module_from_spec(spec_fb); spec_fb.loader.exec_module(fb)

spec_v4 = importlib.util.spec_from_file_location(
    "v4inf", REPO / "scripts/models/inference_v4.py"
)
v4inf = importlib.util.module_from_spec(spec_v4); spec_v4.loader.exec_module(v4inf)

DOLLARS_PER_POINT = 2.0


def build_combo_trades_test(gcid: str, booster, simple_cals,
                            two_stage=None) -> dict:
    """Run OOS backtest on the TEST partition and attach V4 P(win).

    Signature mirrors V3's so _top_perf_common can call either uniformly;
    `two_stage` is accepted but ignored (V4 ships only pooled per-R:R).
    """
    avf = fb.avf
    combo = fb.load_combo_by_id(gcid)
    rr = float(combo["min_rr"])

    df = avf.load_bars(avf.DATA_CSV)
    _, test = avf.split_train_test(df, 0.8)
    df_ind = avf.build_indicators(test, combo)
    stop_pts = avf.resolve_stop_pts(combo, df_ind)
    cfg = avf.make_cfg(combo, stop_pts)
    df_sig = avf.generate_signals(df_ind, cfg)
    trades = avf.run_core(df_sig, cfg)
    n = len(trades["side"])
    if n == 0:
        return {"combo_id": gcid, "n_trades": 0, "error": "no trades"}

    feats = avf.build_features(
        trades, df_sig, stop_pts,
        str(combo["stop_method"]),
        bool(combo["exit_on_opposite_signal"]),
    )

    pwin_simple = v4inf.predict_pwin_v4_at_rr(
        feats, trades, gcid, stop_pts, rr,
        booster=booster, calibrators=simple_cals,
    )

    bar_times = pd.to_datetime(df_sig["time"].to_numpy())
    entry_bars = trades["entry_bar"].astype(np.int64)
    exit_bars = trades["exit_bar"].astype(np.int64)
    entry_times = bar_times[entry_bars]

    sl_pts = np.abs(trades["entry_price"] - trades["sl"]).astype(np.float64)
    pnl_pts = ((trades["exit_price"] - trades["entry_price"]) *
               trades["side"].astype(np.float64))
    label_win = (pnl_pts > 0).astype(np.int8)

    return {
        "combo_id": gcid,
        "rr": rr,
        "stop_pts": float(stop_pts) if np.isscalar(stop_pts) else None,
        "n_trades": int(n),
        "has_per_combo_cal": False,  # V4 has no two-stage path
        "entry_bar": entry_bars,
        "exit_bar": exit_bars,
        "entry_time": entry_times,
        "sl_pts": sl_pts,
        "pnl_pts": pnl_pts,
        "label_win": label_win,
        "pwin_simple": pwin_simple,
        # V4 has no two-stage calibrator (retired Phase 5D). Consumers must
        # gate on has_per_combo_cal=False rather than treating pwin_twostage
        # as distinct evidence — a silent alias would make a V4 "two-stage"
        # row in a results table falsely match the simple path.
        "pwin_twostage": None,
    }
