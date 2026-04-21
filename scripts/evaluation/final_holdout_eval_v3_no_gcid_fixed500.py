"""V3-no-gcid TEST-partition trade builder.

Sibling of `final_holdout_eval_v3_c1_fixed500.py` — same contract, swaps
`inference_v3` for `inference_v3_no_gcid` so `_top_perf_common` can
dispatch V3-combo-agnostic via `version="v3_no_gcid"` with no further
downstream changes.

Unlike shipped V3, this module deliberately does NOT wire a two-stage
per-combo calibrator — per CLAUDE.md Phase 5D, per-combo calibration is a
per-combo memorization path and is antithetical to the combo-agnostic
audit. `build_combo_trades_test` accepts the `two_stage` argument for
signature parity with the V3 caller but ignores it and always returns
`pwin_twostage=None` / `has_per_combo_cal=False`.
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

spec_v3 = importlib.util.spec_from_file_location(
    "v3inf", REPO / "scripts/models/inference_v3_no_gcid.py"
)
v3inf = importlib.util.module_from_spec(spec_v3); spec_v3.loader.exec_module(v3inf)

DOLLARS_PER_POINT = 2.0


def build_combo_trades_test(gcid: str, booster, simple_cals,
                            two_stage=None) -> dict:
    """Run OOS backtest on the TEST partition and attach V3-no-gcid P(win)."""
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

    pwin_simple = v3inf.predict_pwin_v3_no_gcid_at_rr(
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
        "has_per_combo_cal": False,
        "entry_bar": entry_bars,
        "exit_bar": exit_bars,
        "entry_time": entry_times,
        "sl_pts": sl_pts,
        "pnl_pts": pnl_pts,
        "label_win": label_win,
        "pwin_simple": pwin_simple,
        "pwin_twostage": None,
    }
