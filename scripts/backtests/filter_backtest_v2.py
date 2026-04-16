"""Option 1 filter test: use adaptive R:R model as go/no-go filter at fixed R:R.

For each combo, predict P(win) at the combo's native min_rr. Skip trades whose
E[R] = P*R - (1-P) < threshold. Compare fixed baseline vs fixed+filter.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("avf", REPO / "scripts/backtests/adaptive_vs_fixed_backtest_v1.py")
avf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avf)

V2_MODEL = REPO / "data/ml/adaptive_rr_v2/adaptive_rr_model.txt"
HELDOUT_MODEL = REPO / "data/ml/adaptive_rr_heldout_v2/adaptive_rr_model.txt"
OUT = REPO / "data/ml/adaptive_rr_v2/filter_backtest_v2.json"

EV_THRESHOLDS = [0.0, 0.1, 0.25]


def load_combo_by_id(gcid: str) -> dict:
    """Mirror avf.load_top_combo() but for an arbitrary gcid."""
    source_version = int(gcid.split("_")[0][1:])
    combo_id = int(gcid.split("_")[1])
    combo = {"global_combo_id": gcid, "source_version": source_version,
             "combo_id": combo_id}
    parq = REPO / f"data/ml/mfe/ml_dataset_v{source_version}_mfe.parquet"
    df_c = pd.read_parquet(parq, filters=[("combo_id", "==", combo_id)])
    meta_cols = [
        "z_band_k", "z_window", "volume_zscore_window", "ema_fast", "ema_slow",
        "stop_method", "stop_fixed_pts", "atr_multiplier", "swing_lookback",
        "min_rr", "exit_on_opposite_signal", "use_breakeven_stop", "max_hold_bars",
        "zscore_confirmation", "z_input", "z_anchor", "z_denom", "z_type",
        "z_window_2", "z_window_2_weight", "volume_entry_threshold",
        "vol_regime_lookback", "vol_regime_min_pct", "vol_regime_max_pct",
        "session_filter_mode", "tod_exit_hour",
    ]
    first = df_c.iloc[0]
    for c in meta_cols:
        if c in df_c.columns:
            v = first[c]
            combo[c] = None if (isinstance(v, (float, np.floating)) and pd.isna(v)) else v
        else:
            combo[c] = None
    return combo


def predict_pwin_at_rr(features: pd.DataFrame, model, rr: float) -> np.ndarray:
    """Predict P(win) for each row at a fixed target R:R using the V2 booster.

    Args:
        df: Per-trade feature frame (numeric + categorical columns).
        rr: Target R:R level (must appear in `RR_LEVELS`).

    Returns:
        1-D array of calibrated probabilities aligned to `df`.
    """
    f = features.copy()
    f["candidate_rr"] = np.float32(rr)
    f["rr_x_atr"] = np.float32(rr) * f["atr_points"].to_numpy()
    X = f[avf.FEATURE_ORDER]
    return model.predict(X.values)


def backtest_one(gcid: str, model) -> dict:
    """Run one V2 filter backtest at a fixed R:R for one combo.

    Applies `predict_pwin_at_rr`, filters trades by a P(win) threshold, and
    reports Sharpe/return/WR.

    Args:
        df: Per-trade frame for a single combo.
        rr: Target R:R.

    Returns:
        Metrics dict.
    """
    combo = load_combo_by_id(gcid)
    rr = float(combo["min_rr"])
    df = avf.load_bars(avf.DATA_CSV)
    train, _ = avf.split_train_test(df, 0.8)
    df_ind = avf.build_indicators(train, combo)
    stop_pts = avf.resolve_stop_pts(combo, df_ind)
    cfg = avf.make_cfg(combo, stop_pts)
    df_sig = avf.generate_signals(df_ind, cfg)
    trades = avf.run_core(df_sig, cfg)
    n = len(trades["side"])
    if n == 0:
        return {"global_combo_id": gcid, "error": "no trades"}

    feats = avf.build_features(trades, df_sig, stop_pts,
                               str(combo["stop_method"]),
                               bool(combo["exit_on_opposite_signal"]))
    pwin = predict_pwin_at_rr(feats, model, rr)
    ev = pwin * rr - (1.0 - pwin)

    # Fixed baseline (all trades)
    fixed_res = avf.simulate_fixed(trades)
    fixed_m = avf.compute_equity_and_metrics(fixed_res, trades, cfg, "fixed")

    results = {
        "global_combo_id": gcid,
        "min_rr": rr,
        "n_signals": n,
        "mean_pwin": float(pwin.mean()),
        "mean_ev_R": float(ev.mean()),
        "fixed": fixed_m,
    }

    for thr in EV_THRESHOLDS:
        keep = ev >= thr
        n_kept = int(keep.sum())
        if n_kept == 0:
            results[f"filter_ev_ge_{thr}"] = {"n_trades": 0, "skipped": "no trades pass filter"}
            continue
        # Subset trades dict
        subset = {k: (np.asarray(v)[keep] if hasattr(v, "__len__") and len(v) == n else v)
                  for k, v in trades.items()}
        fres = avf.simulate_fixed(subset)
        fm = avf.compute_equity_and_metrics(fres, subset, cfg, f"filter_{thr}")
        fm["skip_rate"] = float(1 - keep.mean())
        results[f"filter_ev_ge_{thr}"] = fm

    return results


def main():
    """B-option-1 V2: use adaptive R:R model as go/no-go filter at fixed R:R."""
    import lightgbm as lgb
    high_freq = ["v10_7649", "v10_8617", "v10_9264", "v10_9393", "v6_1676"]
    low_freq = ["v10_9955", "v5_158", "v5_2904", "v7_2114", "v7_215"]

    model_v2 = lgb.Booster(model_file=str(V2_MODEL))

    all_results = {"model": "v2", "combos": []}
    for gcid in ["v10_9955"] + high_freq + low_freq:
        # Dedup v10_9955 if it appears twice
        if any(r.get("global_combo_id") == gcid for r in all_results["combos"]):
            continue
        print(f"\n=== {gcid} ===")
        try:
            r = backtest_one(gcid, model_v2)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e)}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r:
            fx = r["fixed"]
            print(f"  fixed: ret={fx.get('total_return_pct',0):.1f}% sharpe={fx.get('sharpe_ratio',0):.2f} "
                  f"dd={fx.get('max_drawdown_pct',0):.1f}% wr={fx.get('win_rate',0):.3f} n={fx.get('n_trades',0)}")
            for thr in EV_THRESHOLDS:
                f = r[f"filter_ev_ge_{thr}"]
                if "skipped" in f:
                    print(f"  filter(E[R]>={thr}): skipped ({f['skipped']})")
                else:
                    print(f"  filter(E[R]>={thr}): ret={f.get('total_return_pct',0):.1f}% "
                          f"sharpe={f.get('sharpe_ratio',0):.2f} dd={f.get('max_drawdown_pct',0):.1f}% "
                          f"wr={f.get('win_rate',0):.3f} n={f.get('n_trades',0)} "
                          f"skip={f.get('skip_rate',0):.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
