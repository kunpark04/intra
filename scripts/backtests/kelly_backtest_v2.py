"""B10: Kelly-fraction sizing from calibrated P(win).

For each combo, predict P(win) at its native min_rr via V2 model. Compute
per-trade Kelly fraction f* = max(0, E[R]/R) where E[R] = p*R - (1-p).

Simulate with fixed-equity sizing (matches filter_backtest_v2 convention) but
variable risk fraction per trade. Compare full / half / quarter Kelly
(each optionally capped at 5% = current fixed baseline) vs the fixed 5%
baseline.

Output: data/ml/adaptive_rr_v2/kelly_backtest_v2.json
"""
from __future__ import annotations
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("fb", REPO / "scripts/backtests/filter_backtest_v2.py")
fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)

spec2 = importlib.util.spec_from_file_location(
    "fbs", REPO / "scripts/backtests/filter_backtest_surrogate_v2.py")
fbs_ok = (REPO / "scripts/backtests/filter_backtest_surrogate_v2.py").exists()
if fbs_ok:
    fbs = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fbs)

OUT = REPO / "data/ml/adaptive_rr_v2/kelly_backtest_v2.json"
SURROGATE_CSV = REPO / "data/ml/ml1_results_v2filtered/surrogate_top_combos.csv"

# Kelly fraction multipliers to test; None = no multiplier (full Kelly).
KELLY_VARIANTS = [
    ("kelly_full",         1.00, None),   # cap=None means no cap
    ("kelly_full_cap5",    1.00, 0.05),
    ("kelly_half",         0.50, None),
    ("kelly_half_cap5",    0.50, 0.05),
    ("kelly_quarter",      0.25, None),
    ("kelly_quarter_cap5", 0.25, 0.05),
]

# Hard floor: we do not size below this fraction of the baseline unless f*<=0
# (skip). 0.005 = 0.5% of equity. Smaller fractions round contracts to 0 in
# the fixed-equity sizing model, producing no PnL regardless.
MIN_KELLY_F = 0.005


def compute_kelly_f(pwin: np.ndarray, rr: float) -> np.ndarray:
    """Kelly fraction for binary (win R, lose 1) bet: f = (p*R - (1-p)) / R."""
    ev = pwin * rr - (1.0 - pwin)
    f = ev / rr
    return np.clip(f, 0.0, None)


def equity_metrics_kelly(result: dict, trades: dict, cfg, f_per_trade: np.ndarray,
                         label: str) -> dict:
    """Fixed-equity, variable-risk: contracts_i sized from f_i * START_EQ / stop.

    Mirrors avf.compute_equity_and_metrics but takes per-trade risk fraction.
    Trades with f_i <= 0 are SKIPPED (pnl=0, counted in skip_rate, not n_trades).
    """
    start_eq = float(cfg.STARTING_EQUITY)
    dpp = float(cfg.MNQ_DOLLARS_PER_POINT)
    pnl_pts = result["pnl_pts"]
    sl_pts = result["sl_pts"]
    n = len(pnl_pts)

    pnl_dollars = np.zeros(n, dtype=np.float64)
    taken = np.zeros(n, dtype=bool)
    for i in range(n):
        s = sl_pts[i]
        f = float(f_per_trade[i])
        if s <= 0 or f <= 0.0:
            continue
        contracts = int(start_eq * f // (s * dpp))
        if contracts <= 0:
            continue
        pnl_dollars[i] = pnl_pts[i] * contracts * dpp
        taken[i] = True

    n_taken = int(taken.sum())
    equity = np.concatenate([[start_eq], start_eq + np.cumsum(pnl_dollars)])
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.max(dd))
    final = float(equity[-1])
    total_ret = (final - start_eq) / start_eq * 100.0

    ret_series = pnl_dollars[taken] / start_eq if n_taken else np.array([])
    if len(ret_series) > 1 and np.std(ret_series, ddof=1) > 0:
        sharpe = float(np.mean(ret_series) / np.std(ret_series, ddof=1)
                        * math.sqrt(len(ret_series)))
    else:
        sharpe = 0.0

    wins = int(np.sum(result["labels"][taken])) if n_taken else 0
    wr = wins / n_taken if n_taken else 0.0

    return {
        "label": label,
        "n_trades": n_taken,
        "n_skipped": int(n - n_taken),
        "skip_rate": float(1 - n_taken / n) if n else 0.0,
        "win_rate": round(wr, 4),
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(final, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "sum_pnl_dollars": round(float(np.sum(pnl_dollars)), 2),
        "mean_f": round(float(np.mean(f_per_trade[taken])) if n_taken else 0.0, 4),
        "p95_f": round(float(np.quantile(f_per_trade[taken], 0.95))
                       if n_taken else 0.0, 4),
    }


def backtest_one(gcid: str, combo: dict, model) -> dict:
    avf = fb.avf
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
    pwin = fb.predict_pwin_at_rr(feats, model, rr)
    ev = pwin * rr - (1.0 - pwin)
    kelly_raw = compute_kelly_f(pwin, rr)

    fixed_res = avf.simulate_fixed(trades)
    fixed_m = avf.compute_equity_and_metrics(fixed_res, trades, cfg, "fixed_5pct")

    out = {
        "global_combo_id": gcid,
        "min_rr": rr,
        "n_signals": n,
        "mean_pwin": float(pwin.mean()),
        "mean_ev_R": float(ev.mean()),
        "kelly_full_mean": float(kelly_raw.mean()),
        "kelly_full_p95": float(np.quantile(kelly_raw, 0.95)),
        "fixed_5pct": fixed_m,
    }

    for name, mult, cap in KELLY_VARIANTS:
        f = kelly_raw * mult
        f = np.where(f < MIN_KELLY_F, 0.0, f)  # drop tiny bets
        if cap is not None:
            f = np.minimum(f, cap)
        m = equity_metrics_kelly(fixed_res, trades, cfg, f, name)
        out[name] = m

    return out


def main():
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(fb.V2_MODEL))
    all_results = {"model": "v2",
                   "kelly_variants": [[n, m, c] for n, m, c in KELLY_VARIANTS],
                   "min_kelly_f": MIN_KELLY_F,
                   "combos": []}

    high_freq = ["v10_7649", "v10_8617", "v10_9264", "v10_9393", "v6_1676"]
    low_freq = ["v10_9955", "v5_158", "v5_2904", "v7_2114", "v7_215"]

    for gcid in high_freq + low_freq:
        print(f"\n=== {gcid} ===", flush=True)
        try:
            combo = fb.load_combo_by_id(gcid)
            r = backtest_one(gcid, combo, model)
        except Exception as e:
            r = {"global_combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"  ERROR: {e}")
        all_results["combos"].append(r)
        if "error" not in r:
            fx = r["fixed_5pct"]
            print(f"  fixed_5pct: sharpe={fx['sharpe_ratio']:.2f} "
                  f"ret={fx['total_return_pct']:.1f}% dd={fx['max_drawdown_pct']:.1f}% "
                  f"n={fx['n_trades']}")
            for name, _m, _c in KELLY_VARIANTS:
                k = r[name]
                print(f"  {name:20s}: sharpe={k['sharpe_ratio']:.2f} "
                      f"ret={k['total_return_pct']:.1f}% dd={k['max_drawdown_pct']:.1f}% "
                      f"n={k['n_trades']} skip={k['skip_rate']:.1%} "
                      f"mean_f={k['mean_f']:.3f}")

    # Optional: also evaluate v2filtered top 10 if present
    if SURROGATE_CSV.exists() and fbs_ok:
        print("\n=== v2filtered surrogate top-10 ===", flush=True)
        sdf = pd.read_csv(SURROGATE_CSV).head(10)
        for i, row in sdf.iterrows():
            gcid = f"v2f_surr_{i}"
            print(f"\n--- {gcid} ---", flush=True)
            try:
                combo = fbs.row_to_combo(i, row)
                r = backtest_one(gcid, combo, model)
            except Exception as e:
                r = {"global_combo_id": gcid, "error": str(e),
                     "tb": traceback.format_exc()[:400]}
                print(f"  ERROR: {e}")
            all_results["combos"].append(r)
            if "error" not in r:
                fx = r["fixed_5pct"]
                print(f"  fixed_5pct: sharpe={fx['sharpe_ratio']:.2f} "
                      f"ret={fx['total_return_pct']:.1f}% n={fx['n_trades']}")
                for name, _m, _c in KELLY_VARIANTS:
                    k = r[name]
                    print(f"  {name:20s}: sharpe={k['sharpe_ratio']:.2f} "
                          f"ret={k['total_return_pct']:.1f}% n={k['n_trades']} "
                          f"mean_f={k['mean_f']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
