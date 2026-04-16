"""Phase 2 re-eval: V3 ML#2 stack on C1-selected top-10 under fixed $500 sizing.

Extends `final_holdout_eval_v3.py` with two changes:
1. Combo universe = evaluation/top_strategies.json (C1-selected top-10),
   not the old hardcoded HIGH/LOW_FREQ lists.
2. Adds a `fixed_dollars` sizing policy ($500 per trade, configurable),
   alongside the legacy `fixed5` ($2500 = 5% of starting equity).

Since ML#2's booster + calibrator are sizing-invariant (binary `would_win`
labels + market features), retraining is a no-op. The meaningful Phase-2
question is: does the V3 stack still generalize on the NEW C1-selected
combos when paired with fixed $500 sizing?

Output: data/ml/adaptive_rr_v3/final_holdout_eval_v3_c1_fixed500.json
"""
from __future__ import annotations
import importlib.util
import json
import math
import sys
import time
import traceback
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
    "v3inf", REPO / "scripts/models/inference_v3.py"
)
v3inf = importlib.util.module_from_spec(spec_v3); spec_v3.loader.exec_module(v3inf)

TOP_STRATEGIES = REPO / "evaluation" / "top_strategies.json"
OUT = REPO / "data" / "ml" / "adaptive_rr_v3" / "final_holdout_eval_v3_c1_fixed500.json"

STARTING_EQUITY = 50_000.0
DOLLARS_PER_POINT = 2.0
RISK_CAP_FRAC = 0.05           # legacy fixed5 policy (= $2500)
FIXED_RISK_DOLLARS = 500.0     # new production sizing convention
WARM_MIN_TRADES = 300


def load_top_combos() -> list[str]:
    payload = json.loads(TOP_STRATEGIES.read_text())
    return [entry["global_combo_id"] for entry in payload["top"]]


def build_combo_trades_test(gcid: str, booster, simple_cals, two_stage) -> dict:
    """Run backtest on TEST partition (20% OOS) and attach P(win) predictions."""
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

    pwin_simple = v3inf.predict_pwin_v3_at_rr(
        feats, trades, gcid, stop_pts, rr,
        booster=booster, calibrators=simple_cals,
    )
    pwin_twostage = v3inf.predict_pwin_v3_calibrated_at_rr(
        feats, trades, gcid, stop_pts, rr,
        booster=booster, two_stage=two_stage,
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
        "has_per_combo_cal": gcid in two_stage["per_combo"],
        "entry_bar": entry_bars,
        "exit_bar": exit_bars,
        "entry_time": entry_times,
        "sl_pts": sl_pts,
        "pnl_pts": pnl_pts,
        "label_win": label_win,
        "pwin_simple": pwin_simple,
        "pwin_twostage": pwin_twostage,
    }


def solo_metrics(c: dict, pwin_key: str | None, policy: str,
                 fixed_dollars: float = FIXED_RISK_DOLLARS) -> dict:
    """Single-combo metrics under the given sizing policy."""
    if c.get("error"):
        return {"error": c.get("error")}
    pnl_dollars = []
    equity = STARTING_EQUITY
    for ti in range(c["n_trades"]):
        if policy == "fixed5":
            risk_dollars = STARTING_EQUITY * RISK_CAP_FRAC
        elif policy == "fixed_dollars":
            risk_dollars = fixed_dollars
        else:  # kelly
            p = float(c[pwin_key][ti])
            f = p - (1.0 - p) / c["rr"]
            f = max(0.0, min(RISK_CAP_FRAC, f))
            if f <= 0.0:
                continue
            risk_dollars = equity * f
        contracts = int(risk_dollars // (c["sl_pts"][ti] * DOLLARS_PER_POINT))
        if contracts <= 0:
            continue
        pnl = c["pnl_pts"][ti] * contracts * DOLLARS_PER_POINT
        pnl_dollars.append(pnl)
        if policy == "kelly":
            equity += pnl
    n = len(pnl_dollars)
    if n == 0:
        return {"n_trades_taken": 0}
    arr = np.asarray(pnl_dollars, dtype=np.float64)
    wr = float((arr > 0).mean())
    total_ret = float(arr.sum()) / STARTING_EQUITY * 100.0
    sharpe = (float(arr.mean() / arr.std(ddof=1) * math.sqrt(len(arr)))
              if n > 1 and arr.std(ddof=1) > 0 else 0.0)
    # Max DD in percent
    equity_curve = STARTING_EQUITY + np.cumsum(arr)
    equity_curve = np.concatenate([[STARTING_EQUITY], equity_curve])
    peak = np.maximum.accumulate(equity_curve)
    dd_pct = float(((peak - equity_curve) / peak).max() * 100.0)
    return {
        "policy": (policy if policy in ("fixed5", "fixed_dollars")
                   else f"kelly:{pwin_key}"),
        "n_trades_taken": n,
        "win_rate": round(wr, 4),
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(dd_pct, 2),
        "sum_pnl_dollars": round(float(arr.sum()), 2),
    }


def simulate_portfolio(combos: list[dict], pwin_key: str | None, policy: str,
                       fixed_dollars: float = FIXED_RISK_DOLLARS) -> dict:
    """Event-driven multi-combo portfolio sim."""
    events: list[tuple[int, int, int, int]] = []
    for ci, c in enumerate(combos):
        if c.get("error"):
            continue
        for ti in range(c["n_trades"]):
            events.append((int(c["entry_bar"][ti]), 0, ci, ti))
            events.append((int(c["exit_bar"][ti]), 1, ci, ti))
    events.sort(key=lambda e: (e[0], -e[1]))

    equity = STARTING_EQUITY
    realized = 0.0
    open_pos: dict[tuple[int, int], int] = {}
    per_combo_pnl: list[list[float]] = [[] for _ in range(len(combos))]
    portfolio_equity = [STARTING_EQUITY]

    for bar, kind, ci, ti in events:
        c = combos[ci]
        if kind == 0:
            if policy == "fixed5":
                risk_dollars = STARTING_EQUITY * RISK_CAP_FRAC
            elif policy == "fixed_dollars":
                risk_dollars = fixed_dollars
            else:
                p = float(c[pwin_key][ti])
                f = p - (1.0 - p) / c["rr"]
                f = max(0.0, min(RISK_CAP_FRAC, f))
                if f <= 0.0:
                    continue
                risk_dollars = equity * f
            contracts = int(risk_dollars // (c["sl_pts"][ti] * DOLLARS_PER_POINT))
            if contracts <= 0:
                continue
            open_pos[(ci, ti)] = contracts
        else:
            key = (ci, ti)
            if key not in open_pos:
                continue
            contracts = open_pos.pop(key)
            pnl = c["pnl_pts"][ti] * contracts * DOLLARS_PER_POINT
            realized += pnl
            equity = STARTING_EQUITY + realized
            portfolio_equity.append(equity)
            per_combo_pnl[ci].append(pnl)

    eq_arr = np.asarray(portfolio_equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / peak
    max_dd = float(dd.max())
    total_ret = (equity - STARTING_EQUITY) / STARTING_EQUITY * 100.0

    all_pnl = [p for plist in per_combo_pnl for p in plist]
    if len(all_pnl) > 1:
        rets = np.asarray(all_pnl) / STARTING_EQUITY
        sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(len(rets)))
    else:
        sharpe = 0.0

    return {
        "policy": (f"{policy}:{pwin_key}" if policy == "kelly" else policy),
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(equity, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "n_trades_taken": sum(len(x) for x in per_combo_pnl),
    }


def calibration_summary(c: dict, pwin_key: str) -> dict | None:
    """Brier + ECE for the P(win) series on actual wins."""
    if c.get("error") or c["n_trades"] < 10:
        return None
    y = c["label_win"].astype(np.float64)
    p = np.asarray(c[pwin_key], dtype=np.float64)
    brier = float(((p - y) ** 2).mean())
    # 10-bin ECE
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (p >= edges[i]) & (p < edges[i+1]) if i < 9 else (p >= edges[i]) & (p <= edges[i+1])
        if mask.sum() == 0:
            continue
        avg_p = float(p[mask].mean())
        avg_y = float(y[mask].mean())
        ece += abs(avg_p - avg_y) * (mask.sum() / len(p))
    return {"brier": round(brier, 4), "ece": round(ece, 4),
            "mean_pwin": round(float(p.mean()), 4),
            "observed_wr": round(float(y.mean()), 4)}


def main() -> None:
    t0 = time.time()
    import lightgbm as lgb
    print("=== Phase 2 FINAL V3+C1+Fixed$500 OOS EVAL ===")
    all_combos = load_top_combos()
    print(f"C1-selected top-{len(all_combos)}: {all_combos}")

    print("Loading V3 booster + calibrators...")
    booster = lgb.Booster(model_file=str(v3inf.V3_BOOSTER))
    simple_cals = v3inf._load_calibrators()
    two_stage = v3inf._load_per_combo_calibrators()

    print(f"\nBuilding TEST-partition trades for {len(all_combos)} combos...")
    combos: list[dict] = []
    for gcid in all_combos:
        print(f"  {gcid}...", flush=True)
        try:
            c = build_combo_trades_test(gcid, booster, simple_cals, two_stage)
            if c.get("error"):
                print(f"    ERROR: {c['error']}")
            else:
                print(f"    n_trades={c['n_trades']} rr={c['rr']:.2f} "
                      f"per_combo_cal={'YES' if c['has_per_combo_cal'] else 'no'} "
                      f"wr={c['label_win'].mean():.3f} "
                      f"mean_pwin_simple={c['pwin_simple'].mean():.3f}")
        except Exception as e:
            c = {"combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"    EXCEPTION: {e}")
        combos.append(c)

    print("\n=== Per-combo solo metrics (OOS) ===")
    per_combo = []
    for c in combos:
        entry = {
            "combo_id": c["combo_id"],
            "n_trades": c.get("n_trades", 0),
            "rr": c.get("rr"),
            "has_per_combo_cal": c.get("has_per_combo_cal", False),
            "fixed5_2500": solo_metrics(c, None, "fixed5"),
            "fixed_dollars_500": solo_metrics(c, None, "fixed_dollars",
                                              fixed_dollars=500.0),
            "kelly_simple": solo_metrics(c, "pwin_simple", "kelly"),
            "kelly_twostage": solo_metrics(c, "pwin_twostage", "kelly"),
            "cal_simple": calibration_summary(c, "pwin_simple"),
            "cal_twostage": calibration_summary(c, "pwin_twostage"),
        }
        per_combo.append(entry)
        if c.get("error"):
            print(f"  {c['combo_id']}: ERROR {c['error']}"); continue
        fd = entry["fixed_dollars_500"]
        print(f"  {c['combo_id']}: n={c['n_trades']:>4}  "
              f"$500 sharpe={fd.get('sharpe',0):+.3f}  "
              f"ret={fd.get('total_return_pct',0):+.2f}%  "
              f"dd={fd.get('max_drawdown_pct',0):.2f}%  "
              f"wr={fd.get('win_rate',0):.3f}")

    print("\n=== Portfolio sim (all 10 combined) ===")
    portfolios = {
        "fixed5_2500": simulate_portfolio(combos, None, "fixed5"),
        "fixed_dollars_500": simulate_portfolio(combos, None, "fixed_dollars",
                                                fixed_dollars=500.0),
        "kelly_simple": simulate_portfolio(combos, "pwin_simple", "kelly"),
        "kelly_twostage": simulate_portfolio(combos, "pwin_twostage", "kelly"),
    }
    for name, p in portfolios.items():
        print(f"  {name:22s}  "
              f"sharpe={p['sharpe_ratio']:+.3f}  "
              f"ret={p['total_return_pct']:+.2f}%  "
              f"dd={p['max_drawdown_pct']:.2f}%  "
              f"final_equity=${p['final_equity']:,.0f}  "
              f"n={p['n_trades_taken']}")

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "combos": all_combos,
        "source": "evaluation/top_strategies.json (C1-selected top-10)",
        "sizing_configs": {
            "fixed5_2500": "5% of starting equity = $2500/trade, no compounding",
            "fixed_dollars_500": "FIXED $500/trade, sizing-invariant convention",
            "kelly_simple": "Kelly f = p - (1-p)/rr capped at 5%, compounded",
            "kelly_twostage": "Same but P(win) from per-combo two-stage calibrator",
        },
        "per_combo": per_combo,
        "portfolio": portfolios,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved: {OUT.relative_to(REPO)}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
