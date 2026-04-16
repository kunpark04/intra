"""B16: FINAL held-out evaluation on test partition (20% of bars).

Single-shot OOS eval of the V3 production stack. Do not re-run with parameter
tweaks after peeking — that defeats the held-out test's purpose.

Scope:
- 10 combos (top-5 high-freq + bottom-5 low-freq from filter_backtest_v2.py)
- Per-combo solo metrics: fixed5 sizing vs kelly_twostage sizing
- Portfolio sim (top-5 combined): fixed5 vs kelly_simple vs kelly_twostage
- Warm cohort (>=300 trades) vs cold cohort split
- Decision verdict: % of warm combos where kelly_twostage Sharpe > fixed5 Sharpe

Output: data/ml/adaptive_rr_v3/final_holdout_eval_v3.json
"""
from __future__ import annotations
import importlib.util, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec_fb = importlib.util.spec_from_file_location("fb", REPO / "scripts/backtests/filter_backtest_v2.py")
fb = importlib.util.module_from_spec(spec_fb); spec_fb.loader.exec_module(fb)

spec_v3 = importlib.util.spec_from_file_location("v3inf", REPO / "scripts/models/inference_v3.py")
v3inf = importlib.util.module_from_spec(spec_v3); spec_v3.loader.exec_module(v3inf)

OUT = REPO / "data/ml/adaptive_rr_v3/final_holdout_eval_v3.json"

HIGH_FREQ = ["v10_7649", "v10_8617", "v10_9264", "v10_9393", "v6_1676"]
LOW_FREQ = ["v10_9955", "v5_158", "v5_2904", "v7_2114", "v7_215"]
ALL_COMBOS = HIGH_FREQ + LOW_FREQ

STARTING_EQUITY = 50_000.0
DOLLARS_PER_POINT = 2.0
RISK_CAP_FRAC = 0.05
WARM_MIN_TRADES = 300


def build_combo_trades_test(gcid: str, booster, simple_cals, two_stage) -> dict:
    """Run backtest on TEST partition (20% OOS) and attach P(win) predictions."""
    avf = fb.avf
    combo = fb.load_combo_by_id(gcid)
    rr = float(combo["min_rr"])

    df = avf.load_bars(avf.DATA_CSV)
    _, test = avf.split_train_test(df, 0.8)  # TEST partition
    df_ind = avf.build_indicators(test, combo)
    stop_pts = avf.resolve_stop_pts(combo, df_ind)
    cfg = avf.make_cfg(combo, stop_pts)
    df_sig = avf.generate_signals(df_ind, cfg)
    trades = avf.run_core(df_sig, cfg)
    n = len(trades["side"])
    if n == 0:
        return {"combo_id": gcid, "n_trades": 0, "error": "no trades"}

    feats = avf.build_features(trades, df_sig, stop_pts,
                               str(combo["stop_method"]),
                               bool(combo["exit_on_opposite_signal"]))

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

    return {
        "combo_id": gcid,
        "rr": rr,
        "stop_pts": float(stop_pts),
        "n_trades": int(n),
        "has_per_combo_cal": gcid in two_stage["per_combo"],
        "entry_bar": entry_bars,
        "exit_bar": exit_bars,
        "entry_time": entry_times,
        "sl_pts": sl_pts,
        "pnl_pts": pnl_pts,
        "pwin_simple": pwin_simple,
        "pwin_twostage": pwin_twostage,
    }


def solo_metrics(c: dict, pwin_key: str | None, policy: str) -> dict:
    """Single-combo metrics under a given sizing policy."""
    if c.get("error"):
        return {"error": c.get("error")}
    pnl_dollars = []
    equity = STARTING_EQUITY
    for ti in range(c["n_trades"]):
        if policy == "fixed5":
            risk_dollars = STARTING_EQUITY * RISK_CAP_FRAC
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
        if policy != "fixed5":
            equity += pnl
    n = len(pnl_dollars)
    if n == 0:
        return {"n_trades_taken": 0}
    arr = np.asarray(pnl_dollars, dtype=np.float64)
    wr = float((arr > 0).mean())
    total_ret = float(arr.sum()) / STARTING_EQUITY * 100.0
    sharpe = (float(arr.mean() / arr.std(ddof=1) * math.sqrt(len(arr)))
              if n > 1 and arr.std(ddof=1) > 0 else 0.0)
    return {
        "policy": policy if policy == "fixed5" else f"kelly:{pwin_key}",
        "n_trades_taken": n,
        "win_rate": round(wr, 4),
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 4),
        "sum_pnl_dollars": round(float(arr.sum()), 2),
    }


def simulate_portfolio(combos: list[dict], pwin_key: str, policy: str) -> dict:
    """Event-driven multi-combo portfolio sim (same as portfolio_sim_v3)."""
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
        "policy": f"{policy}:{pwin_key}" if policy == "kelly" else policy,
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(equity, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "n_trades_taken": sum(len(x) for x in per_combo_pnl),
    }


def main() -> None:
    t0 = time.time()
    import lightgbm as lgb
    print("=== B16 FINAL HELD-OUT EVAL (test partition, 20% OOS) ===")
    print("Loading V3 booster + calibrators...")
    booster = lgb.Booster(model_file=str(v3inf.V3_BOOSTER))
    simple_cals = v3inf._load_calibrators()
    two_stage = v3inf._load_per_combo_calibrators()

    print(f"\nBuilding TEST-partition trades for {len(ALL_COMBOS)} combos...")
    combos: list[dict] = []
    for gcid in ALL_COMBOS:
        print(f"  {gcid}...", flush=True)
        try:
            c = build_combo_trades_test(gcid, booster, simple_cals, two_stage)
            if c.get("error"):
                print(f"    ERROR: {c['error']}")
            else:
                print(f"    n_trades={c['n_trades']} rr={c['rr']:.2f} "
                      f"per_combo_cal={'YES' if c['has_per_combo_cal'] else 'no'} "
                      f"mean_pwin_simple={c['pwin_simple'].mean():.3f} "
                      f"mean_pwin_twostage={c['pwin_twostage'].mean():.3f}")
        except Exception as e:
            c = {"combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"    EXCEPTION: {e}")
        combos.append(c)

    # Per-combo solo metrics under each policy.
    print("\n=== Per-combo solo metrics (OOS) ===")
    per_combo = []
    for c in combos:
        entry = {
            "combo_id": c["combo_id"],
            "n_trades": c.get("n_trades", 0),
            "rr": c.get("rr"),
            "has_per_combo_cal": c.get("has_per_combo_cal", False),
            "fixed5": solo_metrics(c, None, "fixed5"),
            "kelly_simple": solo_metrics(c, "pwin_simple", "kelly"),
            "kelly_twostage": solo_metrics(c, "pwin_twostage", "kelly"),
        }
        per_combo.append(entry)
        if c.get("error"):
            print(f"  {c['combo_id']}: ERROR {c['error']}")
            continue
        fx = entry["fixed5"]
        k2 = entry["kelly_twostage"]
        print(f"  {c['combo_id']:12s} n={c['n_trades']:5d} "
              f"fixed5: ret={fx.get('total_return_pct',0):+.1f}% "
              f"sharpe={fx.get('sharpe',0):+.2f} | "
              f"kelly2s: ret={k2.get('total_return_pct',0):+.1f}% "
              f"sharpe={k2.get('sharpe',0):+.2f}")

    # Portfolio sim (top-5).
    print("\n=== Top-5 portfolio sim (OOS) ===")
    top5 = combos[:5]
    port = []
    for policy, pwin_key in [("fixed5", "pwin_simple"),
                              ("kelly", "pwin_simple"),
                              ("kelly", "pwin_twostage")]:
        r = simulate_portfolio(top5, pwin_key, policy)
        port.append(r)
        print(f"  [{r['policy']:22s}] ret={r['total_return_pct']:+.2f}% "
              f"sharpe={r['sharpe_ratio']:+.3f} dd={r['max_drawdown_pct']:.2f}% "
              f"n={r['n_trades_taken']}")

    # Warm/cold decision verdict.
    warm = [c for c in per_combo if not c.get("fixed5", {}).get("error")
            and c["n_trades"] >= WARM_MIN_TRADES]
    cold = [c for c in per_combo if not c.get("fixed5", {}).get("error")
            and c["n_trades"] < WARM_MIN_TRADES]

    def cmp_improve(entries, key_a, key_b):
        """% of entries where entries[key_b].sharpe > entries[key_a].sharpe."""
        valid = [e for e in entries
                 if not e[key_a].get("error") and not e[key_b].get("error")
                 and e[key_a].get("n_trades_taken", 0) > 0
                 and e[key_b].get("n_trades_taken", 0) > 0]
        if not valid:
            return {"pct": None, "n": 0}
        improved = sum(1 for e in valid
                       if e[key_b]["sharpe"] > e[key_a]["sharpe"])
        return {
            "pct": round(improved / len(valid) * 100, 1),
            "n": len(valid), "n_improved": improved,
        }

    verdict = {
        "warm_combos": len(warm),
        "cold_combos": len(cold),
        "warm_kelly_twostage_vs_fixed5": cmp_improve(warm, "fixed5", "kelly_twostage"),
        "warm_kelly_twostage_vs_kelly_simple": cmp_improve(warm, "kelly_simple", "kelly_twostage"),
        "cold_kelly_twostage_vs_fixed5": cmp_improve(cold, "fixed5", "kelly_twostage"),
        "pass_threshold_pct": 60.0,
        "pass_pass":
            (cmp_improve(warm, "fixed5", "kelly_twostage").get("pct") or 0) >= 60.0,
    }

    print("\n=== DECISION VERDICT ===")
    print(f"  warm combos (>={WARM_MIN_TRADES} trades): {verdict['warm_combos']}")
    print(f"  cold combos: {verdict['cold_combos']}")
    v = verdict["warm_kelly_twostage_vs_fixed5"]
    print(f"  warm kelly_twostage beats fixed5: {v.get('n_improved','?')}/{v.get('n','?')} = {v.get('pct','?')}%")
    print(f"  pass (>=60%): {verdict['pass_pass']}")

    output = {
        "script": "scripts/evaluation/final_holdout_eval_v3.py",
        "partition": "test (20% OOS)",
        "combos_evaluated": ALL_COMBOS,
        "warm_min_trades": WARM_MIN_TRADES,
        "starting_equity": STARTING_EQUITY,
        "risk_cap_frac": RISK_CAP_FRAC,
        "per_combo_solo": per_combo,
        "portfolio_top5": port,
        "verdict": verdict,
        "runtime_seconds": time.time() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, default=float))
    print(f"\nSaved: {OUT}")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
