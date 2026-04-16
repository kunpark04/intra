"""Phase 5C: Portfolio-level simulation — top-5 ML#1 combos + V3 + Kelly-cap5.

Runs 5 combos simultaneously on the same bars with shared $50k equity.
Compares two sizing policies:
  A) "fixed5": existing rule — 5% of starting equity per trade (no Kelly).
  B) "kelly_simple": Kelly cap-5% using simple per-R:R calibrator.
  C) "kelly_twostage": Kelly cap-5% using two-stage per-combo calibrator.

The point: Phase 5A showed per-combo calibration doesn't help threshold
filtering. This script tests whether per-combo calibration helps Kelly
sizing, where absolute P(win) matters.

Output: data/ml/adaptive_rr_v3/portfolio_sim_v3.json
  - Per-policy portfolio: Sharpe, max DD, total return, final equity
  - Per-combo correlation matrix (daily PnL)
  - Per-combo individual contributions
"""
from __future__ import annotations
import importlib.util, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

spec_fb = importlib.util.spec_from_file_location("fb", REPO / "scripts/filter_backtest_v2.py")
fb = importlib.util.module_from_spec(spec_fb); spec_fb.loader.exec_module(fb)

spec_v3 = importlib.util.spec_from_file_location("v3inf", REPO / "scripts/inference_v3.py")
v3inf = importlib.util.module_from_spec(spec_v3); spec_v3.loader.exec_module(v3inf)

OUT = REPO / "data/ml/adaptive_rr_v3/portfolio_sim_v3.json"

# Top-5 ML#1 combos (same list used in Phase 3 filter backtests).
TOP5 = ["v10_7649", "v10_8617", "v10_9264", "v10_9393", "v6_1676"]

STARTING_EQUITY = 50_000.0
DOLLARS_PER_POINT = 2.0  # MNQ
RISK_CAP_FRAC = 0.05     # 5% of current equity per trade (Kelly cap)


def kelly_fraction(pwin: np.ndarray, rr: float) -> np.ndarray:
    """Standard Kelly for binary outcome with odds b = rr.
    f* = (b*p - q) / b = p - (1-p)/b
    Clipped to [0, 1] — negative Kelly means skip (return 0).
    """
    f = pwin - (1.0 - pwin) / rr
    return np.clip(f, 0.0, 1.0)


def build_combo_trades(gcid: str, booster, simple_cals, two_stage) -> dict:
    """Run the full backtest pipeline for one combo; return trades + P(win)
    from both simple and two-stage calibrators, plus the bar time index."""
    avf = fb.avf
    combo = fb.load_combo_by_id(gcid)
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

    # Bar times for correlation analysis (use entry_bar to index df_sig).
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
        "n_trades": n,
        "has_per_combo_cal": gcid in two_stage["per_combo"],
        "entry_bar": entry_bars,
        "exit_bar": exit_bars,
        "entry_time": entry_times,
        "sl_pts": sl_pts,
        "pnl_pts": pnl_pts,
        "pwin_simple": pwin_simple,
        "pwin_twostage": pwin_twostage,
        "trade_wins": (pnl_pts > 0).astype(np.int8),
    }


def simulate_portfolio(combos: list[dict], pwin_key: str,
                       policy: str) -> dict:
    """Event-driven portfolio sim. Processes trades in entry-bar order,
    updates equity on each exit, sizes positions via the chosen policy.

    policy:
      - 'fixed5': 5% of STARTING equity per trade (no compounding, no Kelly).
      - 'kelly':  Kelly(pwin, rr) capped at RISK_CAP_FRAC, on current equity.
    pwin_key: 'pwin_simple' or 'pwin_twostage' (ignored if policy='fixed5').
    """
    # Flatten into event stream: (bar, type, combo_idx, trade_idx).
    events: list[tuple[int, int, int, int]] = []  # (bar, kind, combo, trade)
    for ci, c in enumerate(combos):
        if c.get("error"):
            continue
        for ti in range(c["n_trades"]):
            events.append((int(c["entry_bar"][ti]), 0, ci, ti))  # entry
            events.append((int(c["exit_bar"][ti]), 1, ci, ti))   # exit
    # Sort: exits before entries at the same bar (conservative — frees equity first).
    events.sort(key=lambda e: (e[0], -e[1]))

    n_combos = len(combos)
    equity = STARTING_EQUITY
    realized = 0.0
    # Open positions: {(combo_idx, trade_idx): contracts}
    open_pos: dict[tuple[int, int], int] = {}
    # PnL series per combo for correlation.
    per_combo_pnl: list[list[float]] = [[] for _ in range(n_combos)]
    per_combo_entry_times: list[list[pd.Timestamp]] = [[] for _ in range(n_combos)]

    # Walk events.
    portfolio_equity = [STARTING_EQUITY]
    trade_equity_at_entry = []
    trade_contracts = []
    for bar, kind, ci, ti in events:
        c = combos[ci]
        if kind == 0:  # entry
            # Size the position.
            if policy == "fixed5":
                # 5% of starting equity risk
                risk_dollars = STARTING_EQUITY * RISK_CAP_FRAC
                contracts = int(risk_dollars // (c["sl_pts"][ti] * DOLLARS_PER_POINT))
            else:  # kelly
                p = float(c[pwin_key][ti])
                f = p - (1.0 - p) / c["rr"]
                f = max(0.0, min(RISK_CAP_FRAC, f))
                if f <= 0.0:
                    continue  # skip negative Kelly
                risk_dollars = equity * f
                contracts = int(risk_dollars // (c["sl_pts"][ti] * DOLLARS_PER_POINT))
            if contracts <= 0:
                continue
            open_pos[(ci, ti)] = contracts
            trade_equity_at_entry.append(equity)
            trade_contracts.append(contracts)
        else:  # exit
            key = (ci, ti)
            if key not in open_pos:
                continue
            contracts = open_pos.pop(key)
            pnl = c["pnl_pts"][ti] * contracts * DOLLARS_PER_POINT
            realized += pnl
            equity = STARTING_EQUITY + realized
            portfolio_equity.append(equity)
            per_combo_pnl[ci].append(pnl)
            per_combo_entry_times[ci].append(c["entry_time"][ti])

    # Metrics.
    eq_arr = np.asarray(portfolio_equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / peak
    max_dd = float(dd.max())
    total_ret = (equity - STARTING_EQUITY) / STARTING_EQUITY * 100.0

    # Trade-level Sharpe.
    all_pnl = [p for plist in per_combo_pnl for p in plist]
    if len(all_pnl) > 1:
        rets = np.asarray(all_pnl) / STARTING_EQUITY
        sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(len(rets)))
    else:
        sharpe = 0.0

    # Per-combo aggregates.
    per_combo = []
    for ci, c in enumerate(combos):
        if c.get("error"):
            per_combo.append({"combo_id": c["combo_id"], "error": c.get("error")})
            continue
        n = len(per_combo_pnl[ci])
        sum_pnl = float(sum(per_combo_pnl[ci]))
        wr = float(sum(1 for p in per_combo_pnl[ci] if p > 0) / n) if n else 0.0
        per_combo.append({
            "combo_id": c["combo_id"],
            "n_trades_taken": n,
            "n_trades_available": c["n_trades"],
            "sum_pnl_dollars": round(sum_pnl, 2),
            "win_rate": round(wr, 4),
        })

    return {
        "policy": f"{policy}:{pwin_key}" if policy == "kelly" else policy,
        "n_combos_active": sum(1 for c in combos if not c.get("error")),
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(equity, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "n_trades_taken": sum(len(x) for x in per_combo_pnl),
        "per_combo": per_combo,
    }


def compute_correlation_matrix(combos: list[dict], pnl_by_combo: list[list[tuple]]) -> dict:
    """Daily PnL correlation between combos (Pearson).
    pnl_by_combo: list of [(entry_time, pnl)] per combo."""
    n = len(combos)
    labels = [c.get("combo_id", f"combo_{i}") for i, c in enumerate(combos)]

    # Build daily pnl series per combo.
    daily_series = []
    all_days = set()
    for plist in pnl_by_combo:
        if not plist:
            daily_series.append({})
            continue
        df = pd.DataFrame({"t": [t for t, _ in plist],
                           "p": [p for _, p in plist]})
        df["day"] = pd.to_datetime(df["t"]).dt.date
        s = df.groupby("day")["p"].sum().to_dict()
        daily_series.append(s)
        all_days.update(s.keys())

    all_days = sorted(all_days)
    mat = np.zeros((n, len(all_days)), dtype=np.float64)
    for i, s in enumerate(daily_series):
        for j, d in enumerate(all_days):
            mat[i, j] = s.get(d, 0.0)

    if len(all_days) < 2:
        return {"labels": labels, "matrix": None, "n_days": len(all_days)}

    corr = np.corrcoef(mat)
    # Convert NaN (zero-variance combos) to 0 for JSON.
    corr = np.nan_to_num(corr, nan=0.0)
    return {
        "labels": labels,
        "matrix": [[round(float(corr[i, j]), 4) for j in range(n)] for i in range(n)],
        "n_days": len(all_days),
    }


def main() -> None:
    t0 = time.time()
    import lightgbm as lgb
    print("Loading V3 booster + calibrators...")
    booster = lgb.Booster(model_file=str(v3inf.V3_BOOSTER))
    simple_cals = v3inf._load_calibrators()
    two_stage = v3inf._load_per_combo_calibrators()
    print(f"  Two-stage: {len(two_stage['per_combo'])} per-combo, "
          f"{len(two_stage['pooled_per_rr'])} pooled per-R:R")

    print(f"\nBuilding trades for {len(TOP5)} combos...")
    combos: list[dict] = []
    for gcid in TOP5:
        print(f"  {gcid}...", flush=True)
        try:
            c = build_combo_trades(gcid, booster, simple_cals, two_stage)
            if c.get("error"):
                print(f"    ERROR: {c['error']}")
            else:
                print(f"    n_trades={c['n_trades']} rr={c['rr']:.2f} "
                      f"stop_pts={c['stop_pts']:.2f} "
                      f"per_combo_cal={'YES' if c['has_per_combo_cal'] else 'no'} "
                      f"mean_pwin_simple={c['pwin_simple'].mean():.3f} "
                      f"mean_pwin_twostage={c['pwin_twostage'].mean():.3f}")
        except Exception as e:
            c = {"combo_id": gcid, "error": str(e),
                 "tb": traceback.format_exc()[:400]}
            print(f"    EXCEPTION: {e}")
        combos.append(c)

    # Run the 3 policies.
    print(f"\nSimulating policies...")
    policies = [
        ("fixed5", "pwin_simple"),    # baseline; pwin_key unused
        ("kelly",  "pwin_simple"),
        ("kelly",  "pwin_twostage"),
    ]
    results = []
    # Collect pnl-with-times for correlation (same across policies — use first kelly run).
    pnl_by_combo_for_corr: list[list[tuple]] = []
    for policy, pwin_key in policies:
        r = simulate_portfolio(combos, pwin_key, policy)
        results.append(r)
        print(f"  [{r['policy']}] "
              f"ret={r['total_return_pct']:.2f}% "
              f"sharpe={r['sharpe_ratio']:.3f} "
              f"max_dd={r['max_drawdown_pct']:.2f}% "
              f"n_taken={r['n_trades_taken']}")

    # Correlation: use trade outcomes regardless of sizing (they're the same pools).
    # We compute per-combo daily pnl based on the full "fixed5" trade set.
    for ci, c in enumerate(combos):
        if c.get("error"):
            pnl_by_combo_for_corr.append([])
            continue
        # dollar pnl for each trade at fixed5 sizing
        pts_pnl = []
        for ti in range(c["n_trades"]):
            risk_dollars = STARTING_EQUITY * RISK_CAP_FRAC
            contracts = int(risk_dollars // (c["sl_pts"][ti] * DOLLARS_PER_POINT))
            if contracts <= 0:
                continue
            dollars = c["pnl_pts"][ti] * contracts * DOLLARS_PER_POINT
            pts_pnl.append((c["entry_time"][ti], dollars))
        pnl_by_combo_for_corr.append(pts_pnl)

    corr = compute_correlation_matrix(combos, pnl_by_combo_for_corr)
    print(f"\nCorrelation matrix ({corr['n_days']} overlapping days):")
    if corr["matrix"]:
        for lbl, row in zip(corr["labels"], corr["matrix"]):
            print(f"  {lbl:12s} " + " ".join(f"{v:+.3f}" for v in row))

    # Per-combo fixed-5 solo metrics for reference.
    per_combo_solo = []
    for c in combos:
        if c.get("error"):
            per_combo_solo.append({"combo_id": c["combo_id"], "error": c.get("error")})
            continue
        pts_pnl_dollars = []
        for ti in range(c["n_trades"]):
            risk_dollars = STARTING_EQUITY * RISK_CAP_FRAC
            contracts = int(risk_dollars // (c["sl_pts"][ti] * DOLLARS_PER_POINT))
            pts_pnl_dollars.append(c["pnl_pts"][ti] * contracts * DOLLARS_PER_POINT)
        arr = np.asarray(pts_pnl_dollars, dtype=np.float64)
        sharpe = (float(arr.mean() / arr.std(ddof=1) * math.sqrt(len(arr)))
                  if len(arr) > 1 and arr.std(ddof=1) > 0 else 0.0)
        per_combo_solo.append({
            "combo_id": c["combo_id"],
            "n_trades": int(c["n_trades"]),
            "rr": c["rr"],
            "has_per_combo_cal": c["has_per_combo_cal"],
            "solo_return_pct": round(float(arr.sum()) / STARTING_EQUITY * 100.0, 2),
            "solo_sharpe": round(sharpe, 4),
            "mean_pwin_simple": round(float(c["pwin_simple"].mean()), 4),
            "mean_pwin_twostage": round(float(c["pwin_twostage"].mean()), 4),
        })

    output = {
        "script": "scripts/portfolio_sim_v3.py",
        "top5_combos": TOP5,
        "starting_equity": STARTING_EQUITY,
        "risk_cap_frac": RISK_CAP_FRAC,
        "dollars_per_point": DOLLARS_PER_POINT,
        "policies": results,
        "correlation": corr,
        "per_combo_solo": per_combo_solo,
        "runtime_seconds": time.time() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, default=float))
    print(f"\nSaved: {OUT}")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
