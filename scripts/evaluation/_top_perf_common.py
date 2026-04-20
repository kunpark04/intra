"""Shared setup, data loaders, and plot helpers for the 6 top-performance notebooks.

Each of the s{1..6}_*.ipynb notebooks imports this module and calls `load_setup()`
to obtain cached/rebuilt test-partition bars, per-combo backtest results, and ML2
portfolio sims. All plot helpers take DataFrames/dicts and a sizing policy; they
render a single matplotlib Figure via plt.show().

Sizing policy:
  - fixed_dollars_500: risk $500 on every trade, forever.

matplotlib backend note: adaptive_rr_model_v3 calls `matplotlib.use('Agg')` at
import time and gets pulled in transitively via inference_v3 (used by the v3
eval module). We import that module FIRST, then switch to the inline backend,
so pyplot figures get captured as image/png in downstream notebook cells.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Heavy imports BEFORE any matplotlib backend configuration
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.evaluation.composed_strategy_runner import run_strategy, load_test_bars  # noqa: E402
from src.reporting import apply_sizing, mc_policy_samples, monte_carlo  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_v3eval", REPO / "scripts/evaluation/final_holdout_eval_v3_c1_fixed500.py"
)
v3eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3eval)

# V4 eval module is loaded lazily inside load_setup(version="v4") so notebooks
# that only reference V3 don't pay the import cost. See _load_eval_module.


def _load_eval_module(version: str):
    """Return the OOS eval module for the requested ML#2 version.

    v3: final_holdout_eval_v3_c1_fixed500 — V3 booster + per-R:R isotonic +
        per-combo two-stage calibrator.
    v4: final_holdout_eval_v4_fixed500 — V4 booster + per-R:R isotonic only
        (two-stage retired in Phase 5D; V4 did not rebuild it).
    """
    if version == "v3":
        return v3eval
    if version == "v4":
        spec = importlib.util.spec_from_file_location(
            "_v4eval", REPO / "scripts/evaluation/final_holdout_eval_v4_fixed500.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    if version == "v4_no_gcid":
        spec = importlib.util.spec_from_file_location(
            "_v4nogcideval",
            REPO / "scripts/evaluation/final_holdout_eval_v4_no_gcid_fixed500.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise ValueError(
        f"Unknown ML#2 version: {version!r} "
        f"(expected 'v3', 'v4', or 'v4_no_gcid')")

# NOW configure matplotlib: inline wins over the Agg that v3eval's transitive
# imports selected.
import matplotlib  # noqa: E402

matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib_inline.backend_inline import set_matplotlib_formats  # noqa: E402

set_matplotlib_formats("png")

# ─── Constants ───────────────────────────────────────────────────────────────
STARTING_EQUITY = 50_000.0
POLICIES = ["fixed_dollars_500"]
TOP_STRATEGIES_PATH = REPO / "evaluation" / "top_strategies.json"
_FIG_W, _FIG_H = 9, 4.5
_MC_SIM_PATHS = 200

# Liberal MNQ cost model (used only when load_setup is called with
# cost_per_contract_rt>0). Breakdown of the $5 liberal default:
#   commission  ~= $2.50-$3.00 RT/contract (retail brokers + exchange fees)
#   slippage    ~= 2 ticks/side × 4 ticks RT × $0.50/tick = $2.00/contract
# Total ≈ $5.00/contract round-trip. Applied by subtracting
# contracts * cost_per_contract_rt from every trade's pnl.
DEFAULT_COST_PER_CONTRACT_RT = 5.0

_CACHE_DIR = REPO / "evaluation" / "_cache"
_RESULTS_RAW_CACHE = _CACHE_DIR / "results_raw.pkl"
_ML2_CACHE = REPO / "evaluation" / "_ml2_cache.pkl"  # legacy (V3 default)


def _ml2_cache_path(version: str) -> Path:
    """V3 cache path stays at the legacy location for backwards compatibility;
    V4 gets its own file so the two versions can coexist."""
    if version == "v3":
        return _ML2_CACHE
    return REPO / "evaluation" / f"_ml2_cache_{version}.pkl"

# ML#2 is applied as a net-of-friction E[R] filter: keep a trade only if
# p*rr - (1-p) - friction_in_R > 0, where friction_in_R = cost_rt /
# (sl_pts * DOLLARS_PER_POINT). This is sizing-invariant (friction scales
# with contracts and so does dollar risk), and it makes the filter
# tightness depend on both the calibrated win probability AND the stop's
# size relative to the cost model. Trades with tight stops have higher
# friction-in-R, so they require a higher pwin to clear the filter — which
# is exactly the friction-awareness that was missing.
ML2_PWIN_KEY = "pwin_simple"
# CLAUDE.md Phase 5D: per-combo two-stage calibrator is deprecated; pooled
# per-R:R isotonic is the production calibrator. Use pwin_simple.
# v3: net-of-friction E[R] filter now applied in _combo_ml2_base and
# _sim_ml2_portfolio. Previous v2 cached unfiltered pass-through outputs.
_ML2_CACHE_VERSION = "v3"


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics_from_pnl(pnl, years_span, policy="fixed_dollars_500", r=None,
                     start_equity=STARTING_EQUITY):
    """Headline metrics for a per-trade dollar-PnL series.

    Sharpe is annualized on per-trade $-PnL:
      sharpe = (mean/std) * sqrt(trades_per_year).
    `r` is accepted for signature stability with archived notebooks but is
    ignored — compounding sizing was removed project-wide.
    """
    p = np.asarray(pnl, dtype=float)
    n = len(p)
    if n == 0:
        return dict(n_trades=0, trades_per_year=0.0, win_rate=0.0,
                    total_pnl_dollars=0.0, total_return_pct=0.0,
                    sharpe_ratio=0.0, max_drawdown_pct=0.0,
                    max_drawdown_dollars=0.0)
    eq_full = np.concatenate([[start_equity], start_equity + np.cumsum(p)])
    peak = np.maximum.accumulate(eq_full)
    dd_d = float((peak - eq_full).max())
    dd_pct = float(np.nan_to_num((peak - eq_full) / peak, nan=0.0).max() * 100)
    total = float(p.sum())
    tpy = n / years_span if years_span > 0 else 0.0
    std_s = p.std(ddof=1) if n > 1 else 0.0
    sharpe = (float(p.mean() / std_s * np.sqrt(tpy))
              if std_s > 0 and tpy > 0 else 0.0)
    return dict(n_trades=int(n),
                trades_per_year=round(tpy, 1),
                win_rate=round(float((p > 0).mean()), 4),
                total_pnl_dollars=round(total, 2),
                total_return_pct=round(total / start_equity * 100, 2),
                sharpe_ratio=round(sharpe, 4),
                max_drawdown_pct=round(dd_pct, 2),
                max_drawdown_dollars=round(dd_d, 2))


# ─── Data loaders ────────────────────────────────────────────────────────────

def _build_results_raw(strategies, bars):
    print("Running unfiltered composed_strategy_runner for each combo...", flush=True)
    results = []
    for s in strategies:
        print(f"  {s['global_combo_id']}...", flush=True)
        results.append(run_strategy(s, bars=bars))
    return results


def _load_or_build_results_raw(strategies, bars):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = (
        "v1",
        tuple(sorted(s["global_combo_id"] for s in strategies)),
        str(bars["time"].iloc[-1]),
        (REPO / "scripts/evaluation/composed_strategy_runner.py").stat().st_mtime_ns,
    )
    if _RESULTS_RAW_CACHE.exists():
        try:
            with open(_RESULTS_RAW_CACHE, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == key:
                print(f"Loaded results_raw from cache ({len(blob['results_raw'])} combos).")
                return blob["results_raw"]
            print("results_raw cache stale; rebuilding.")
        except Exception as e:
            print(f"results_raw cache read failed ({e!r}); rebuilding.")
    results_raw = _build_results_raw(strategies, bars)
    with open(_RESULTS_RAW_CACHE, "wb") as f:
        pickle.dump({"key": key, "results_raw": results_raw}, f)
    print(f"Wrote cache -> {_RESULTS_RAW_CACHE.name}")
    return results_raw


def _load_or_build_combos_ml2(combo_ids, bars, version: str = "v3"):
    import lightgbm as lgb
    eval_mod = _load_eval_module(version)
    inf = eval_mod.v3inf if version == "v3" else eval_mod.v4inf
    booster_path = inf.V3_BOOSTER if version == "v3" else inf.V4_BOOSTER
    cal_path = inf.V3_CALIBRATORS if version == "v3" else inf.V4_CALIBRATORS
    # Both "v4" and "v4_no_gcid" expose V4_BOOSTER/V4_CALIBRATORS on their
    # respective inference modules; the combo-agnostic paths differ by
    # virtue of the inference module itself (data/ml/adaptive_rr_v4_no_gcid/).

    booster = lgb.Booster(model_file=str(booster_path))
    simple_cals = inf._load_calibrators()
    # V4 has no two-stage calibrator; pass None (build_combo_trades_test ignores).
    two_stage = (inf._load_per_combo_calibrators() if version == "v3" else None)

    cache_components = [
        _ML2_CACHE_VERSION,
        version,
        Path(booster_path).stat().st_mtime_ns,
        Path(cal_path).stat().st_mtime_ns,
        Path(eval_mod.__file__).stat().st_mtime_ns,
        str(bars["time"].iloc[-1]),
        tuple(sorted(combo_ids)),
    ]
    if version == "v3":
        cache_components.insert(
            4, Path(inf.V3_PER_COMBO_CALIBRATORS).stat().st_mtime_ns
        )
    key = tuple(cache_components)

    cache_path = _ml2_cache_path(version)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == key:
                print(f"Loaded combos_ml2 from cache ({len(blob['combos_ml2'])} combos).")
                return blob["combos_ml2"]
            print("combos_ml2 cache stale; rebuilding.")
        except Exception as e:
            print(f"combos_ml2 cache read failed ({e!r}); rebuilding.")

    print(f"Building {version.upper()}-filtered trades per combo...", flush=True)
    combos_ml2 = []
    for gcid in combo_ids:
        print(f"  {gcid}...", flush=True)
        try:
            c = eval_mod.build_combo_trades_test(
                gcid, booster, simple_cals, two_stage)
            print(f"    n_trades={c.get('n_trades', 0)}  rr={c.get('rr', float('nan')):.2f}")
        except Exception as e:
            c = {"combo_id": gcid, "error": str(e)}
            print(f"    ERROR: {e}")
        combos_ml2.append(c)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"key": key, "combos_ml2": combos_ml2}, f)
    print(f"Wrote cache -> {cache_path.name}")
    return combos_ml2


def _build_combined_raw(results_raw):
    frames = []
    for r in results_raw:
        if r["trades"].empty:
            continue
        t = r["trades"][["date", "actual_pnl", "dollar_risk"]].copy()
        t.insert(0, "combo_id", r["combo_id"])
        frames.append(t)
    if not frames:
        return pd.DataFrame(columns=["combo_id", "date", "actual_pnl", "dollar_risk"])
    return (pd.concat(frames, ignore_index=True)
            .sort_values("date", kind="mergesort")
            .reset_index(drop=True))


def _ml2_net_ev_mask(c, cost_per_contract_rt: float) -> np.ndarray:
    """Boolean mask selecting trades with positive net-of-friction E[R].

    net_ev_R = pwin * rr - (1 - pwin) - cost_rt / (sl_pts * $/pt)

    The last term converts the dollar-per-contract round-trip friction into
    R-multiples, making the filter sizing-invariant. Trades on tight stops
    pay relatively more friction-per-R, so they need a higher pwin to
    clear the threshold — the friction-awareness the old path was missing.

    When cost_rt == 0, this reduces to the standard positive-gross-EV filter.
    """
    n = int(c.get("n_trades", 0))
    if n == 0 or c.get("error"):
        return np.zeros(0, dtype=bool)
    rr = float(c["rr"])
    pwin = np.asarray(c[ML2_PWIN_KEY], dtype=np.float64)
    sl = np.asarray(c["sl_pts"], dtype=np.float64)
    gross_ev_R = pwin * rr - (1.0 - pwin)
    friction_R = (cost_per_contract_rt /
                  np.maximum(sl * v3eval.DOLLARS_PER_POINT, 1e-9))
    return (gross_ev_R - friction_R) > 0.0


def _combo_ml2_base(c, cost_per_contract_rt: float = 0.0):
    """Return (pnl_base, risk_base, exit_bars) at $500 fixed sizing for an
    ML2-filtered combo dict. Applies a net-of-friction E[R] filter using
    calibrated pwin first, then the contracts>0 mask, then (if cost>0)
    subtracts contracts*cost from each surviving trade's PnL.
    """
    if c.get("error") or c.get("n_trades", 0) == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int))
    ev_keep = _ml2_net_ev_mask(c, cost_per_contract_rt)
    sl = np.asarray(c["sl_pts"], dtype=float)[ev_keep]
    pts = np.asarray(c["pnl_pts"], dtype=float)[ev_keep]
    eb = np.asarray(c["exit_bar"], dtype=int)[ev_keep]
    if len(sl) == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int))
    contracts = (500.0 // (sl * v3eval.DOLLARS_PER_POINT)).astype(int)
    mask = contracts > 0
    pnl = (pts[mask] * contracts[mask] * v3eval.DOLLARS_PER_POINT).astype(float)
    if cost_per_contract_rt > 0:
        pnl = pnl - contracts[mask].astype(float) * cost_per_contract_rt
    risk = (sl[mask] * contracts[mask] * v3eval.DOLLARS_PER_POINT).astype(float)
    exit_bars = eb[mask]
    return pnl, risk, exit_bars


def _build_s4_pnl_by_combo(combos_ml2, cost_per_contract_rt: float = 0.0):
    """Returns {cid: {'pnl_base': ..., 'risk_base': ..., 'exit_bars': ...,
    'by_policy': {policy: pnl_policy}}}."""
    out = {}
    for c in combos_ml2:
        cid = c.get("combo_id")
        pnl_base, risk_base, exit_bars = _combo_ml2_base(c, cost_per_contract_rt)
        entry = {"pnl_base": pnl_base, "risk_base": risk_base,
                 "exit_bars": exit_bars, "by_policy": {}}
        for policy in POLICIES:
            entry["by_policy"][policy] = (apply_sizing(pnl_base, risk_base, policy)
                                          if len(pnl_base) else np.array([]))
        out[cid] = entry
    return out


def _sim_ml2_portfolio(combos_ml2, policy, bars, cost_per_contract_rt: float = 0.0):
    events = []
    for ci, c in enumerate(combos_ml2):
        if c.get("error") or c.get("n_trades", 0) == 0:
            continue
        ev_keep = _ml2_net_ev_mask(c, cost_per_contract_rt)
        for ti in range(c["n_trades"]):
            if not ev_keep[ti]:
                continue
            events.append((int(c["entry_bar"][ti]), 0, ci, ti))
            events.append((int(c["exit_bar"][ti]), 1, ci, ti))
    events.sort(key=lambda e: (e[0], -e[1]))
    equity = STARTING_EQUITY
    realized = 0.0
    open_pos = {}
    rows = []
    for bar, kind, ci, ti in events:
        c = combos_ml2[ci]
        sl = float(c["sl_pts"][ti])
        if kind == 0:
            if policy == "fixed_dollars_500":
                budget = 500.0
            else:
                raise ValueError(f"unknown policy: {policy}")
            contracts = int(budget // (sl * v3eval.DOLLARS_PER_POINT))
            if contracts <= 0:
                continue
            risk_dollars = sl * contracts * v3eval.DOLLARS_PER_POINT
            open_pos[(ci, ti)] = (contracts, risk_dollars)
        else:
            key = (ci, ti)
            if key not in open_pos:
                continue
            contracts, risk_dollars = open_pos.pop(key)
            pnl = c["pnl_pts"][ti] * contracts * v3eval.DOLLARS_PER_POINT
            if cost_per_contract_rt > 0:
                pnl -= contracts * cost_per_contract_rt
            realized += pnl
            equity = STARTING_EQUITY + realized
            exit_time = (pd.to_datetime(bars["time"].iloc[bar])
                         if bar < len(bars) else pd.NaT)
            rows.append({"combo_id": c["combo_id"], "exit_time": exit_time,
                         "actual_pnl": pnl, "dollar_risk": risk_dollars,
                         "equity_after": equity})
    return pd.DataFrame(rows)


def _apply_friction_unfiltered(results_raw, cost_per_contract_rt: float):
    """Return a shallow-copied results_raw with each trades df's `actual_pnl`
    reduced by `contracts * cost`. Leaves the on-disk cache untouched so
    both gross and net notebooks can share it."""
    out = []
    for r in results_raw:
        if r["trades"].empty:
            out.append(r); continue
        tt = r["trades"].copy()
        tt["actual_pnl"] = tt["actual_pnl"] - tt["contracts"] * cost_per_contract_rt
        out.append({**r, "trades": tt})
    return out


def load_setup(cost_per_contract_rt: float = 0.0, top_strategies_path=None,
               version: str = "v3"):
    """Returns dict with bars, years_span, strategies, results_raw,
    combined_raw, combos_ml2, s4_pnl_by_combo, ml2_portfolio. All expensive
    inputs (results_raw, combos_ml2) are loaded from disk cache when valid.

    When cost_per_contract_rt > 0, subtract `contracts * cost` from every
    trade's PnL (both unfiltered and ML2). Caches always store gross data;
    friction is applied in-memory so both modes share the on-disk cache.

    top_strategies_path defaults to evaluation/top_strategies.json (v10);
    pass a different path (e.g. evaluation/top_strategies_v11.json) to
    evaluate a different top-K source. The JSON must have a `top` list of
    entries each with `global_combo_id` and `parameters`.

    version selects the ML#2 stack: 'v3' (default, trained on v2-v10 MFE
    sweep) or 'v4' (Phase 6.7 retrain on v11 friction-aware sweep — rebuilds
    the filter for OOD v11 combos). V4 has no per-combo two-stage calibrator.
    """
    tsp = Path(top_strategies_path) if top_strategies_path else TOP_STRATEGIES_PATH
    print(f"Top-K source: {tsp.name}")

    bars = load_test_bars()
    print(f"Test partition: {len(bars):,} bars  "
          f"{bars['time'].iloc[0]} -> {bars['time'].iloc[-1]}")
    years_span = ((pd.to_datetime(bars["time"].iloc[-1])
                   - pd.to_datetime(bars["time"].iloc[0])).total_seconds()
                  / (365.25 * 86400))
    print(f"Years span: {years_span:.3f}  (used to annualize Sharpe)")
    if cost_per_contract_rt > 0:
        print(f"Applying friction: ${cost_per_contract_rt:.2f}/contract RT "
              f"(commission + slippage).")

    payload = json.loads(tsp.read_text())
    strategies = payload["top"]
    combo_ids = [s["global_combo_id"] for s in strategies]
    print(f"Loaded {len(strategies)} strategies.")

    results_raw_gross = _load_or_build_results_raw(strategies, bars)
    results_raw = (_apply_friction_unfiltered(results_raw_gross, cost_per_contract_rt)
                   if cost_per_contract_rt > 0 else results_raw_gross)
    combined_raw = _build_combined_raw(results_raw)
    print(f"Combined unfiltered trades: {len(combined_raw):,}")

    combos_ml2 = _load_or_build_combos_ml2(combo_ids, bars, version=version)
    s4_pnl_by_combo = _build_s4_pnl_by_combo(combos_ml2, cost_per_contract_rt)
    ml2_portfolio = {p: _sim_ml2_portfolio(combos_ml2, p, bars, cost_per_contract_rt)
                     for p in POLICIES}
    print(f"ML2 portfolio trade counts: "
          f"{ {p: len(ml2_portfolio[p]) for p in POLICIES} }")

    return dict(bars=bars, years_span=years_span, strategies=strategies,
                results_raw=results_raw, combined_raw=combined_raw,
                combos_ml2=combos_ml2, s4_pnl_by_combo=s4_pnl_by_combo,
                ml2_portfolio=ml2_portfolio,
                cost_per_contract_rt=cost_per_contract_rt)


# ─── Plot helpers ────────────────────────────────────────────────────────────

def _combo_base_pnl(trades):
    if trades.empty:
        return np.array([]), np.array([]), np.array([], dtype="datetime64[ns]")
    t = trades.sort_values("date", kind="mergesort")
    return (t["actual_pnl"].to_numpy(dtype=float),
            t["dollar_risk"].to_numpy(dtype=float),
            t["date"].to_numpy())


def _equity_curve(pnl_policy, start_equity=STARTING_EQUITY):
    return start_equity + np.cumsum(pnl_policy)


def _drawdown_curve(pnl_policy, start_equity=STARTING_EQUITY):
    eq_full = np.concatenate([[start_equity], start_equity + np.cumsum(pnl_policy)])
    peak = np.maximum.accumulate(eq_full)
    return (peak - eq_full) / peak * 100


def plot_indiv_equity(results, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for r in results:
        if r["trades"].empty:
            continue
        pnl_base, risk_base, times = _combo_base_pnl(r["trades"])
        pnl = apply_sizing(pnl_base, risk_base, policy)
        ax.plot(times, _equity_curve(pnl), linewidth=1.0, alpha=0.85,
                label=r["combo_id"])
    ax.axhline(STARTING_EQUITY, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"individual unfiltered - equity ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("equity ($)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout(); plt.show()


def plot_indiv_dd(results, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for r in results:
        if r["trades"].empty:
            continue
        pnl_base, risk_base, times = _combo_base_pnl(r["trades"])
        pnl = apply_sizing(pnl_base, risk_base, policy)
        dd = _drawdown_curve(pnl)
        t_full = np.concatenate([[times[0]], times]) if len(times) else times
        ax.plot(t_full, dd, linewidth=1.0, alpha=0.85, label=r["combo_id"])
    ax.invert_yaxis()
    ax.set_title(f"individual unfiltered - drawdown ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("drawdown (%)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout(); plt.show()


def plot_combined_equity(df, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f"combined unfiltered - {policy} (no trades)")
        fig.tight_layout(); plt.show(); return
    pnl_base = df["actual_pnl"].to_numpy(dtype=float)
    risk_base = df["dollar_risk"].to_numpy(dtype=float)
    pnl = apply_sizing(pnl_base, risk_base, policy)
    ax.plot(df["date"], _equity_curve(pnl), linewidth=1.3)
    ax.axhline(STARTING_EQUITY, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"combined unfiltered - equity ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("equity ($)"); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()


def plot_combined_dd(df, policy, bars):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f"combined unfiltered - {policy} (no trades)")
        fig.tight_layout(); plt.show(); return
    pnl_base = df["actual_pnl"].to_numpy(dtype=float)
    risk_base = df["dollar_risk"].to_numpy(dtype=float)
    pnl = apply_sizing(pnl_base, risk_base, policy)
    dd = _drawdown_curve(pnl)
    times = pd.concat([pd.Series([bars["time"].iloc[0]]),
                       pd.Series(df["date"].values)]).reset_index(drop=True)
    ax.plot(times, dd, linewidth=1.3, color="#d62728")
    ax.invert_yaxis()
    ax.set_title(f"combined unfiltered - drawdown ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("drawdown (%)"); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()


def plot_ml2_indiv_equity(s4_pnl_by_combo, bars, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for cid, entry in s4_pnl_by_combo.items():
        pnl = entry["by_policy"].get(policy, np.array([]))
        exit_bars = entry["exit_bars"]
        if len(pnl) == 0:
            continue
        times = pd.to_datetime(bars["time"].to_numpy()[exit_bars])
        ax.plot(times, _equity_curve(pnl), linewidth=1.0, alpha=0.85, label=cid)
    ax.axhline(STARTING_EQUITY, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"individual ML2-filtered - equity ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("equity ($)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout(); plt.show()


def plot_ml2_indiv_dd(s4_pnl_by_combo, bars, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for cid, entry in s4_pnl_by_combo.items():
        pnl = entry["by_policy"].get(policy, np.array([]))
        exit_bars = entry["exit_bars"]
        if len(pnl) == 0:
            continue
        times = pd.to_datetime(bars["time"].to_numpy()[exit_bars])
        dd = _drawdown_curve(pnl)
        t0 = times.min() if len(times) else pd.to_datetime(bars["time"].iloc[0])
        t_full = pd.concat([pd.Series([t0]), pd.Series(times)]).reset_index(drop=True)
        ax.plot(t_full, dd, linewidth=1.0, alpha=0.85, label=cid)
    ax.invert_yaxis()
    ax.set_title(f"individual ML2-filtered - drawdown ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("drawdown (%)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout(); plt.show()


def plot_ml2_combined_equity(df, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f"ML2 combined portfolio - {policy} (no trades)")
        fig.tight_layout(); plt.show(); return
    ax.plot(df["exit_time"], df["equity_after"], linewidth=1.3, color="#1f77b4")
    ax.axhline(STARTING_EQUITY, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"ML2 combined portfolio - equity ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("equity ($)"); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()


def plot_ml2_combined_dd(df, bars, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f"ML2 combined portfolio - {policy} (no trades)")
        fig.tight_layout(); plt.show(); return
    eq_full = np.concatenate([[STARTING_EQUITY], df["equity_after"].to_numpy()])
    peak = np.maximum.accumulate(eq_full)
    dd = (peak - eq_full) / peak * 100
    times = pd.concat([pd.Series([bars["time"].iloc[0]]),
                       pd.Series(df["exit_time"].values)]).reset_index(drop=True)
    ax.plot(times, dd, linewidth=1.3, color="#d62728")
    ax.invert_yaxis()
    ax.set_title(f"ML2 combined portfolio - drawdown ({policy})")
    ax.set_xlabel("time"); ax.set_ylabel("drawdown (%)"); ax.grid(alpha=0.3)
    fig.tight_layout(); plt.show()


# ── Monte-Carlo plot helpers ────────────────────────────────────────────────

def _mc_source(df):
    if df.empty:
        return np.array([]), np.array([])
    return (df["actual_pnl"].to_numpy(dtype=float),
            df["dollar_risk"].to_numpy(dtype=float))


def plot_mc_sims(df, policy, title_prefix, years_span, n_paths=_MC_SIM_PATHS):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f"{title_prefix} MC - equity paths ({policy}) no trades")
        fig.tight_layout(); plt.show(); return
    _, equity_paths = mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=n_paths, seed=42)
    trade_axis = np.arange(equity_paths.shape[1])
    for path in equity_paths:
        ax.plot(trade_axis, path, linewidth=0.5, alpha=0.25, color="#1f77b4")
    ax.plot(trade_axis, np.median(equity_paths, axis=0), linewidth=1.5,
            color="#b2182b", label="median path")
    ax.axhline(STARTING_EQUITY, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"{title_prefix} MC - {n_paths} equity paths ({policy})")
    ax.set_xlabel("trade #"); ax.set_ylabel("equity ($)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); plt.show()


def _hist_with_markers(ax, values, fmt):
    ax.hist(values, bins=60, color="#6a8cbb", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.6)
    for pct, colour, ls in [(2.5, "#b2182b", ":"), (50, "#1a9850", "--"),
                            (97.5, "#b2182b", ":")]:
        v = np.percentile(values, pct)
        ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                   label=f"p{pct:g}={fmt(v)}")


def plot_mc_pnl(df, policy, title_prefix, years_span, n_sims=10_000):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f"{title_prefix} MC - final PnL ({policy}) no trades")
        fig.tight_layout(); plt.show(); return
    _, equity_paths = mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=n_sims, seed=42)
    final_pnl = equity_paths[:, -1] - STARTING_EQUITY
    _hist_with_markers(ax, final_pnl, lambda v: f"${v:,.0f}")
    ax.set_title(f"{title_prefix} MC - final PnL ({policy})")
    ax.set_xlabel("final PnL ($)"); ax.set_ylabel("freq")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); plt.show()


def plot_mc_sharpe(df, policy, title_prefix, years_span, n_sims=10_000):
    """Sharpe basis: $-PnL, annualized by sqrt(trades_per_year)."""
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f"{title_prefix} MC - Sharpe ({policy}) no trades")
        fig.tight_layout(); plt.show(); return
    samples_pnl, _ = mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=n_sims, seed=42)
    n = samples_pnl.shape[1]
    tpy = n / years_span if years_span > 0 else 0.0
    sim_vals = samples_pnl
    mu = sim_vals.mean(axis=1)
    sig = sim_vals.std(axis=1, ddof=1) if n > 1 else np.zeros(sim_vals.shape[0])
    sharpe = np.where(sig > 0, (mu / sig) * np.sqrt(tpy), 0.0)
    ax.hist(sharpe, bins=60, color="#d48c6a", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.6)
    for pct, colour, ls in [(2.5, "#b2182b", ":"), (50, "#1a9850", "--"),
                            (97.5, "#b2182b", ":")]:
        v = np.percentile(sharpe, pct)
        ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                   label=f"p{pct:g}={v:.2f}")
    ax.set_title(f"{title_prefix} MC - annualized Sharpe ({policy})")
    ax.set_xlabel("Sharpe"); ax.set_ylabel("freq")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); plt.show()


def plot_mc_dd(df, policy, title_prefix, years_span, n_sims=10_000):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f"{title_prefix} MC - max drawdown ({policy}) no trades")
        fig.tight_layout(); plt.show(); return
    _, equity_paths = mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=n_sims, seed=42)
    peak = np.maximum.accumulate(equity_paths, axis=1)
    dd_pct = np.nan_to_num((peak - equity_paths) / peak, nan=0.0).max(axis=1) * 100
    ax.hist(dd_pct, bins=60, color="#c49a6c", alpha=0.85, edgecolor="white")
    for pct, colour, ls in [(50, "#1a9850", "--"), (90, "#fc8d59", ":"),
                            (95, "#b2182b", ":"), (99, "#67000d", ":")]:
        v = np.percentile(dd_pct, pct)
        ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                   label=f"p{pct:g}={v:.2f}%")
    ax.set_title(f"{title_prefix} MC - max drawdown ({policy})")
    ax.set_xlabel("max drawdown (%)"); ax.set_ylabel("freq")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); plt.show()
