"""Build the 6-section top-performance notebook.

Section layout:
  1) individual              (unfiltered, raw composed_strategy_runner)
  2) combined                (unfiltered portfolio aggregate)
  3) Monte Carlo on combined (unfiltered)
  4) individual with ML2     (V3 booster + calibrator filter)
  5) combined with ML2       (event-driven portfolio of V3-filtered trades)
  6) Monte Carlo on combined ML2

Two sizing policies are compared:
  - fixed_dollars_500: risk $500 on every trade, forever.
  - pct5_compound:     risk 5% of *current* equity on every trade.
                       Starts at $2,500 (=5% of $50k) and compounds trade-by-trade.

Every plot is its own cell, and every plot cell is rendered once per policy
(separate cells — no side-by-side). Sharpe is annualized by sqrt(trades_per_year)
from the test-partition span, applied consistently across every section.

The full trade log is produced separately by `build_trade_log_xlsx.py` as an
Excel workbook (`evaluation/top_trade_log.xlsx`).

Execute the notebook in-place with nbclient afterwards.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent.parent
EVAL = REPO / "evaluation"

NB_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.12",
    },
}


def _md(src: str, cid: str) -> nbf.NotebookNode:
    c = nbf.v4.new_markdown_cell(src)
    c["id"] = cid
    return c


def _code(src: str, cid: str) -> nbf.NotebookNode:
    c = nbf.v4.new_code_cell(src)
    c["id"] = cid
    return c


# ─── Shared setup ────────────────────────────────────────────────────────────

SETUP_IMPORTS = """import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless; we render explicitly via IPython.display
import matplotlib.pyplot as plt
from IPython.display import display

def show(fig):
    '''Render a matplotlib Figure as an image output cell, then close it.
    Used in place of plt.show() because nbclient + inline backend is unreliable
    across package combos; display(fig) always produces image/png output.'''
    display(fig); plt.close(fig)

REPO = Path.cwd().resolve()
while not (REPO / 'src').exists() and REPO.parent != REPO:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from scripts.evaluation.composed_strategy_runner import run_strategy, load_test_bars

# Import the V3 phase-2 eval module (provides build_combo_trades_test,
# solo_metrics, simulate_portfolio). We load it by file path because it
# sits under scripts/evaluation/ which isn't a proper package.
_spec = importlib.util.spec_from_file_location(
    '_v3eval', REPO / 'scripts/evaluation/final_holdout_eval_v3_c1_fixed500.py'
)
v3eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3eval)

TOP_STRATEGIES_PATH = REPO / 'evaluation' / 'top_strategies.json'
STARTING_EQUITY = 50_000.0
RISK_FRAC = 0.05  # 5% of current equity per trade under pct5_compound
POLICIES = ['fixed_dollars_500', 'pct5_compound']


def apply_sizing(pnl_base, risk_base, policy, equity0=STARTING_EQUITY):
    '''Return per-trade dollar PnL (ordered) under the named sizing policy.

    Inputs are the $500-fixed baseline series:
      - pnl_base[t]:  realized $PnL for trade t at fixed-$500 sizing
      - risk_base[t]: realized $ at risk for trade t at fixed-$500 sizing
                     (= contracts_500 * sl_pts * DOLLARS_PER_POINT)

    Define the policy-invariant r-multiple r_t = pnl_base/risk_base. Then:
      - fixed_dollars_500: pnl_t = pnl_base[t] (unchanged)
      - pct5_compound:    eq_{t+1} = eq_t * (1 + RISK_FRAC * r_t), starting
                          eq_0 = equity0; pnl_t = eq_{t+1} - eq_t.
    '''
    pnl = np.asarray(pnl_base, dtype=float)
    risk = np.asarray(risk_base, dtype=float)
    if policy == 'fixed_dollars_500':
        return pnl
    if policy != 'pct5_compound':
        raise ValueError(f'unknown policy: {policy}')
    r = np.where(risk > 0, pnl / risk, 0.0)
    equity = equity0 * np.cumprod(1.0 + RISK_FRAC * r)
    return np.diff(np.concatenate([[equity0], equity]))


def metrics_from_pnl(pnl, years_span, policy='fixed_dollars_500', r=None,
                     start_equity=STARTING_EQUITY):
    '''Headline metrics for a per-trade dollar-PnL series.

    For policy='fixed_dollars_500', Sharpe is annualized on per-trade $-PnL:
      sharpe = (mean/std) * sqrt(trades_per_year).
    For policy='pct5_compound', Sharpe is computed on log-returns of the
    compounded equity curve: log_ret = log1p(RISK_FRAC * r), where r is the
    per-trade r-multiple (pnl_base / risk_base). This is scale-invariant and
    the correct annualization under compounding. If `r` is not provided under
    pct5_compound, falls back to $-PnL Sharpe (same formula as fixed).

    PnL, total return, and drawdown are always computed on the supplied
    policy-adjusted `pnl` series.
    '''
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
    if policy == 'pct5_compound' and r is not None and len(r) == n:
        log_ret = np.log1p(RISK_FRAC * np.asarray(r, dtype=float))
        std_s = log_ret.std(ddof=1) if n > 1 else 0.0
        sharpe = (float(log_ret.mean() / std_s * np.sqrt(tpy))
                  if std_s > 0 and tpy > 0 else 0.0)
    else:
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


def _mc_policy_samples(pnl_base, risk_base, policy, n_sims=10_000, seed=42,
                       start_equity=STARTING_EQUITY):
    '''Bootstrap resample the trade-order and return per-sim per-trade $PnL
    under `policy`. For fixed_dollars_500 we resample pnl_base directly; for
    pct5_compound we resample r-multiples and compound from start_equity.

    Returns: samples_pnl (n_sims, n), equity_paths (n_sims, n+1) where path
    column 0 is start_equity.
    '''
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl_base, dtype=float)
    risk = np.asarray(risk_base, dtype=float)
    n = len(pnl)
    if n == 0:
        return np.empty((0, 0)), np.empty((0, 0))
    idx = rng.integers(0, n, size=(n_sims, n))
    if policy == 'fixed_dollars_500':
        samples_pnl = pnl[idx]
    elif policy == 'pct5_compound':
        r = np.where(risk > 0, pnl / risk, 0.0)
        r_samples = r[idx]
        growth = 1.0 + RISK_FRAC * r_samples
        equity = start_equity * np.cumprod(growth, axis=1)
        samples_pnl = np.diff(np.concatenate(
            [np.full((n_sims, 1), start_equity), equity], axis=1))
    else:
        raise ValueError(f'unknown policy: {policy}')
    equity_paths = np.concatenate(
        [np.full((n_sims, 1), start_equity),
         start_equity + np.cumsum(samples_pnl, axis=1)], axis=1)
    return samples_pnl, equity_paths


def monte_carlo(pnl_base, risk_base, policy, years_span,
                start_equity=STARTING_EQUITY, n_sims=10_000, seed=42):
    '''IID bootstrap MC on a trade sequence under `policy`. Returns DD
    percentiles + VaR/CVaR + risk-of-ruin (>=50% DD) + annualized Sharpe CI.

    For fixed_dollars_500: resample dollar-PnL directly; Sharpe is annualized
    on per-sim per-trade $-PnL.
    For pct5_compound: resample r-multiples, compound with RISK_FRAC; Sharpe
    is annualized on per-sim per-trade log-returns (log1p(RISK_FRAC*r)),
    which is scale-invariant under compounding.
    '''
    pnl = np.asarray(pnl_base, dtype=float)
    risk = np.asarray(risk_base, dtype=float)
    n = len(pnl)
    if n == 0:
        return {'n_sims': n_sims, 'n_trades': 0, 'note': 'empty'}
    samples_pnl, equity_paths = _mc_policy_samples(
        pnl, risk, policy, n_sims=n_sims, seed=seed,
        start_equity=start_equity)
    peak = np.maximum.accumulate(equity_paths, axis=1)
    dd_pct = np.nan_to_num((peak - equity_paths) / peak, nan=0.0).max(axis=1) * 100
    tpy = n / years_span if years_span > 0 else 0.0
    rng = np.random.default_rng(seed)
    if policy == 'pct5_compound':
        r_full = np.where(risk > 0, pnl / risk, 0.0)
        idx = rng.integers(0, n, size=(n_sims, n))
        log_ret = np.log1p(RISK_FRAC * r_full[idx])
        mu = log_ret.mean(axis=1)
        sig = log_ret.std(axis=1, ddof=1) if n > 1 else np.zeros(n_sims)
    else:
        mu = samples_pnl.mean(axis=1)
        sig = samples_pnl.std(axis=1, ddof=1) if n > 1 else np.zeros(n_sims)
    sharpe_boot = np.where(sig > 0, (mu / sig) * np.sqrt(tpy), 0.0)
    # Trade-level VaR under policy: take the p5 of all resampled per-trade PnLs.
    var5 = float(np.percentile(samples_pnl.reshape(-1), 5))
    tail = samples_pnl[samples_pnl <= var5]
    cvar = float(tail.mean()) if tail.size else var5
    wins = (samples_pnl > 0).mean(axis=1)
    wr_ci = (float(np.percentile(wins, 2.5)), float(np.percentile(wins, 97.5)))
    sharpe_ci = (float(np.percentile(sharpe_boot, 2.5)),
                 float(np.percentile(sharpe_boot, 97.5)))
    return {
        'n_sims': n_sims, 'n_trades': int(n),
        'win_rate': round(float(wins.mean()), 4),
        'wr_ci_95': (round(wr_ci[0], 4), round(wr_ci[1], 4)),
        'sharpe_p50': round(float(np.percentile(sharpe_boot, 50)), 4),
        'sharpe_ci_95': (round(sharpe_ci[0], 4), round(sharpe_ci[1], 4)),
        'sharpe_pos_prob': round(float((sharpe_boot > 0).mean()), 4),
        'dd_p50_pct': round(float(np.percentile(dd_pct, 50)), 2),
        'dd_p90_pct': round(float(np.percentile(dd_pct, 90)), 2),
        'dd_p95_pct': round(float(np.percentile(dd_pct, 95)), 2),
        'dd_p99_pct': round(float(np.percentile(dd_pct, 99)), 2),
        'dd_worst_pct': round(float(dd_pct.max()), 2),
        'var_5pct_trade': round(var5, 2),
        'cvar_5pct_trade': round(cvar, 2),
        'risk_of_ruin_50pct_dd': round(float((dd_pct >= 50.0).mean()), 4),
    }
"""


RUN_UNFILTERED = """payload = json.loads(TOP_STRATEGIES_PATH.read_text())
strategies = payload['top']
print(f'Loaded {len(strategies)} strategies (min_trades={payload["min_trades"]}, '
      f'eligible={payload["pool_sizes"]["eligible_combos"]:,}/'
      f'{payload["pool_sizes"]["total_combos"]:,})')
bars = load_test_bars()
print(f'Test partition: {len(bars):,} bars  {bars["time"].iloc[0]} -> {bars["time"].iloc[-1]}')

YEARS_SPAN = (pd.to_datetime(bars['time'].iloc[-1]) -
              pd.to_datetime(bars['time'].iloc[0])).total_seconds() / (365.25 * 86400)
print(f'Years span: {YEARS_SPAN:.3f}  (used to annualize Sharpe)')

print('\\nRunning unfiltered composed_strategy_runner for each combo...')
results_raw = []
for s in strategies:
    print(f'  {s["global_combo_id"]}...', flush=True)
    results_raw.append(run_strategy(s, bars=bars))
print('Done (unfiltered).')"""


RUN_ML2 = """import lightgbm as lgb, pickle
print('Loading V3 booster + calibrators...')
_v3inf = v3eval.v3inf
booster = lgb.Booster(model_file=str(_v3inf.V3_BOOSTER))
simple_cals = _v3inf._load_calibrators()
two_stage = _v3inf._load_per_combo_calibrators()

combo_ids = [s['global_combo_id'] for s in strategies]

# Cache ML2 per-combo trades. combos_ml2 is a deterministic function of
# (booster, calibrators, build_combo_trades_test logic, combo params, test
# bars). Key includes mtimes for every dependency that can change combos_ml2
# without changing the combo_ids, so a silent stale-cache hit is impossible:
#   - V3 booster file
#   - Pooled per-R:R isotonic calibrators
#   - Per-combo two-stage calibrators
#   - The final_holdout_eval_v3_c1_fixed500 module itself (build_combo_trades_test)
#   - Test-partition end time (bars grow via update_bars_yfinance.py)
_ML2_CACHE_VERSION = 'v2'
_ML2_CACHE = REPO / 'evaluation' / '_ml2_cache.pkl'
_cache_key = (
    _ML2_CACHE_VERSION,
    Path(_v3inf.V3_BOOSTER).stat().st_mtime_ns,
    Path(_v3inf.V3_CALIBRATORS).stat().st_mtime_ns,
    Path(_v3inf.V3_PER_COMBO_CALIBRATORS).stat().st_mtime_ns,
    Path(v3eval.__file__).stat().st_mtime_ns,
    str(bars['time'].iloc[-1]),
    tuple(sorted(combo_ids)),
)

combos_ml2 = None
if _ML2_CACHE.exists():
    try:
        with open(_ML2_CACHE, 'rb') as f:
            blob = pickle.load(f)
        if blob.get('key') == _cache_key:
            combos_ml2 = blob['combos_ml2']
            print(f'Loaded combos_ml2 from cache: {_ML2_CACHE.name} '
                  f'({len(combos_ml2)} combos).')
        else:
            print(f'Cache {_ML2_CACHE.name} is stale; rebuilding.')
    except Exception as e:
        print(f'Cache read failed ({e!r}); rebuilding.')

if combos_ml2 is None:
    print('\\nBuilding V3-filtered trades per combo (may take a few minutes)...')
    combos_ml2 = []
    for gcid in combo_ids:
        print(f'  {gcid}...', flush=True)
        try:
            c = v3eval.build_combo_trades_test(gcid, booster, simple_cals, two_stage)
            print(f'    n_trades={c.get("n_trades", 0)}  '
                  f'rr={c.get("rr", float("nan")):.2f}')
        except Exception as e:
            c = {'combo_id': gcid, 'error': str(e)}
            print(f'    ERROR: {e}')
        combos_ml2.append(c)
    _ML2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_ML2_CACHE, 'wb') as f:
        pickle.dump({'key': _cache_key, 'combos_ml2': combos_ml2}, f)
    print(f'Wrote cache -> {_ML2_CACHE.name}')
print('Done (ML2).')"""


# ─── Plot helpers ────────────────────────────────────────────────────────────
#
# Every plot helper takes a `policy` string (one of POLICIES) and internally
# calls apply_sizing / _mc_policy_samples. Each helper returns no value; it
# builds a single Figure and renders it via show(fig).

PLOT_HELPERS = """_FIG_W, _FIG_H = 9, 4.5
_MC_SIM_PATHS = 200  # number of overlaid equity paths in plot_mc_sims


def _combo_base_pnl(trades):
    '''Per-trade ($500-fixed baseline) PnL and $-at-risk from the unfiltered
    composed_strategy_runner trade log. Trades are assumed already sorted by
    date upstream; we sort defensively here too.'''
    if trades.empty:
        return np.array([]), np.array([]), np.array([], dtype='datetime64[ns]')
    t = trades.sort_values('date', kind='mergesort')
    pnl = t['actual_pnl'].to_numpy(dtype=float)
    risk = t['dollar_risk'].to_numpy(dtype=float)
    times = t['date'].to_numpy()
    return pnl, risk, times


def _equity_curve(pnl_policy, start_equity=STARTING_EQUITY):
    return start_equity + np.cumsum(pnl_policy)


def _drawdown_curve(pnl_policy, start_equity=STARTING_EQUITY):
    eq_full = np.concatenate([[start_equity], start_equity + np.cumsum(pnl_policy)])
    peak = np.maximum.accumulate(eq_full)
    return (peak - eq_full) / peak * 100


def plot_indiv_equity(results, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for r in results:
        if r['trades'].empty:
            continue
        pnl_base, risk_base, times = _combo_base_pnl(r['trades'])
        pnl = apply_sizing(pnl_base, risk_base, policy)
        ax.plot(times, _equity_curve(pnl), linewidth=1.0, alpha=0.85,
                label=r['combo_id'])
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6)
    ax.set_title(f'individual unfiltered - equity ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    fig.tight_layout(); show(fig)


def plot_indiv_dd(results, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for r in results:
        if r['trades'].empty:
            continue
        pnl_base, risk_base, times = _combo_base_pnl(r['trades'])
        pnl = apply_sizing(pnl_base, risk_base, policy)
        dd = _drawdown_curve(pnl)
        t_full = np.concatenate([[times[0]], times]) if len(times) else times
        ax.plot(t_full, dd, linewidth=1.0, alpha=0.85, label=r['combo_id'])
    ax.invert_yaxis()
    ax.set_title(f'individual unfiltered - drawdown ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    fig.tight_layout(); show(fig)


def plot_combined_equity(df, policy):
    '''df: combined_raw with actual_pnl + dollar_risk + date columns.'''
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f'combined unfiltered - {policy} (no trades)')
        fig.tight_layout(); show(fig); return
    pnl_base = df['actual_pnl'].to_numpy(dtype=float)
    risk_base = df['dollar_risk'].to_numpy(dtype=float)
    pnl = apply_sizing(pnl_base, risk_base, policy)
    ax.plot(df['date'], _equity_curve(pnl), linewidth=1.3)
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6)
    ax.set_title(f'combined unfiltered - equity ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
    fig.tight_layout(); show(fig)


def plot_combined_dd(df, policy, bars):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f'combined unfiltered - {policy} (no trades)')
        fig.tight_layout(); show(fig); return
    pnl_base = df['actual_pnl'].to_numpy(dtype=float)
    risk_base = df['dollar_risk'].to_numpy(dtype=float)
    pnl = apply_sizing(pnl_base, risk_base, policy)
    dd = _drawdown_curve(pnl)
    times = pd.concat([pd.Series([bars['time'].iloc[0]]),
                       pd.Series(df['date'].values)]).reset_index(drop=True)
    ax.plot(times, dd, linewidth=1.3, color='#d62728')
    ax.invert_yaxis()
    ax.set_title(f'combined unfiltered - drawdown ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
    fig.tight_layout(); show(fig)


def plot_ml2_indiv_equity(s4_pnl_by_combo, bars, policy):
    '''s4_pnl_by_combo[cid][policy] = (pnl_policy, exit_bars).'''
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for cid, by_pol in s4_pnl_by_combo.items():
        pnl, exit_bars = by_pol.get(policy, (np.array([]), np.array([])))
        if len(pnl) == 0:
            continue
        times = pd.to_datetime(bars['time'].to_numpy()[exit_bars])
        ax.plot(times, _equity_curve(pnl), linewidth=1.0, alpha=0.85, label=cid)
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6)
    ax.set_title(f'individual ML2-filtered - equity ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    fig.tight_layout(); show(fig)


def plot_ml2_indiv_dd(s4_pnl_by_combo, bars, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    for cid, by_pol in s4_pnl_by_combo.items():
        pnl, exit_bars = by_pol.get(policy, (np.array([]), np.array([])))
        if len(pnl) == 0:
            continue
        times = pd.to_datetime(bars['time'].to_numpy()[exit_bars])
        dd = _drawdown_curve(pnl)
        t0 = times.min() if len(times) else pd.to_datetime(bars['time'].iloc[0])
        t_full = pd.concat([pd.Series([t0]), pd.Series(times)]).reset_index(drop=True)
        ax.plot(t_full, dd, linewidth=1.0, alpha=0.85, label=cid)
    ax.invert_yaxis()
    ax.set_title(f'individual ML2-filtered - drawdown ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    fig.tight_layout(); show(fig)


def plot_ml2_combined_equity(df, policy):
    '''df: ml2_portfolio[policy] with exit_time + equity_after columns.'''
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f'ML2 combined portfolio - {policy} (no trades)')
        fig.tight_layout(); show(fig); return
    ax.plot(df['exit_time'], df['equity_after'], linewidth=1.3, color='#1f77b4')
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6)
    ax.set_title(f'ML2 combined portfolio - equity ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
    fig.tight_layout(); show(fig)


def plot_ml2_combined_dd(df, bars, policy):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    if df.empty:
        ax.set_title(f'ML2 combined portfolio - {policy} (no trades)')
        fig.tight_layout(); show(fig); return
    eq_full = np.concatenate([[STARTING_EQUITY], df['equity_after'].to_numpy()])
    peak = np.maximum.accumulate(eq_full)
    dd = (peak - eq_full) / peak * 100
    times = pd.concat([pd.Series([bars['time'].iloc[0]]),
                       pd.Series(df['exit_time'].values)]).reset_index(drop=True)
    ax.plot(times, dd, linewidth=1.3, color='#d62728')
    ax.invert_yaxis()
    ax.set_title(f'ML2 combined portfolio - drawdown ({policy})')
    ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
    fig.tight_layout(); show(fig)


# ── Monte-Carlo plot helpers ────────────────────────────────────────────────
# All accept a DataFrame (`df`) with columns `actual_pnl` + `dollar_risk` as
# the trade source, plus a `policy` and a `title_prefix` for the title text.

def _mc_source(df):
    if df.empty:
        return np.array([]), np.array([])
    return (df['actual_pnl'].to_numpy(dtype=float),
            df['dollar_risk'].to_numpy(dtype=float))


def plot_mc_sims(df, policy, title_prefix, years_span, n_paths=_MC_SIM_PATHS):
    '''Overlay of ~n_paths bootstrapped equity trajectories under `policy`.'''
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f'{title_prefix} MC - equity paths ({policy}) no trades')
        fig.tight_layout(); show(fig); return
    _, equity_paths = _mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=n_paths, seed=42)
    trade_axis = np.arange(equity_paths.shape[1])
    for path in equity_paths:
        ax.plot(trade_axis, path, linewidth=0.5, alpha=0.25, color='#1f77b4')
    ax.plot(trade_axis, np.median(equity_paths, axis=0), linewidth=1.5,
            color='#b2182b', label='median path')
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6)
    ax.set_title(f'{title_prefix} MC - {n_paths} equity paths ({policy})')
    ax.set_xlabel('trade #'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout(); show(fig)


def _hist_with_markers(ax, values, fmt):
    ax.hist(values, bins=60, color='#6a8cbb', alpha=0.85, edgecolor='white')
    ax.axvline(0, color='k', linewidth=0.8, alpha=0.6)
    for pct, colour, ls in [(2.5, '#b2182b', ':'), (50, '#1a9850', '--'),
                            (97.5, '#b2182b', ':')]:
        v = np.percentile(values, pct)
        ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                   label=f'p{pct:g}={fmt(v)}')


def plot_mc_pnl(df, policy, title_prefix, years_span):
    '''Histogram of final-equity-minus-start under `policy`.'''
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f'{title_prefix} MC - final PnL ({policy}) no trades')
        fig.tight_layout(); show(fig); return
    _, equity_paths = _mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=10_000, seed=42)
    final_pnl = equity_paths[:, -1] - STARTING_EQUITY
    _hist_with_markers(ax, final_pnl, lambda v: f'${v:,.0f}')
    ax.set_title(f'{title_prefix} MC - final PnL ({policy})')
    ax.set_xlabel('final PnL ($)'); ax.set_ylabel('freq')
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout(); show(fig)


def plot_mc_sharpe(df, policy, title_prefix, years_span):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f'{title_prefix} MC - Sharpe ({policy}) no trades')
        fig.tight_layout(); show(fig); return
    samples_pnl, _ = _mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=10_000, seed=42)
    n = samples_pnl.shape[1]
    tpy = n / years_span if years_span > 0 else 0.0
    mu = samples_pnl.mean(axis=1)
    sig = samples_pnl.std(axis=1, ddof=1) if n > 1 else np.zeros(samples_pnl.shape[0])
    sharpe = np.where(sig > 0, (mu / sig) * np.sqrt(tpy), 0.0)
    ax.hist(sharpe, bins=60, color='#d48c6a', alpha=0.85, edgecolor='white')
    ax.axvline(0, color='k', linewidth=0.8, alpha=0.6)
    for pct, colour, ls in [(2.5, '#b2182b', ':'), (50, '#1a9850', '--'),
                            (97.5, '#b2182b', ':')]:
        v = np.percentile(sharpe, pct)
        ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                   label=f'p{pct:g}={v:.2f}')
    ax.set_title(f'{title_prefix} MC - annualized Sharpe ({policy})')
    ax.set_xlabel('Sharpe'); ax.set_ylabel('freq')
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout(); show(fig)


def plot_mc_dd(df, policy, title_prefix, years_span):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    pnl_base, risk_base = _mc_source(df)
    if len(pnl_base) == 0:
        ax.set_title(f'{title_prefix} MC - max drawdown ({policy}) no trades')
        fig.tight_layout(); show(fig); return
    _, equity_paths = _mc_policy_samples(
        pnl_base, risk_base, policy, n_sims=10_000, seed=42)
    peak = np.maximum.accumulate(equity_paths, axis=1)
    dd_pct = np.nan_to_num((peak - equity_paths) / peak, nan=0.0).max(axis=1) * 100
    ax.hist(dd_pct, bins=60, color='#c49a6c', alpha=0.85, edgecolor='white')
    for pct, colour, ls in [(50, '#1a9850', '--'), (90, '#fc8d59', ':'),
                            (95, '#b2182b', ':'), (99, '#67000d', ':')]:
        v = np.percentile(dd_pct, pct)
        ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                   label=f'p{pct:g}={v:.2f}%')
    ax.set_title(f'{title_prefix} MC - max drawdown ({policy})')
    ax.set_xlabel('max drawdown (%)'); ax.set_ylabel('freq')
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout(); show(fig)"""


# ─── Section 1 / 2 / 3 (unfiltered) ──────────────────────────────────────────

S1_PERF = """rows = []
for r in results_raw:
    if r['trades'].empty:
        for policy in POLICIES:
            rows.append({'combo_id': r['combo_id'], 'policy': policy,
                         **metrics_from_pnl(np.array([]), YEARS_SPAN,
                                             policy=policy)})
        continue
    t = r['trades'].sort_values('date', kind='mergesort')
    pnl_base = t['actual_pnl'].to_numpy(dtype=float)
    risk_base = t['dollar_risk'].to_numpy(dtype=float)
    r_mult = np.where(risk_base > 0, pnl_base / risk_base, 0.0)
    for policy in POLICIES:
        pnl = apply_sizing(pnl_base, risk_base, policy)
        rows.append({'combo_id': r['combo_id'], 'policy': policy,
                     **metrics_from_pnl(pnl, YEARS_SPAN,
                                         policy=policy, r=r_mult)})
perf1 = pd.DataFrame(rows)
perf1"""


S2_COMBINED_BUILD = """frames = []
for r in results_raw:
    if r['trades'].empty:
        continue
    t = r['trades'][['date', 'actual_pnl', 'dollar_risk']].copy()
    t.insert(0, 'combo_id', r['combo_id'])
    frames.append(t)
combined_raw = (pd.concat(frames, ignore_index=True)
                .sort_values('date', kind='mergesort')
                .reset_index(drop=True)
                if frames else pd.DataFrame(
                    columns=['combo_id', 'date', 'actual_pnl', 'dollar_risk']))
print(f'Combined unfiltered trades: {len(combined_raw):,}')
rows = []
for policy in POLICIES:
    if combined_raw.empty:
        rows.append({'policy': policy,
                     **metrics_from_pnl(np.array([]), YEARS_SPAN,
                                         policy=policy)})
        continue
    pnl_base = combined_raw['actual_pnl'].to_numpy(dtype=float)
    risk_base = combined_raw['dollar_risk'].to_numpy(dtype=float)
    r_mult = np.where(risk_base > 0, pnl_base / risk_base, 0.0)
    pnl = apply_sizing(pnl_base, risk_base, policy)
    rows.append({'policy': policy,
                 **metrics_from_pnl(pnl, YEARS_SPAN,
                                     policy=policy, r=r_mult)})
combined_table_raw = pd.DataFrame(rows)
combined_table_raw"""


S3_MC = """rows = []
for policy in POLICIES:
    if combined_raw.empty:
        rows.append({'policy': policy, 'n_trades': 0})
        continue
    pnl_base = combined_raw['actual_pnl'].to_numpy(dtype=float)
    risk_base = combined_raw['dollar_risk'].to_numpy(dtype=float)
    rows.append({'policy': policy,
                 **monte_carlo(pnl_base, risk_base, policy, YEARS_SPAN)})
mc_raw = pd.DataFrame(rows)
mc_raw"""


# ─── Section 4 / 5 / 6 (ML2) ─────────────────────────────────────────────────

S4_PERF = """# Per-combo ML2-filtered PnL under fixed-$500 baseline; then pct5_compound
# applies apply_sizing() on top of that baseline so both policies share the
# same integer-contract rounding at entry.

def _combo_pnl_base(c):
    '''Return (pnl_base, risk_base, exit_bars) at $500 fixed sizing.

    For the $500-fixed policy: contracts = int(500 // (sl_pts * $/pt)); if
    contracts==0 the trade is skipped. pnl_base = pts * contracts * $/pt,
    risk_base = sl_pts * contracts * $/pt.
    '''
    if c.get('error') or c.get('n_trades', 0) == 0:
        return (np.array([]), np.array([]), np.array([], dtype=int))
    sl = np.asarray(c['sl_pts'], dtype=float)
    pts = np.asarray(c['pnl_pts'], dtype=float)
    contracts = (500.0 // (sl * v3eval.DOLLARS_PER_POINT)).astype(int)
    mask = contracts > 0
    pnl_base = (pts[mask] * contracts[mask] * v3eval.DOLLARS_PER_POINT).astype(float)
    risk_base = (sl[mask] * contracts[mask] * v3eval.DOLLARS_PER_POINT).astype(float)
    exit_bars = np.asarray(c['exit_bar'], dtype=int)[mask]
    return pnl_base, risk_base, exit_bars


s4_pnl_by_combo = {}  # combo_id -> {policy -> (pnl, exit_bars)}
rows = []
for c in combos_ml2:
    cid = c.get('combo_id')
    s4_pnl_by_combo[cid] = {}
    pnl_base, risk_base, exit_bars = _combo_pnl_base(c)
    if len(pnl_base) == 0:
        for policy in POLICIES:
            rows.append({'combo_id': cid, 'policy': policy,
                         **metrics_from_pnl(np.array([]), YEARS_SPAN,
                                             policy=policy)})
            s4_pnl_by_combo[cid][policy] = (np.array([]), np.array([], dtype=int))
        continue
    r_mult = np.where(risk_base > 0, pnl_base / risk_base, 0.0)
    for policy in POLICIES:
        pnl = apply_sizing(pnl_base, risk_base, policy)
        s4_pnl_by_combo[cid][policy] = (pnl, exit_bars)
        rows.append({'combo_id': cid, 'policy': policy,
                     **metrics_from_pnl(pnl, YEARS_SPAN,
                                         policy=policy, r=r_mult)})
perf4 = pd.DataFrame(rows)
perf4"""


S5_PORTFOLIO_BUILD = """# Event-driven portfolio simulator per policy. For fixed_dollars_500 the
# risked-$ per trade is always $500 (contracts = int(500 // (sl*$/pt))). For
# pct5_compound, contracts at each entry uses the live equity at that bar:
# contracts = int((equity * RISK_FRAC) // (sl*$/pt)). A trade is skipped if
# contracts <= 0 under the active policy.
#
# Each resulting DataFrame carries both `actual_pnl` (realized $) and
# `dollar_risk` (risk at entry under the active policy) so downstream MC
# helpers work uniformly via apply_sizing / _mc_policy_samples.

def _sim_portfolio(combos, policy):
    events = []
    for ci, c in enumerate(combos):
        if c.get('error') or c.get('n_trades', 0) == 0:
            continue
        for ti in range(c['n_trades']):
            events.append((int(c['entry_bar'][ti]), 0, ci, ti))
            events.append((int(c['exit_bar'][ti]), 1, ci, ti))
    events.sort(key=lambda e: (e[0], -e[1]))
    equity = STARTING_EQUITY
    realized = 0.0
    open_pos = {}  # (ci, ti) -> (contracts, risk_dollars)
    trade_rows = []
    for bar, kind, ci, ti in events:
        c = combos[ci]
        sl = float(c['sl_pts'][ti])
        if kind == 0:
            if policy == 'fixed_dollars_500':
                budget = 500.0
            elif policy == 'pct5_compound':
                budget = equity * RISK_FRAC
            else:
                raise ValueError(f'unknown policy: {policy}')
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
            pnl = c['pnl_pts'][ti] * contracts * v3eval.DOLLARS_PER_POINT
            realized += pnl
            equity = STARTING_EQUITY + realized
            exit_time = (pd.to_datetime(bars['time'].iloc[bar])
                         if bar < len(bars) else pd.NaT)
            trade_rows.append({
                'combo_id': c['combo_id'],
                'exit_time': exit_time,
                'actual_pnl': pnl,
                'dollar_risk': risk_dollars,
                'equity_after': equity,
            })
    return pd.DataFrame(trade_rows)


ml2_portfolio = {policy: _sim_portfolio(combos_ml2, policy) for policy in POLICIES}

rows = []
for policy in POLICIES:
    df = ml2_portfolio[policy]
    if df.empty:
        rows.append({'policy': policy, 'n_trades': 0}); continue
    pnl_p = df['actual_pnl'].to_numpy(dtype=float)
    risk_p = df['dollar_risk'].to_numpy(dtype=float)
    # r-multiple is sizing-invariant (pnl/risk scales together under any
    # sizing policy), so log-return Sharpe under compounding is well-defined.
    r_p = np.where(risk_p > 0, pnl_p / risk_p, 0.0)
    rows.append({'policy': policy,
                 **metrics_from_pnl(pnl_p, YEARS_SPAN,
                                     policy=policy, r=r_p)})
combined_table_ml2 = pd.DataFrame(rows)
combined_table_ml2"""


S6_MC = """rows = []
for policy in POLICIES:
    df = ml2_portfolio[policy]
    if df.empty:
        rows.append({'policy': policy, 'n_trades': 0}); continue
    pnl_p = df['actual_pnl'].to_numpy(dtype=float)
    risk_p = df['dollar_risk'].to_numpy(dtype=float)
    # pnl_p and risk_p are both sized under `policy`, so the ratio r=pnl/risk
    # is the sizing-invariant r-multiple (identical for fixed and pct5 at the
    # same trade). That means:
    #   policy='fixed_dollars_500' -> bootstrap the $-PnLs directly.
    #   policy='pct5_compound'     -> bootstrap r and re-compound from start
    #                                 equity; no double-compounding, since the
    #                                 input r IS sizing-invariant.
    # Both branches mirror Section 3's semantics exactly.
    rows.append({'policy': policy,
                 **monte_carlo(pnl_p, risk_p, policy, YEARS_SPAN)})
mc_ml2 = pd.DataFrame(rows)
mc_ml2"""


# ─── Cell-source constants for individual plot cells ────────────────────────

S1_EQ_500   = "plot_indiv_equity(results_raw, 'fixed_dollars_500')"
S1_EQ_PCT5  = "plot_indiv_equity(results_raw, 'pct5_compound')"
S1_DD_500   = "plot_indiv_dd(results_raw, 'fixed_dollars_500')"
S1_DD_PCT5  = "plot_indiv_dd(results_raw, 'pct5_compound')"

S2_EQ_500   = "plot_combined_equity(combined_raw, 'fixed_dollars_500')"
S2_EQ_PCT5  = "plot_combined_equity(combined_raw, 'pct5_compound')"
S2_DD_500   = "plot_combined_dd(combined_raw, 'fixed_dollars_500', bars)"
S2_DD_PCT5  = "plot_combined_dd(combined_raw, 'pct5_compound', bars)"

S3_SIMS_500   = "plot_mc_sims(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"
S3_SIMS_PCT5  = "plot_mc_sims(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"
S3_PNL_500    = "plot_mc_pnl(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"
S3_PNL_PCT5   = "plot_mc_pnl(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"
S3_SHARPE_500 = "plot_mc_sharpe(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"
S3_SHARPE_PCT5= "plot_mc_sharpe(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"
S3_DD_500     = "plot_mc_dd(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"
S3_DD_PCT5    = "plot_mc_dd(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"

S4_EQ_500   = "plot_ml2_indiv_equity(s4_pnl_by_combo, bars, 'fixed_dollars_500')"
S4_EQ_PCT5  = "plot_ml2_indiv_equity(s4_pnl_by_combo, bars, 'pct5_compound')"
S4_DD_500   = "plot_ml2_indiv_dd(s4_pnl_by_combo, bars, 'fixed_dollars_500')"
S4_DD_PCT5  = "plot_ml2_indiv_dd(s4_pnl_by_combo, bars, 'pct5_compound')"

S5_EQ_500   = "plot_ml2_combined_equity(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500')"
S5_EQ_PCT5  = "plot_ml2_combined_equity(ml2_portfolio['pct5_compound'], 'pct5_compound')"
S5_DD_500   = "plot_ml2_combined_dd(ml2_portfolio['fixed_dollars_500'], bars, 'fixed_dollars_500')"
S5_DD_PCT5  = "plot_ml2_combined_dd(ml2_portfolio['pct5_compound'], bars, 'pct5_compound')"

S6_SIMS_500   = "plot_mc_sims(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"
S6_SIMS_PCT5  = "plot_mc_sims(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"
S6_PNL_500    = "plot_mc_pnl(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"
S6_PNL_PCT5   = "plot_mc_pnl(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"
S6_SHARPE_500 = "plot_mc_sharpe(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"
S6_SHARPE_PCT5= "plot_mc_sharpe(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"
S6_DD_500     = "plot_mc_dd(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"
S6_DD_PCT5    = "plot_mc_dd(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"


# ─── Top-performance notebook ────────────────────────────────────────────────

def build_performance() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(); nb.metadata = NB_META
    nb.cells = [
        _md("# Top-K composed strategies - 6-section evaluation\n\n"
            "Evaluates the C1-selected top-10 combos on the 20% OOS test partition,\n"
            "under two sizing policies:\n"
            "- `fixed_dollars_500` — risk $500 on every trade, forever.\n"
            "- `pct5_compound` — risk 5% of *current* equity on every trade;\n"
            "  starts at $2,500 (=5% of $50k) and compounds trade-by-trade.\n\n"
            "Sections:\n"
            "1. Individual (unfiltered)\n"
            "2. Combined portfolio (unfiltered)\n"
            "3. Monte Carlo on combined (unfiltered)\n"
            "4. Individual with ML#2 (V3 booster + calibrator)\n"
            "5. Combined portfolio with ML#2\n"
            "6. Monte Carlo on combined ML#2\n\n"
            "Every plot is in its own cell, and every plot type is rendered once\n"
            "per sizing policy (separate cells).", "intro"),
        _code(SETUP_IMPORTS, "setup"),
        _code(PLOT_HELPERS, "plot-helpers"),
        _code(RUN_UNFILTERED, "run-unfiltered"),
        _code(RUN_ML2, "run-ml2"),

        _md("## 1) Individual (unfiltered)", "s1-md"),
        _code(S1_PERF, "s1-perf"),
        _code(S1_EQ_500, "s1-eq-500"),
        _code(S1_EQ_PCT5, "s1-eq-pct5"),
        _code(S1_DD_500, "s1-dd-500"),
        _code(S1_DD_PCT5, "s1-dd-pct5"),

        _md("## 2) Combined portfolio (unfiltered)", "s2-md"),
        _code(S2_COMBINED_BUILD, "s2-table"),
        _code(S2_EQ_500, "s2-eq-500"),
        _code(S2_EQ_PCT5, "s2-eq-pct5"),
        _code(S2_DD_500, "s2-dd-500"),
        _code(S2_DD_PCT5, "s2-dd-pct5"),

        _md("## 3) Monte Carlo on combined (unfiltered)", "s3-md"),
        _code(S3_MC, "s3-mc"),
        _code(S3_SIMS_500, "s3-sims-500"),
        _code(S3_SIMS_PCT5, "s3-sims-pct5"),
        _code(S3_PNL_500, "s3-pnl-500"),
        _code(S3_PNL_PCT5, "s3-pnl-pct5"),
        _code(S3_SHARPE_500, "s3-sharpe-500"),
        _code(S3_SHARPE_PCT5, "s3-sharpe-pct5"),
        _code(S3_DD_500, "s3-dd-500"),
        _code(S3_DD_PCT5, "s3-dd-pct5"),

        _md("## 4) Individual with ML#2 (V3 filter)", "s4-md"),
        _code(S4_PERF, "s4-perf"),
        _code(S4_EQ_500, "s4-eq-500"),
        _code(S4_EQ_PCT5, "s4-eq-pct5"),
        _code(S4_DD_500, "s4-dd-500"),
        _code(S4_DD_PCT5, "s4-dd-pct5"),

        _md("## 5) Combined portfolio with ML#2", "s5-md"),
        _code(S5_PORTFOLIO_BUILD, "s5-table"),
        _code(S5_EQ_500, "s5-eq-500"),
        _code(S5_EQ_PCT5, "s5-eq-pct5"),
        _code(S5_DD_500, "s5-dd-500"),
        _code(S5_DD_PCT5, "s5-dd-pct5"),

        _md("## 6) Monte Carlo on combined ML#2", "s6-md"),
        _code(S6_MC, "s6-mc"),
        _code(S6_SIMS_500, "s6-sims-500"),
        _code(S6_SIMS_PCT5, "s6-sims-pct5"),
        _code(S6_PNL_500, "s6-pnl-500"),
        _code(S6_PNL_PCT5, "s6-pnl-pct5"),
        _code(S6_SHARPE_500, "s6-sharpe-500"),
        _code(S6_SHARPE_PCT5, "s6-sharpe-pct5"),
        _code(S6_DD_500, "s6-dd-500"),
        _code(S6_DD_PCT5, "s6-dd-pct5"),
    ]
    return nb


def main() -> None:
    perf = build_performance()
    nbf.write(perf, str(EVAL / 'top_performance.ipynb'))
    print(f"Wrote {EVAL / 'top_performance.ipynb'}")


if __name__ == "__main__":
    main()
