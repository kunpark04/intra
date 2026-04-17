"""Build the 6-section top-performance notebook.

Section layout:
  1) individual              (unfiltered, raw composed_strategy_runner)
  2) combined                (unfiltered portfolio aggregate)
  3) Monte Carlo on combined (unfiltered)
  4) individual with ML2     (V3 booster + calibrator filter)
  5) combined with ML2       (event-driven portfolio of V3-filtered trades)
  6) Monte Carlo on combined ML2

Both $500-fixed and $2500 (5% of starting equity) sizing policies are shown.
Sharpe is annualized by `sqrt(trades_per_year)` using the test-partition span,
applied consistently across every section.

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

SETUP_IMPORTS = """%matplotlib inline
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
SIZINGS = [('fixed_dollars_500', 1.0), ('fixed5_2500', 5.0)]


def metrics_from_pnl(pnl, years_span, start_equity=STARTING_EQUITY):
    '''Headline metrics for a trade-PnL series.

    Sharpe is annualized: sharpe = (mean/std) * sqrt(trades_per_year), where
    trades_per_year = len(pnl) / years_span. This matches the standard
    convention for per-trade PnL under an i.i.d. assumption and is consistent
    across Sections 1/2/4/5.
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
    std = p.std(ddof=1) if n > 1 else 0.0
    sharpe = float(p.mean() / std * np.sqrt(tpy)) if std > 0 and tpy > 0 else 0.0
    return dict(n_trades=int(n),
                trades_per_year=round(tpy, 1),
                win_rate=round(float((p > 0).mean()), 4),
                total_pnl_dollars=round(total, 2),
                total_return_pct=round(total / start_equity * 100, 2),
                sharpe_ratio=round(sharpe, 4),
                max_drawdown_pct=round(dd_pct, 2),
                max_drawdown_dollars=round(dd_d, 2))


def monte_carlo(pnl, years_span, start_equity=STARTING_EQUITY, n_sims=10_000, seed=42):
    '''IID bootstrap MC on a PnL series. Returns DD percentiles + VaR/CVaR +
    risk-of-ruin (>=50% DD) + annualized Sharpe CI + permutation-test on WR.

    Annualized Sharpe per resample: (mean/std)*sqrt(trades_per_year) where
    trades_per_year = n / years_span. Matches metrics_from_pnl convention.
    '''
    rng = np.random.default_rng(seed)
    p = np.asarray(pnl, dtype=float)
    n = len(p)
    if n == 0:
        return {'n_sims': n_sims, 'n_trades': 0, 'note': 'empty'}
    idx = rng.integers(0, n, size=(n_sims, n))
    samples = p[idx]
    equity = start_equity + np.cumsum(samples, axis=1)
    equity = np.concatenate([np.full((n_sims, 1), start_equity), equity], axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    dd_pct = np.nan_to_num((peak - equity) / peak, nan=0.0).max(axis=1) * 100
    # Annualized Sharpe per resample (i.i.d. assumption).
    tpy = n / years_span if years_span > 0 else 0.0
    mu = samples.mean(axis=1)
    sig = samples.std(axis=1, ddof=1) if n > 1 else np.zeros(n_sims)
    sharpe_boot = np.where(sig > 0, (mu / sig) * np.sqrt(tpy), 0.0)
    var5 = float(np.percentile(p, 5))
    tail = p[p <= var5]
    cvar = float(tail.mean()) if len(tail) else var5
    wr = float((p > 0).mean())
    boot_wr = (samples > 0).mean(axis=1)
    wr_ci = (float(np.percentile(boot_wr, 2.5)), float(np.percentile(boot_wr, 97.5)))
    sharpe_ci = (float(np.percentile(sharpe_boot, 2.5)),
                 float(np.percentile(sharpe_boot, 97.5)))
    return {
        'n_sims': n_sims, 'n_trades': int(n),
        'win_rate': round(wr, 4),
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

RUN_ML2 = """import lightgbm as lgb
print('Loading V3 booster + calibrators...')
_v3inf = v3eval.v3inf
booster = lgb.Booster(model_file=str(_v3inf.V3_BOOSTER))
simple_cals = _v3inf._load_calibrators()
two_stage = _v3inf._load_per_combo_calibrators()

print('\\nBuilding V3-filtered trades per combo (may take a few minutes)...')
combo_ids = [s['global_combo_id'] for s in strategies]
combos_ml2 = []
for gcid in combo_ids:
    print(f'  {gcid}...', flush=True)
    try:
        c = v3eval.build_combo_trades_test(gcid, booster, simple_cals, two_stage)
        print(f'    n_trades={c.get("n_trades", 0)}  rr={c.get("rr", float("nan")):.2f}')
    except Exception as e:
        c = {'combo_id': gcid, 'error': str(e)}
        print(f'    ERROR: {e}')
    combos_ml2.append(c)
print('Done (ML2).')"""


# ─── Section 1 / 2 / 3 (unfiltered) ──────────────────────────────────────────

S1_PERF = """rows = []
for r in results_raw:
    pnl = r['trades']['actual_pnl'].to_numpy() if not r['trades'].empty else np.array([])
    for label, scale in SIZINGS:
        m = metrics_from_pnl(pnl * scale, YEARS_SPAN)
        rows.append({'combo_id': r['combo_id'], 'sizing': label, **m})
perf1 = pd.DataFrame(rows)
perf1"""

S1_EQUITY = """# Per-combo equity curves (overlaid).
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (label, scale) in zip(axes, SIZINGS):
    for r in results_raw:
        if r['trades'].empty:
            continue
        t = r['trades'].sort_values('date')
        pnl = t['actual_pnl'].to_numpy() * scale
        eq = STARTING_EQUITY + np.cumsum(pnl)
        ax.plot(t['date'], eq, linewidth=1.0, alpha=0.85, label=r['combo_id'])
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title(f'individual unfiltered - equity ({label})')
    ax.set_xlabel('time'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
plt.tight_layout(); plt.show()"""

S1_DD = """# Per-combo drawdown curves (overlaid).
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (label, scale) in zip(axes, SIZINGS):
    for r in results_raw:
        if r['trades'].empty:
            continue
        t = r['trades'].sort_values('date')
        pnl = t['actual_pnl'].to_numpy() * scale
        eq_full = np.concatenate([[STARTING_EQUITY], STARTING_EQUITY + np.cumsum(pnl)])
        peak = np.maximum.accumulate(eq_full)
        dd = (peak - eq_full) / peak * 100
        times = pd.concat([pd.Series([t['date'].iloc[0]]),
                           pd.Series(t['date'].values)]).reset_index(drop=True)
        ax.plot(times, dd, linewidth=1.0, alpha=0.85, label=r['combo_id'])
    ax.invert_yaxis()
    ax.set_title(f'individual unfiltered - drawdown ({label})')
    ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='lower left', ncol=2)
plt.tight_layout(); plt.show()"""

S2_COMBINED_BUILD = """frames = []
for r in results_raw:
    if r['trades'].empty:
        continue
    t = r['trades'][['date', 'actual_pnl']].copy()
    t.insert(0, 'combo_id', r['combo_id'])
    frames.append(t)
combined_raw = (pd.concat(frames, ignore_index=True)
                .sort_values('date', kind='mergesort')
                .reset_index(drop=True)
                if frames else pd.DataFrame())
print(f'Combined unfiltered trades: {len(combined_raw):,}')
rows = []
for label, scale in SIZINGS:
    pnl = combined_raw['actual_pnl'].to_numpy() * scale if not combined_raw.empty else np.array([])
    rows.append({'sizing': label, **metrics_from_pnl(pnl, YEARS_SPAN)})
combined_table_raw = pd.DataFrame(rows)
combined_table_raw"""

S2_EQUITY = """if not combined_raw.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (label, scale) in zip(axes, SIZINGS):
        pnl = combined_raw['actual_pnl'].to_numpy() * scale
        eq = STARTING_EQUITY + np.cumsum(pnl)
        ax.plot(combined_raw['date'], eq, linewidth=1.3)
        ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'combined unfiltered - equity ({label})')
        ax.set_xlabel('time'); ax.set_ylabel('equity ($)')
        ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
else:
    print('No trades.')"""

S2_DD = """if not combined_raw.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (label, scale) in zip(axes, SIZINGS):
        pnl = combined_raw['actual_pnl'].to_numpy() * scale
        eq_full = np.concatenate([[STARTING_EQUITY], STARTING_EQUITY + np.cumsum(pnl)])
        peak = np.maximum.accumulate(eq_full)
        dd = (peak - eq_full) / peak * 100
        times = pd.concat([pd.Series([bars['time'].iloc[0]]),
                           pd.Series(combined_raw['date'].values)]).reset_index(drop=True)
        ax.plot(times, dd, linewidth=1.3, color='#d62728')
        ax.invert_yaxis()
        ax.set_title(f'combined unfiltered - drawdown ({label})')
        ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
else:
    print('No trades.')"""

S3_MC = """rows = []
if not combined_raw.empty:
    for label, scale in SIZINGS:
        pnl = combined_raw['actual_pnl'].to_numpy() * scale
        rows.append({'sizing': label, **monte_carlo(pnl, YEARS_SPAN)})
mc_raw = pd.DataFrame(rows)
mc_raw"""

# Helper shared by both MC plot cells: bootstrap final-PnL + annualized-Sharpe
# distributions, then draw 2x2 (row=metric, col=sizing) with percentile lines.
MC_PLOT_HELPER = """def _mc_plot_grid(title, pnl_series_per_sizing, years_span, n_sims=10_000, seed=42):
    '''pnl_series_per_sizing: list[(label, pnl_array)].'''
    sizings = [(lab, p) for lab, p in pnl_series_per_sizing if len(p) > 0]
    if not sizings:
        print('No trades.'); return
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, len(sizings), figsize=(7 * len(sizings), 8),
                             squeeze=False)
    for col, (label, pnl_src) in enumerate(sizings):
        n = len(pnl_src)
        idx = rng.integers(0, n, size=(n_sims, n))
        samples = pnl_src[idx]
        final_pnl = samples.sum(axis=1)
        tpy = n / years_span if years_span > 0 else 0.0
        mu = samples.mean(axis=1)
        sig = samples.std(axis=1, ddof=1) if n > 1 else np.zeros(n_sims)
        sharpe = np.where(sig > 0, (mu / sig) * np.sqrt(tpy), 0.0)
        # Top row: final PnL ($)
        ax = axes[0, col]
        ax.hist(final_pnl, bins=60, color='#6a8cbb', alpha=0.85, edgecolor='white')
        ax.axvline(0, color='k', linewidth=0.8, alpha=0.6)
        for pct, colour, ls in [(2.5, '#b2182b', ':'), (50, '#1a9850', '--'),
                                (97.5, '#b2182b', ':')]:
            v = np.percentile(final_pnl, pct)
            ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                       label=f'p{pct:g}=${v:,.0f}')
        ax.set_title(f'{title} MC - final PnL ({label})')
        ax.set_xlabel('final PnL ($)'); ax.set_ylabel('freq')
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')
        # Bottom row: annualized Sharpe
        ax = axes[1, col]
        ax.hist(sharpe, bins=60, color='#d48c6a', alpha=0.85, edgecolor='white')
        ax.axvline(0, color='k', linewidth=0.8, alpha=0.6)
        for pct, colour, ls in [(2.5, '#b2182b', ':'), (50, '#1a9850', '--'),
                                (97.5, '#b2182b', ':')]:
            v = np.percentile(sharpe, pct)
            ax.axvline(v, color=colour, linestyle=ls, linewidth=1.2,
                       label=f'p{pct:g}={v:.2f}')
        ax.set_title(f'{title} MC - annualized Sharpe ({label})')
        ax.set_xlabel('Sharpe'); ax.set_ylabel('freq')
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout(); plt.show()"""

S3_MC_PLOT = """# Bootstrap final-PnL + annualized-Sharpe distributions (unfiltered).
if not combined_raw.empty:
    pnl_by_sizing = [(label, combined_raw['actual_pnl'].to_numpy() * scale)
                     for label, scale in SIZINGS]
    _mc_plot_grid('unfiltered', pnl_by_sizing, YEARS_SPAN)
else:
    print('No trades.')"""

S6_MC_PLOT = """# Bootstrap final-PnL + annualized-Sharpe distributions (ML2).
pnl_by_sizing = [(label, df['pnl_dollars'].to_numpy())
                 for label, df in ml2_portfolio.items()]
_mc_plot_grid('ML2', pnl_by_sizing, YEARS_SPAN)"""


# ─── Section 4 / 5 / 6 (ML2) ─────────────────────────────────────────────────

S4_PERF = """# Per-combo ML2-filtered PnL under fixed-$ sizing, computed locally so the
# Sharpe annualization matches Sections 1-2-5 (via metrics_from_pnl + YEARS_SPAN).
def _combo_pnl(c, fixed_dollars):
    if c.get('error') or c.get('n_trades', 0) == 0:
        return np.array([]), None
    sl = np.asarray(c['sl_pts'], dtype=float)
    pts = np.asarray(c['pnl_pts'], dtype=float)
    contracts = (fixed_dollars // (sl * v3eval.DOLLARS_PER_POINT)).astype(int)
    mask = contracts > 0
    pnl = (pts[mask] * contracts[mask] * v3eval.DOLLARS_PER_POINT).astype(float)
    exit_bars = np.asarray(c['exit_bar'])[mask]
    return pnl, exit_bars

s4_pnl_by_combo = {}  # combo_id -> {label -> (pnl, exit_bars)}
rows = []
for c in combos_ml2:
    cid = c.get('combo_id')
    s4_pnl_by_combo[cid] = {}
    if c.get('error') or c.get('n_trades', 0) == 0:
        for label, _ in SIZINGS:
            rows.append({'combo_id': cid, 'sizing': label, 'n_trades': 0,
                         'note': c.get('error', 'no trades')})
            s4_pnl_by_combo[cid][label] = (np.array([]), np.array([]))
        continue
    for label, fixed in [('fixed_dollars_500', 500.0), ('fixed5_2500', 2500.0)]:
        pnl, exit_bars = _combo_pnl(c, fixed)
        s4_pnl_by_combo[cid][label] = (pnl, exit_bars)
        rows.append({'combo_id': cid, 'sizing': label,
                     **metrics_from_pnl(pnl, YEARS_SPAN)})
perf4 = pd.DataFrame(rows)
perf4"""

S4_EQUITY = """# Per-combo ML2 equity curves (overlaid), x-axis = exit bar time.
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (label, _) in zip(axes, SIZINGS):
    for cid, by_lab in s4_pnl_by_combo.items():
        pnl, exit_bars = by_lab.get(label, (np.array([]), np.array([])))
        if len(pnl) == 0:
            continue
        times = pd.to_datetime(bars['time'].to_numpy()[exit_bars])
        eq = STARTING_EQUITY + np.cumsum(pnl)
        ax.plot(times, eq, linewidth=1.0, alpha=0.85, label=cid)
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title(f'individual ML2-filtered - equity ({label})')
    ax.set_xlabel('time'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
plt.tight_layout(); plt.show()"""

S4_DD = """# Per-combo ML2 drawdown curves (overlaid).
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (label, _) in zip(axes, SIZINGS):
    for cid, by_lab in s4_pnl_by_combo.items():
        pnl, exit_bars = by_lab.get(label, (np.array([]), np.array([])))
        if len(pnl) == 0:
            continue
        times = pd.to_datetime(bars['time'].to_numpy()[exit_bars])
        eq_full = np.concatenate([[STARTING_EQUITY], STARTING_EQUITY + np.cumsum(pnl)])
        peak = np.maximum.accumulate(eq_full)
        dd = (peak - eq_full) / peak * 100
        t0 = times.min() if len(times) else pd.to_datetime(bars['time'].iloc[0])
        t_full = pd.concat([pd.Series([t0]), pd.Series(times)]).reset_index(drop=True)
        ax.plot(t_full, dd, linewidth=1.0, alpha=0.85, label=cid)
    ax.invert_yaxis()
    ax.set_title(f'individual ML2-filtered - drawdown ({label})')
    ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='lower left', ncol=2)
plt.tight_layout(); plt.show()"""

S5_PORTFOLIO_BUILD = """# Event-driven portfolio per sizing (uses v3eval.simulate_portfolio).
# Also reconstruct equity curve per sizing by replaying events inline.

def _sim_with_equity_curve(combos, fixed_dollars):
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
    open_pos = {}
    trade_rows = []
    for bar, kind, ci, ti in events:
        c = combos[ci]
        if kind == 0:
            contracts = int(fixed_dollars // (c['sl_pts'][ti] * v3eval.DOLLARS_PER_POINT))
            if contracts <= 0:
                continue
            open_pos[(ci, ti)] = contracts
        else:
            key = (ci, ti)
            if key not in open_pos:
                continue
            contracts = open_pos.pop(key)
            pnl = c['pnl_pts'][ti] * contracts * v3eval.DOLLARS_PER_POINT
            realized += pnl
            equity = STARTING_EQUITY + realized
            exit_time = pd.to_datetime(bars['time'].iloc[bar]) \
                if bar < len(bars) else pd.NaT
            trade_rows.append({
                'combo_id': c['combo_id'], 'exit_time': exit_time,
                'pnl_dollars': pnl, 'equity_after': equity,
            })
    return pd.DataFrame(trade_rows)

ml2_portfolio = {label: _sim_with_equity_curve(combos_ml2, fixed)
                 for label, fixed in [('fixed_dollars_500', 500.0),
                                      ('fixed5_2500', 2500.0)]}

rows = []
for label, df in ml2_portfolio.items():
    if df.empty:
        rows.append({'sizing': label, 'n_trades': 0})
        continue
    pnl = df['pnl_dollars'].to_numpy()
    rows.append({'sizing': label, **metrics_from_pnl(pnl, YEARS_SPAN)})
combined_table_ml2 = pd.DataFrame(rows)
combined_table_ml2"""

S5_EQUITY = """fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (label, df) in zip(axes, ml2_portfolio.items()):
    if df.empty:
        ax.set_title(f'ML2 portfolio {label} - no trades'); continue
    ax.plot(df['exit_time'], df['equity_after'], linewidth=1.3, color='#1f77b4')
    ax.axhline(STARTING_EQUITY, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_title(f'ML2 combined portfolio - equity ({label})')
    ax.set_xlabel('time'); ax.set_ylabel('equity ($)'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""

S5_DD = """fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (label, df) in zip(axes, ml2_portfolio.items()):
    if df.empty:
        ax.set_title(f'ML2 portfolio {label} - no trades'); continue
    eq_full = np.concatenate([[STARTING_EQUITY], df['equity_after'].to_numpy()])
    peak = np.maximum.accumulate(eq_full)
    dd = (peak - eq_full) / peak * 100
    times = pd.concat([pd.Series([bars['time'].iloc[0]]),
                       pd.Series(df['exit_time'].values)]).reset_index(drop=True)
    ax.plot(times, dd, linewidth=1.3, color='#d62728'); ax.invert_yaxis()
    ax.set_title(f'ML2 combined portfolio - drawdown ({label})')
    ax.set_xlabel('time'); ax.set_ylabel('drawdown (%)'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""

S6_MC = """rows = []
for label, df in ml2_portfolio.items():
    if df.empty:
        rows.append({'sizing': label, 'n_trades': 0}); continue
    rows.append({'sizing': label,
                 **monte_carlo(df['pnl_dollars'].to_numpy(), YEARS_SPAN)})
mc_ml2 = pd.DataFrame(rows)
mc_ml2"""


# ─── Top-performance notebook ────────────────────────────────────────────────

def build_performance() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(); nb.metadata = NB_META
    nb.cells = [
        _md("# Top-K composed strategies - 6-section evaluation\n\n"
            "Evaluates the C1-selected top-10 combos on the 20% OOS test partition,\n"
            "under two sizing policies: `fixed_dollars_500` ($500/trade) and\n"
            "`fixed5_2500` (5% of $50k starting equity = $2500/trade).\n\n"
            "Sections:\n"
            "1. Individual (unfiltered)\n"
            "2. Combined portfolio (unfiltered)\n"
            "3. Monte Carlo on combined (unfiltered)\n"
            "4. Individual with ML#2 (V3 booster + calibrator)\n"
            "5. Combined portfolio with ML#2\n"
            "6. Monte Carlo on combined ML#2", "intro"),
        _code(SETUP_IMPORTS, "setup"),
        _code(MC_PLOT_HELPER, "mc-plot-helper"),
        _code(RUN_UNFILTERED, "run-unfiltered"),
        _code(RUN_ML2, "run-ml2"),
        _md("## 1) Individual (unfiltered)", "s1-md"),
        _code(S1_PERF, "s1-perf"),
        _code(S1_EQUITY, "s1-equity"),
        _code(S1_DD, "s1-dd"),
        _md("## 2) Combined portfolio (unfiltered)", "s2-md"),
        _code(S2_COMBINED_BUILD, "s2-table"),
        _code(S2_EQUITY, "s2-equity"),
        _code(S2_DD, "s2-dd"),
        _md("## 3) Monte Carlo on combined (unfiltered)", "s3-md"),
        _code(S3_MC, "s3-mc"),
        _code(S3_MC_PLOT, "s3-mc-plot"),
        _md("## 4) Individual with ML#2 (V3 filter)", "s4-md"),
        _code(S4_PERF, "s4-perf"),
        _code(S4_EQUITY, "s4-equity"),
        _code(S4_DD, "s4-dd"),
        _md("## 5) Combined portfolio with ML#2", "s5-md"),
        _code(S5_PORTFOLIO_BUILD, "s5-table"),
        _code(S5_EQUITY, "s5-equity"),
        _code(S5_DD, "s5-dd"),
        _md("## 6) Monte Carlo on combined ML#2", "s6-md"),
        _code(S6_MC, "s6-mc"),
        _code(S6_MC_PLOT, "s6-mc-plot"),
    ]
    return nb


def main() -> None:
    perf = build_performance()
    nbf.write(perf, str(EVAL / 'top_performance.ipynb'))
    print(f"Wrote {EVAL / 'top_performance.ipynb'}")


if __name__ == "__main__":
    main()
