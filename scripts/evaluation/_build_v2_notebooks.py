"""Build 6 small top-performance notebooks, one per analysis section.

Each notebook imports shared logic from `_top_perf_common.py`, calls
`load_setup()` once at the top (which uses disk caches for the expensive
results_raw + combos_ml2 builds), and then runs its section-specific table
and plot cells.

Layout:
  evaluation/s1_individual.ipynb          # individual unfiltered
  evaluation/s2_combined.ipynb            # combined portfolio unfiltered
  evaluation/s3_mc_combined.ipynb         # Monte Carlo on combined unfiltered
  evaluation/s4_individual_ml2.ipynb      # individual with ML#2 filter
  evaluation/s5_combined_ml2.ipynb        # combined portfolio with ML#2
  evaluation/s6_mc_combined_ml2.ipynb     # Monte Carlo on combined ML#2

Sizing policies compared in every section:
  - fixed_dollars_500: risk $500 on every trade, forever.
  - pct5_compound:     risk 5% of *current* equity; equity starts at $50k.

Execute with nbclient afterwards (one kernel process per notebook; disk
caches keep the total runtime close to a single-notebook run on re-execute).
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
    c = nbf.v4.new_markdown_cell(src); c["id"] = cid; return c


def _code(src: str, cid: str) -> nbf.NotebookNode:
    c = nbf.v4.new_code_cell(src); c["id"] = cid; return c


# ─── Shared setup cell (identical in every notebook) ────────────────────────

SETUP_TEMPLATE = '''import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path.cwd().resolve()
while not (REPO / 'src').exists() and REPO.parent != REPO:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from scripts.evaluation._top_perf_common import (
    STARTING_EQUITY, RISK_FRAC, POLICIES,
    apply_sizing, metrics_from_pnl, monte_carlo,
    load_setup,
    plot_indiv_equity, plot_indiv_dd,
    plot_combined_equity, plot_combined_dd,
    plot_ml2_indiv_equity, plot_ml2_indiv_dd,
    plot_ml2_combined_equity, plot_ml2_combined_dd,
    plot_mc_sims, plot_mc_pnl, plot_mc_sharpe, plot_mc_dd,
)

_ctx = load_setup(cost_per_contract_rt={cost_rt}, top_strategies_path={tsp})
bars            = _ctx['bars']
YEARS_SPAN      = _ctx['years_span']
strategies      = _ctx['strategies']
results_raw     = _ctx['results_raw']
combined_raw    = _ctx['combined_raw']
combos_ml2      = _ctx['combos_ml2']
s4_pnl_by_combo = _ctx['s4_pnl_by_combo']
ml2_portfolio   = _ctx['ml2_portfolio']
'''


def _setup(cost_rt: str, tsp_expr: str) -> str:
    return SETUP_TEMPLATE.format(cost_rt=cost_rt, tsp=tsp_expr)


SETUP = _setup("0.0", "None")
SETUP_NET = _setup("5.0", "None")

NET_COST_BANNER = (
    "\n\n**Cost model:** every trade is charged "
    "`contracts × $5.00` round-trip (≈ $3 retail commission + 2 ticks/side "
    "slippage on MNQ at $0.50/tick). Applied to both sizing policies."
)


# ─── Per-section table cells (kept inline so each notebook is self-readable) ─

S1_PERF = '''rows = []
for r in results_raw:
    if r['trades'].empty:
        for policy in POLICIES:
            rows.append({'combo_id': r['combo_id'], 'policy': policy,
                         **metrics_from_pnl(np.array([]), YEARS_SPAN, policy=policy)})
        continue
    t = r['trades'].sort_values('date', kind='mergesort')
    pnl_base = t['actual_pnl'].to_numpy(dtype=float)
    risk_base = t['dollar_risk'].to_numpy(dtype=float)
    r_mult = np.where(risk_base > 0, pnl_base / risk_base, 0.0)
    for policy in POLICIES:
        pnl = apply_sizing(pnl_base, risk_base, policy)
        rows.append({'combo_id': r['combo_id'], 'policy': policy,
                     **metrics_from_pnl(pnl, YEARS_SPAN, policy=policy, r=r_mult)})
perf1 = pd.DataFrame(rows)
perf1'''

S2_TABLE = '''rows = []
for policy in POLICIES:
    if combined_raw.empty:
        rows.append({'policy': policy,
                     **metrics_from_pnl(np.array([]), YEARS_SPAN, policy=policy)})
        continue
    pnl_base = combined_raw['actual_pnl'].to_numpy(dtype=float)
    risk_base = combined_raw['dollar_risk'].to_numpy(dtype=float)
    r_mult = np.where(risk_base > 0, pnl_base / risk_base, 0.0)
    pnl = apply_sizing(pnl_base, risk_base, policy)
    rows.append({'policy': policy,
                 **metrics_from_pnl(pnl, YEARS_SPAN, policy=policy, r=r_mult)})
combined_table_raw = pd.DataFrame(rows)
combined_table_raw'''

S3_TABLE = '''rows = []
for policy in POLICIES:
    if combined_raw.empty:
        rows.append({'policy': policy, 'n_trades': 0}); continue
    pnl_base = combined_raw['actual_pnl'].to_numpy(dtype=float)
    risk_base = combined_raw['dollar_risk'].to_numpy(dtype=float)
    rows.append({'policy': policy,
                 **monte_carlo(pnl_base, risk_base, policy, YEARS_SPAN)})
mc_raw = pd.DataFrame(rows)
mc_raw'''

S4_PERF = '''rows = []
for cid, entry in s4_pnl_by_combo.items():
    pnl_base = entry['pnl_base']; risk_base = entry['risk_base']
    if len(pnl_base) == 0:
        for policy in POLICIES:
            rows.append({'combo_id': cid, 'policy': policy,
                         **metrics_from_pnl(np.array([]), YEARS_SPAN, policy=policy)})
        continue
    r_mult = np.where(risk_base > 0, pnl_base / risk_base, 0.0)
    for policy in POLICIES:
        pnl = entry['by_policy'][policy]
        rows.append({'combo_id': cid, 'policy': policy,
                     **metrics_from_pnl(pnl, YEARS_SPAN, policy=policy, r=r_mult)})
perf4 = pd.DataFrame(rows)
perf4'''

S5_TABLE = '''rows = []
for policy in POLICIES:
    df = ml2_portfolio[policy]
    if df.empty:
        rows.append({'policy': policy, 'n_trades': 0}); continue
    pnl_p = df['actual_pnl'].to_numpy(dtype=float)
    risk_p = df['dollar_risk'].to_numpy(dtype=float)
    r_p = np.where(risk_p > 0, pnl_p / risk_p, 0.0)
    rows.append({'policy': policy,
                 **metrics_from_pnl(pnl_p, YEARS_SPAN, policy=policy, r=r_p)})
combined_table_ml2 = pd.DataFrame(rows)
combined_table_ml2'''

S6_TABLE = '''rows = []
for policy in POLICIES:
    df = ml2_portfolio[policy]
    if df.empty:
        rows.append({'policy': policy, 'n_trades': 0}); continue
    pnl_p = df['actual_pnl'].to_numpy(dtype=float)
    risk_p = df['dollar_risk'].to_numpy(dtype=float)
    rows.append({'policy': policy,
                 **monte_carlo(pnl_p, risk_p, policy, YEARS_SPAN)})
mc_ml2 = pd.DataFrame(rows)
mc_ml2'''


# ─── Per-notebook cell definitions ──────────────────────────────────────────

SECTIONS = [
    dict(
        filename="s1_individual.ipynb",
        title="# §1 Individual (unfiltered)",
        body=(
            "Per-combo metrics and per-combo equity/drawdown curves on the\n"
            "20% OOS test partition with no ML#2 filter. Two sizing policies\n"
            "compared: `fixed_dollars_500` and `pct5_compound`."
        ),
        cells=[
            ("s1-perf",     "code", S1_PERF),
            ("s1-eq-500",   "code", "plot_indiv_equity(results_raw, 'fixed_dollars_500')"),
            ("s1-eq-pct5",  "code", "plot_indiv_equity(results_raw, 'pct5_compound')"),
            ("s1-dd-500",   "code", "plot_indiv_dd(results_raw, 'fixed_dollars_500')"),
            ("s1-dd-pct5",  "code", "plot_indiv_dd(results_raw, 'pct5_compound')"),
        ],
    ),
    dict(
        filename="s2_combined.ipynb",
        title="# §2 Combined portfolio (unfiltered)",
        body=(
            "Aggregate of all 10 combos' unfiltered trades, sorted by exit time.\n"
            "Shows what running the full top-K as a single portfolio would look\n"
            "like on the OOS partition."
        ),
        cells=[
            ("s2-table",    "code", S2_TABLE),
            ("s2-eq-500",   "code", "plot_combined_equity(combined_raw, 'fixed_dollars_500')"),
            ("s2-eq-pct5",  "code", "plot_combined_equity(combined_raw, 'pct5_compound')"),
            ("s2-dd-500",   "code", "plot_combined_dd(combined_raw, 'fixed_dollars_500', bars)"),
            ("s2-dd-pct5",  "code", "plot_combined_dd(combined_raw, 'pct5_compound', bars)"),
        ],
    ),
    dict(
        filename="s3_mc_combined.ipynb",
        title="# §3 Monte Carlo on combined (unfiltered)",
        body=(
            "IID bootstrap (10,000 sims) on the combined unfiltered trade stream.\n"
            "Plots: bootstrap equity paths, final PnL histogram, annualized Sharpe\n"
            "histogram, max-drawdown histogram — one of each per sizing policy."
        ),
        cells=[
            ("s3-mc",         "code", S3_TABLE),
            ("s3-sims-500",   "code", "plot_mc_sims(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"),
            ("s3-sims-pct5",  "code", "plot_mc_sims(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"),
            ("s3-pnl-500",    "code", "plot_mc_pnl(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"),
            ("s3-pnl-pct5",   "code", "plot_mc_pnl(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"),
            ("s3-sharpe-500", "code", "plot_mc_sharpe(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"),
            ("s3-sharpe-pct5","code", "plot_mc_sharpe(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"),
            ("s3-dd-500",     "code", "plot_mc_dd(combined_raw, 'fixed_dollars_500', 'unfiltered', YEARS_SPAN)"),
            ("s3-dd-pct5",    "code", "plot_mc_dd(combined_raw, 'pct5_compound', 'unfiltered', YEARS_SPAN)"),
        ],
    ),
    dict(
        filename="s4_individual_ml2.ipynb",
        title="# §4 Individual with ML#2 filter",
        body=(
            "Per-combo metrics and equity/drawdown curves after applying the V3\n"
            "booster + pooled-R:R isotonic calibrator filter."
        ),
        cells=[
            ("s4-perf",    "code", S4_PERF),
            ("s4-eq-500",  "code", "plot_ml2_indiv_equity(s4_pnl_by_combo, bars, 'fixed_dollars_500')"),
            ("s4-eq-pct5", "code", "plot_ml2_indiv_equity(s4_pnl_by_combo, bars, 'pct5_compound')"),
            ("s4-dd-500",  "code", "plot_ml2_indiv_dd(s4_pnl_by_combo, bars, 'fixed_dollars_500')"),
            ("s4-dd-pct5", "code", "plot_ml2_indiv_dd(s4_pnl_by_combo, bars, 'pct5_compound')"),
        ],
    ),
    dict(
        filename="s5_combined_ml2.ipynb",
        title="# §5 Combined portfolio with ML#2",
        body=(
            "Event-driven portfolio simulator over the ML#2-filtered trade stream.\n"
            "For `pct5_compound`, each entry uses the *live* equity at that bar\n"
            "(natural compounding); for `fixed_dollars_500` every entry risks $500."
        ),
        cells=[
            ("s5-table",   "code", S5_TABLE),
            ("s5-eq-500",  "code", "plot_ml2_combined_equity(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500')"),
            ("s5-eq-pct5", "code", "plot_ml2_combined_equity(ml2_portfolio['pct5_compound'], 'pct5_compound')"),
            ("s5-dd-500",  "code", "plot_ml2_combined_dd(ml2_portfolio['fixed_dollars_500'], bars, 'fixed_dollars_500')"),
            ("s5-dd-pct5", "code", "plot_ml2_combined_dd(ml2_portfolio['pct5_compound'], bars, 'pct5_compound')"),
        ],
    ),
    dict(
        filename="s6_mc_combined_ml2.ipynb",
        title="# §6 Monte Carlo on combined ML#2",
        body=(
            "IID bootstrap (10,000 sims) on the ML#2-filtered portfolio trade stream.\n"
            "Same plot set as §3."
        ),
        cells=[
            ("s6-mc",         "code", S6_TABLE),
            ("s6-sims-500",   "code", "plot_mc_sims(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"),
            ("s6-sims-pct5",  "code", "plot_mc_sims(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"),
            ("s6-pnl-500",    "code", "plot_mc_pnl(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"),
            ("s6-pnl-pct5",   "code", "plot_mc_pnl(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"),
            ("s6-sharpe-500", "code", "plot_mc_sharpe(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"),
            ("s6-sharpe-pct5","code", "plot_mc_sharpe(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"),
            ("s6-dd-500",     "code", "plot_mc_dd(ml2_portfolio['fixed_dollars_500'], 'fixed_dollars_500', 'ML2', YEARS_SPAN)"),
            ("s6-dd-pct5",    "code", "plot_mc_dd(ml2_portfolio['pct5_compound'], 'pct5_compound', 'ML2', YEARS_SPAN)"),
        ],
    ),
]


def build_section(section, setup_src: str, title_suffix: str = "",
                  body_suffix: str = "") -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(); nb.metadata = NB_META
    title = section["title"] + title_suffix
    body = section["body"] + body_suffix
    cells = [
        _md(f"{title}\n\n{body}", "intro"),
        _code(setup_src, "setup"),
    ]
    for cid, kind, src in section["cells"]:
        cells.append(_code(src, cid) if kind == "code" else _md(src, cid))
    nb.cells = cells
    return nb


OUTPUTS = [s["filename"] for s in SECTIONS]


def _build_variant(gross_dir: Path, net_dir: Path,
                   setup_src: str, setup_net_src: str,
                   title_tag: str = "") -> None:
    gross_dir.mkdir(parents=True, exist_ok=True)
    net_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f" {title_tag}" if title_tag else ""
    for section in SECTIONS:
        nb = build_section(section, setup_src, title_suffix=title_suffix)
        path = gross_dir / section["filename"]
        nbf.write(nb, str(path))
        print(f"Wrote {path}")
    for section in SECTIONS:
        net_name = section["filename"].replace(".ipynb", "_net.ipynb")
        nb = build_section(section, setup_net_src,
                           title_suffix=f"{title_suffix} — net of costs",
                           body_suffix=NET_COST_BANNER)
        path = net_dir / net_name
        nbf.write(nb, str(path))
        print(f"Wrote {path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["v10", "v11", "v12",
                             "v12_k05", "v12_k10", "v12_top50", "all"],
                    default="all",
                    help="Which top-K source to build for.")
    args = ap.parse_args()

    if args.variant in ("v10", "all"):
        _build_variant(
            EVAL / "v10_topk", EVAL / "v10_topk_net",
            SETUP, SETUP_NET,
        )

    if args.variant in ("v11", "all"):
        tsp = "REPO / 'evaluation' / 'top_strategies_v11.json'"
        _build_variant(
            EVAL / "v11_topk", EVAL / "v11_topk_net",
            _setup("0.0", tsp), _setup("5.0", tsp),
            title_tag="(v11)",
        )

    if args.variant in ("v12", "all"):
        tsp = "REPO / 'evaluation' / 'top_strategies_v12.json'"
        _build_variant(
            EVAL / "v12_topk", EVAL / "v12_topk_net",
            _setup("0.0", tsp), _setup("5.0", tsp),
            title_tag="(v12)",
        )

    # Tier 1A — UCB kappa variants on the v12 booster (same model, different
    # selection pressure). Tests whether penalizing high-p90-p10-spread combos
    # lifts OOS Sharpe.
    if args.variant in ("v12_k05", "all"):
        tsp = "REPO / 'evaluation' / 'top_strategies_v12_k05.json'"
        _build_variant(
            EVAL / "v12_topk_k05", EVAL / "v12_topk_k05_net",
            _setup("0.0", tsp), _setup("5.0", tsp),
            title_tag="(v12 kappa=0.5)",
        )

    if args.variant in ("v12_k10", "all"):
        tsp = "REPO / 'evaluation' / 'top_strategies_v12_k10.json'"
        _build_variant(
            EVAL / "v12_topk_k10", EVAL / "v12_topk_k10_net",
            _setup("0.0", tsp), _setup("5.0", tsp),
            title_tag="(v12 kappa=1.0)",
        )

    # Tier 1B — top-50 expansion (same kappa=0, bigger slice). Tests whether
    # the ranker is fine and only the top-10 cutoff is too noisy.
    if args.variant in ("v12_top50", "all"):
        tsp = "REPO / 'evaluation' / 'top_strategies_v12_top50.json'"
        _build_variant(
            EVAL / "v12_topk_top50", EVAL / "v12_topk_top50_net",
            _setup("0.0", tsp), _setup("5.0", tsp),
            title_tag="(v12 top-50)",
        )

    # Remove the old monolithic notebook if present — it's replaced by the
    # 6 section notebooks above.
    old = EVAL / "top_performance.ipynb"
    if old.exists():
        old.unlink()
        print(f"Removed {old}")


if __name__ == "__main__":
    main()
