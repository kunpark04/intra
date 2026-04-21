"""Probe 2 readout — mechanically apply the §4 gates from probe2_preregistration.md.

Formula (per §3):
  net_sharpe(tf) = mean(net_pnl_dollars) / std(net_pnl_dollars, ddof=1)
                 * sqrt(n_trades / YEARS_SPAN_TEST)
  net_dollars_per_year(tf) = mean(net_pnl_dollars) * n_trades / YEARS_SPAN_TEST

Gates (§4) — ALL THREE on AT LEAST ONE timeframe:
  (1) net_sharpe(tf)           >= 1.3
  (2) n_trades(tf)             >= 50
  (3) net_dollars_per_year(tf) >= 5000

Tie-breaking (§4): "approximately meets" is not meets. 1.29 sharpe = FAIL.
Cross-timeframe mixing NOT admissible.

Inputs:
  data/ml/probe2/combo865_15m_test.parquet
  data/ml/probe2/combo865_1h_test.parquet

Outputs:
  stdout: per-timeframe metrics + gate application table
  data/ml/probe2/readout.json: machine-readable verdict record
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

YEARS_SPAN_TEST = 1.4799  # per §3 preregistration; frozen at signing

GATE_SHARPE = 1.3
GATE_TRADES = 50
GATE_DOLLARS = 5000.0

TIMEFRAMES = [
    ("15m", Path("data/ml/probe2/combo865_15m_test.parquet")),
    ("1h", Path("data/ml/probe2/combo865_1h_test.parquet")),
]


def compute_tf_metrics(df: pd.DataFrame) -> dict:
    """Compute net metrics for a single-combo, single-timeframe trade log."""
    n = len(df)
    pnl = df["net_pnl_dollars"].to_numpy(dtype=float)
    gross = df["gross_pnl_dollars"].to_numpy(dtype=float)

    mean_net = float(pnl.mean())
    std_net = float(pnl.std(ddof=1))  # sample stdev
    net_sharpe = (mean_net / std_net) * math.sqrt(n / YEARS_SPAN_TEST) if std_net > 0 else 0.0
    net_dollars_per_year = mean_net * n / YEARS_SPAN_TEST

    # Audit-only (not gate-bound)
    mean_gross = float(gross.mean())
    std_gross = float(gross.std(ddof=1))
    gross_sharpe = (mean_gross / std_gross) * math.sqrt(n / YEARS_SPAN_TEST) if std_gross > 0 else 0.0
    win_rate = float((df["label_win"] == 1).mean())
    total_friction = float(df["friction_dollars"].sum())

    return {
        "n_trades": n,
        "net_sharpe": net_sharpe,
        "net_dollars_per_year": net_dollars_per_year,
        "mean_net_pnl": mean_net,
        "std_net_pnl": std_net,
        "total_net_pnl": float(pnl.sum()),
        "gross_sharpe": gross_sharpe,
        "mean_gross_pnl": mean_gross,
        "total_gross_pnl": float(gross.sum()),
        "total_friction": total_friction,
        "win_rate": win_rate,
    }


def apply_gates(m: dict) -> dict:
    """Mechanically apply §4 gates. Strict inequalities (>=) only."""
    g1 = m["net_sharpe"] >= GATE_SHARPE
    g2 = m["n_trades"] >= GATE_TRADES
    g3 = m["net_dollars_per_year"] >= GATE_DOLLARS
    return {
        "gate_1_sharpe_ge_1.3": bool(g1),
        "gate_2_trades_ge_50": bool(g2),
        "gate_3_dollars_ge_5000": bool(g3),
        "all_three_pass": bool(g1 and g2 and g3),
    }


def fmt_pass(b: bool) -> str:
    return "PASS" if b else "FAIL"


def main() -> None:
    print("=" * 78)
    print("Probe 2 Readout — Combo-865 Isolation on Holdout Bars (2024-10-22 → 2026-04-08)")
    print("=" * 78)
    print(f"YEARS_SPAN_TEST = {YEARS_SPAN_TEST}  (frozen per §3 preregistration)")
    print()

    per_tf = {}
    for tf_label, parquet_path in TIMEFRAMES:
        if not parquet_path.exists():
            print(f"[{tf_label}] MISSING: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        m = compute_tf_metrics(df)
        gates = apply_gates(m)
        per_tf[tf_label] = {"metrics": m, "gates": gates}

        print(f"── {tf_label} ──")
        print(f"  n_trades                : {m['n_trades']:>10,}")
        print(f"  net_sharpe              : {m['net_sharpe']:>10.4f}   [gate {GATE_SHARPE}: {fmt_pass(gates['gate_1_sharpe_ge_1.3'])}]")
        print(f"  net_dollars_per_year    : {m['net_dollars_per_year']:>10,.2f}   [gate {GATE_DOLLARS}: {fmt_pass(gates['gate_3_dollars_ge_5000'])}]")
        print(f"  trades                  : {m['n_trades']:>10,}   [gate {GATE_TRADES}: {fmt_pass(gates['gate_2_trades_ge_50'])}]")
        print(f"  mean_net_pnl (per trade): ${m['mean_net_pnl']:>10.4f}")
        print(f"  std_net_pnl  (per trade): ${m['std_net_pnl']:>10.4f}")
        print(f"  total_net_pnl           : ${m['total_net_pnl']:>10,.2f}")
        print(f"  [audit] gross_sharpe    : {m['gross_sharpe']:>10.4f}")
        print(f"  [audit] total_friction  : ${m['total_friction']:>10,.2f}")
        print(f"  [audit] win_rate        : {m['win_rate']:>10.4f}")
        print(f"  ALL THREE GATES         : {fmt_pass(gates['all_three_pass'])}")
        print()

    # Branch decision (§4)
    any_tf_passes = any(r["gates"]["all_three_pass"] for r in per_tf.values())
    passing_tfs = [tf for tf, r in per_tf.items() if r["gates"]["all_three_pass"]]

    print("=" * 78)
    print("BRANCH DECISION (§4)")
    print("=" * 78)
    if any_tf_passes:
        print(f"  VERDICT: PASS — Probe 3 authorized.")
        print(f"  Timeframes clearing all three gates simultaneously: {passing_tfs}")
        print(f"  Next action: fresh LLM Council for Probe 3, then draft")
        print(f"    tasks/probe3_preregistration.md for Option Y session-structure sweep.")
    else:
        print(f"  VERDICT: FAIL — Option Z (project sunset) fires.")
        print(f"  Both timeframes failed at least one gate.")
        print(f"  Next action: draft tasks/project_sunset_verdict.md, update CLAUDE.md")
        print(f"    with a Z-score-family sunset banner, add a lesson to lessons.md.")
    print("=" * 78)

    # Machine-readable record
    out = {
        "probe": "probe2_combo865_isolation",
        "signed_commit": "a49f370",
        "years_span_test": YEARS_SPAN_TEST,
        "gates": {
            "sharpe_floor": GATE_SHARPE,
            "trades_floor": GATE_TRADES,
            "dollars_floor": GATE_DOLLARS,
        },
        "per_timeframe": per_tf,
        "any_timeframe_passes": any_tf_passes,
        "passing_timeframes": passing_tfs,
        "branch": "PASS_probe3" if any_tf_passes else "FAIL_option_z_sunset",
    }
    out_path = Path("data/ml/probe2/readout.json")
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
