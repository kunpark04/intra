"""Probe 3 gate §4.1 — Regime halves on combo 865 1h test partition.

Reads the already-computed Probe 2 1h test parquet; splits trades at
2025-07-15 00:00 UTC; computes per-half {n_trades, net_sharpe,
net_dollars_per_year}; applies the §4.1 composite gate:

  Half h ∈ {H1, H2} PASSES its sub-gate iff
    net_sharpe(h) ≥ 1.3  AND  n_trades(h) ≥ 25  AND  net_$/yr(h) ≥ 5000

  Gate §4.1 PASSES iff BOTH halves pass.

No engine invocation. Runs anywhere (local or remote); expected wall-clock
< 30 s. Writes data/ml/probe3/regime_halves.json.

References:
- tasks/probe3_preregistration.md §3.2.1, §4.1
- tasks/probe3_multiplicity_memo.md §4 (power calibration)
- Probe 2 parquet: data/ml/probe2/combo865_1h_test.parquet
  (signed preregistration commit a49f370; schema per LOG_SCHEMA.md)
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# UTF-8 stdout (Windows cp1252 chokes on any non-ASCII symbol).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Repo root on path for src/ imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import split_train_test

# ── Frozen constants (preregistration §3.1, §4.1, §6.5) ──────────────────────
YEARS_SPAN_TEST   = 1.4799           # frozen at Probe 2 signing
SPLIT_DT_UTC      = pd.Timestamp("2025-07-15 00:00:00")  # §3.2.1 midpoint
SHARPE_GATE       = 1.3
NTRADES_GATE_HALF = 25
DOLLARS_GATE      = 5_000.0

PROBE2_PARQUET = REPO_ROOT / "data" / "ml" / "probe2" / "combo865_1h_test.parquet"
BARS_1H        = REPO_ROOT / "data" / "NQ_1h.parquet"
OUT_PATH       = REPO_ROOT / "data" / "ml" / "probe3" / "regime_halves.json"


def _half_metrics(pnl: np.ndarray, n_bars_half: int, n_bars_total: int) -> dict:
    """Compute {n_trades, net_sharpe, net_dollars_per_year} for one half.

    YEARS_SPAN_HALF is bar-count-proportional: YEARS_SPAN_TEST * (n_h / n_total)
    — respects §3.2.1 "computed from actual bar counts in each half".
    """
    n = int(pnl.size)
    if n < 2:
        return {
            "n_trades": n,
            "net_sharpe": 0.0,
            "net_dollars_per_year": 0.0,
            "years_span_half": 0.0,
            "mean_pnl_dollars": 0.0,
            "std_pnl_dollars": 0.0,
        }
    years_span_half = YEARS_SPAN_TEST * (n_bars_half / n_bars_total)
    mean_pnl = float(pnl.mean())
    std_pnl  = float(pnl.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        net_sharpe = 0.0
    else:
        net_sharpe = (mean_pnl / std_pnl) * math.sqrt(n / years_span_half)
    net_dollars_per_year = mean_pnl * n / years_span_half
    return {
        "n_trades":             n,
        "net_sharpe":           float(net_sharpe),
        "net_dollars_per_year": float(net_dollars_per_year),
        "years_span_half":      float(years_span_half),
        "mean_pnl_dollars":     mean_pnl,
        "std_pnl_dollars":      std_pnl,
    }


def _half_gate(metrics: dict) -> dict:
    """Apply the three sub-gates from §4.1 to one half's metrics."""
    sub = {
        "sharpe_ge_1.3":    metrics["net_sharpe"] >= SHARPE_GATE,
        "n_trades_ge_25":   metrics["n_trades"] >= NTRADES_GATE_HALF,
        "dollars_ge_5000":  metrics["net_dollars_per_year"] >= DOLLARS_GATE,
    }
    return {**sub, "half_pass": all(sub.values())}


def main() -> None:
    print(f"[regime_halves] loading probe 2 parquet: {PROBE2_PARQUET}")
    trades = pd.read_parquet(PROBE2_PARQUET)
    print(f"[regime_halves] total trades: {len(trades)}")

    # Map entry_bar_idx (partition-relative) -> entry_time via 1h bar parquet.
    print(f"[regime_halves] loading 1h bar cache: {BARS_1H}")
    bars = pd.read_parquet(BARS_1H)
    _, test_part = split_train_test(bars, 0.8)
    test_part = test_part.reset_index(drop=True)
    print(f"[regime_halves] test partition: {len(test_part):,} 1h bars  "
          f"range=[{test_part['time'].iloc[0]}, {test_part['time'].iloc[-1]}]")

    # Attach entry_time; drop rows whose entry_bar_idx is out-of-range (should be none).
    idx = trades["entry_bar_idx"].to_numpy()
    if idx.max() >= len(test_part):
        raise RuntimeError(
            f"entry_bar_idx max {idx.max()} >= test_part len {len(test_part)}. "
            f"Partition mismatch — check src.data_loader.split_train_test."
        )
    trades = trades.copy()
    trades["entry_time"] = test_part.loc[idx, "time"].values

    # Split at SPLIT_DT_UTC. Bar times are naive UTC from the CSV; compare naive.
    split_naive = SPLIT_DT_UTC.tz_localize(None) if SPLIT_DT_UTC.tzinfo else SPLIT_DT_UTC
    entry_ts = pd.to_datetime(trades["entry_time"])
    if getattr(entry_ts.dt, "tz", None) is not None:
        entry_ts = entry_ts.dt.tz_convert("UTC").dt.tz_localize(None)
    h1_mask = entry_ts < split_naive
    h2_mask = ~h1_mask

    # Bar-count partition (for YEARS_SPAN_HALF).
    bar_ts = pd.to_datetime(test_part["time"])
    if getattr(bar_ts.dt, "tz", None) is not None:
        bar_ts = bar_ts.dt.tz_convert("UTC").dt.tz_localize(None)
    n_bars_h1 = int((bar_ts < split_naive).sum())
    n_bars_h2 = int((bar_ts >= split_naive).sum())
    n_bars_total = n_bars_h1 + n_bars_h2
    print(f"[regime_halves] bar split H1 / H2 / total = "
          f"{n_bars_h1:,} / {n_bars_h2:,} / {n_bars_total:,}")

    pnl_all = trades["net_pnl_dollars"].to_numpy()
    h1_pnl  = pnl_all[h1_mask.to_numpy()]
    h2_pnl  = pnl_all[h2_mask.to_numpy()]
    print(f"[regime_halves] trade split H1 / H2 = {h1_pnl.size} / {h2_pnl.size}")

    m_h1 = _half_metrics(h1_pnl, n_bars_h1, n_bars_total)
    m_h2 = _half_metrics(h2_pnl, n_bars_h2, n_bars_total)
    g_h1 = _half_gate(m_h1)
    g_h2 = _half_gate(m_h2)

    gate_pass = bool(g_h1["half_pass"] and g_h2["half_pass"])

    readout = {
        "gate_id":   "§4.1_regime_halves",
        "split_dt_utc": str(SPLIT_DT_UTC),
        "years_span_test": YEARS_SPAN_TEST,
        "n_bars_h1": n_bars_h1,
        "n_bars_h2": n_bars_h2,
        "h1": {"metrics": m_h1, "gate": g_h1},
        "h2": {"metrics": m_h2, "gate": g_h2},
        "thresholds": {
            "net_sharpe":           SHARPE_GATE,
            "n_trades_per_half":    NTRADES_GATE_HALF,
            "net_dollars_per_year": DOLLARS_GATE,
        },
        "gate_pass": gate_pass,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(readout, indent=2, default=str))
    print(f"[regime_halves] wrote {OUT_PATH}")
    print(f"[regime_halves] H1: {g_h1}  metrics={m_h1}")
    print(f"[regime_halves] H2: {g_h2}  metrics={m_h2}")
    print(f"[regime_halves] GATE §4.1 -> {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
