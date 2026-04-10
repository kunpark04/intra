"""
reporting.py - Artifact generation for the MNQ 1-minute backtest system.

Functions
---------
write_trades_csv      - ML-ready full trade log (all LOG_SCHEMA fields)
write_trader_log      - Minimal 7-column human-readable trades-only log
write_daily_ledger    - One row per calendar day (including no-trade days)
write_equity_curve    - Bar-by-bar equity curve
write_metadata        - JSON metadata from config + extra fields
run_monte_carlo       - Bootstrap Monte Carlo risk metrics
save_iteration        - Orchestrate all artifact writes for one iteration Vn
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


# ── 1. write_trades_csv ───────────────────────────────────────────────────────

def write_trades_csv(trades: List[Dict[str, Any]], path: Path) -> None:
    """Write the full ML-ready trade list to CSV. All LOG_SCHEMA fields must
    be present (populated by run_backtest)."""
    pd.DataFrame(trades).to_csv(path, index=False)


# ── 2. write_trader_log ───────────────────────────────────────────────────────

def write_trader_log(trades: List[Dict[str, Any]], path: Path) -> None:
    """Write the minimal 7-column human-readable log (trades only, no no-trade
    days). Columns exactly:
      entry_time, side, entry_fill_price, sl_price, tp_price,
      net_pnl_dollars, cumulative_net_pnl_dollars
    """
    rows = []
    cum = 0.0
    for t in trades:
        cum += t["net_pnl_dollars"]
        rows.append({
            "entry_time":                t["entry_time"],
            "side":                      t["side"],
            "entry_fill_price":          t["entry_fill_price"],
            "sl_price":                  t["sl_price"],
            "tp_price":                  t["tp_price"],
            "net_pnl_dollars":           t["net_pnl_dollars"],
            "cumulative_net_pnl_dollars": cum,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ── 3. write_daily_ledger ─────────────────────────────────────────────────────

def write_daily_ledger(
    trades: List[Dict[str, Any]],
    df: pd.DataFrame,
    path: Path,
) -> None:
    """One row per calendar day in the partition's date range (including
    no-trade days). Columns: date, trades_count, pnl_day_dollars, equity_eod.
    """
    # Build full date range from the bar series
    dates = pd.to_datetime(df["time"]).dt.date
    all_dates = pd.date_range(dates.min(), dates.max(), freq="D").date

    # Aggregate trades by entry_date
    if trades:
        trade_df = pd.DataFrame(trades)[["entry_date", "net_pnl_dollars"]]
    else:
        trade_df = pd.DataFrame(columns=["entry_date", "net_pnl_dollars"])

    by_day = (
        trade_df
        .groupby("entry_date")
        .agg(
            trades_count=("net_pnl_dollars", "count"),
            pnl_day_dollars=("net_pnl_dollars", "sum"),
        )
        .reset_index()
        .rename(columns={"entry_date": "date"})
    )

    # Merge with full date range (left join keeps all calendar days)
    ledger = pd.DataFrame({"date": all_dates})
    ledger = ledger.merge(by_day, on="date", how="left")
    ledger["trades_count"] = ledger["trades_count"].fillna(0).astype(int)
    ledger["pnl_day_dollars"] = ledger["pnl_day_dollars"].fillna(0.0)

    # Equity end-of-day: running cumulative from starting equity
    starting_equity = float(trades[0]["equity_before"]) if trades else 2000.0
    ledger["equity_eod"] = starting_equity + ledger["pnl_day_dollars"].cumsum()

    ledger.to_csv(path, index=False)


# ── 4. write_equity_curve ─────────────────────────────────────────────────────

def write_equity_curve(equity_curve: List[Dict[str, Any]], path: Path) -> None:
    """Write bar-by-bar equity curve to CSV."""
    pd.DataFrame(equity_curve).to_csv(path, index=False)


# ── 5. write_metadata ─────────────────────────────────────────────────────────

def write_metadata(cfg, extra: dict, path: Path) -> None:
    """Write a JSON file with all uppercase config parameters plus any extra
    fields. Non-serialisable values are converted via default=str."""
    # Collect all uppercase config vars (skip dunders and callables)
    params = {
        k: v
        for k, v in vars(cfg).items()
        if k.isupper() and not callable(v)
    }

    meta = {**params, **extra}
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ── 6. run_monte_carlo ────────────────────────────────────────────────────────

def run_monte_carlo(trades: List[Dict[str, Any]], cfg) -> dict:
    """Bootstrap Monte Carlo on trade net_pnl_dollars. Seeded and reproducible.

    Draws cfg.MC_N_SIMS simulated equity paths (IID bootstrap), each with the
    same number of trades as the observed sample. Reports max-drawdown
    distribution, VaR/CVaR at trade level, and risk-of-ruin probability.
    """
    if not trades:
        return {
            "n_trades": 0,
            "n_sims":   cfg.MC_N_SIMS,
            "seed":     cfg.MC_SEED,
            "error":    "no trades",
        }

    pnls = np.array([t["net_pnl_dollars"] for t in trades], dtype=np.float64)
    starting_equity = float(trades[0]["equity_before"])
    n = len(pnls)
    rng = np.random.default_rng(cfg.MC_SEED)

    n_sims = cfg.MC_N_SIMS
    max_dds = np.empty(n_sims, dtype=np.float64)
    final_equities = np.empty(n_sims, dtype=np.float64)
    ruin_threshold_dollars = starting_equity * cfg.MC_RUIN_THRESHOLD
    ruin_count = 0

    for i in range(n_sims):
        # IID bootstrap: resample n trades with replacement
        sample = rng.choice(pnls, size=n, replace=True)
        equity_path = starting_equity + np.cumsum(sample)

        # Peak-to-trough max drawdown in dollars
        running_max = np.maximum.accumulate(
            np.concatenate([[starting_equity], equity_path])
        )
        drawdowns = running_max[1:] - equity_path  # positive = drawdown
        max_dd = float(drawdowns.max())
        max_dds[i] = max_dd
        final_equities[i] = float(equity_path[-1])

        if max_dd >= ruin_threshold_dollars:
            ruin_count += 1

    # VaR / CVaR on trade-level PnL (5th percentile horizon = 1 trade)
    var_5 = float(np.percentile(pnls, 5))
    cvar_mask = pnls <= var_5
    cvar = float(pnls[cvar_mask].mean()) if cvar_mask.any() else var_5

    return {
        "version":          getattr(cfg, "_VERSION", "unknown"),
        "n_trades":         n,
        "n_sims":           n_sims,
        "seed":             cfg.MC_SEED,
        "bootstrap_method": cfg.MC_BOOTSTRAP,
        "max_drawdown": {
            "p50":   float(np.percentile(max_dds, 50)),
            "p90":   float(np.percentile(max_dds, 90)),
            "p95":   float(np.percentile(max_dds, 95)),
            "p99":   float(np.percentile(max_dds, 99)),
            "worst": float(max_dds.max()),
        },
        "var_trade_pnl":     var_5,
        "cvar_trade_pnl":    cvar,
        "risk_of_ruin_prob": ruin_count / n_sims,
        "ruin_definition": (
            f"max_drawdown >= {cfg.MC_RUIN_THRESHOLD * 100:.0f}% of starting equity"
            f" (${ruin_threshold_dollars:.0f})"
        ),
        "notes": (
            "IID bootstrap on trade net_pnl_dollars; "
            "horizon = n_trades per path; "
            "VaR/CVaR at 5th percentile trade-level"
        ),
    }


# ── 7. save_iteration ─────────────────────────────────────────────────────────

def save_iteration(version: str, results: dict, df: pd.DataFrame, cfg) -> Path:
    """Orchestrate writing all artifacts for one training iteration Vn.

    Writes:
      trades.csv, trader_log.csv, daily_ledger.csv, equity_curve.csv,
      monte_carlo.json, metadata.json

    Note: plotly_price_indicators.html is NOT written here — handled by
    notebook 02.

    Parameters
    ----------
    version : e.g. "V1"
    results : dict returned by run_backtest()
    df      : the partition DataFrame used for backtesting
    cfg     : config module

    Returns
    -------
    Path to the iteration output directory.
    """
    from src.io_paths import iteration_dir

    out_dir = iteration_dir(version)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = results["trades"]

    write_trades_csv(trades, out_dir / "trades.csv")
    write_trader_log(trades, out_dir / "trader_log.csv")
    write_daily_ledger(trades, df, out_dir / "daily_ledger.csv")
    write_equity_curve(results["equity_curve"], out_dir / "equity_curve.csv")

    # Monte Carlo
    mc = run_monte_carlo(trades, cfg)
    mc["version"] = version  # inject version (overrides any placeholder)
    with open(out_dir / "monte_carlo.json", "w") as f:
        json.dump(mc, f, indent=2, default=str)

    # Metadata
    extra = {
        "version":                version,
        "n_trades":               results["n_trades"],
        "final_equity":           results["final_equity"],
        "data_rows":              len(df),
        "fill_model":             cfg.FILL_MODEL,
        "same_bar_collision":     cfg.SAME_BAR_COLLISION,
        "exit_on_opposite_signal": cfg.EXIT_ON_OPPOSITE_SIGNAL,
        "stop_method":            cfg.STOP_METHOD,
        "bootstrap_method":       cfg.MC_BOOTSTRAP,
    }
    write_metadata(cfg, extra, out_dir / "metadata.json")

    print(f"Saved iteration {version} to {out_dir}")
    return out_dir
