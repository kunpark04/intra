"""Artifact generation for the MNQ 1-minute backtest system.

This module is the single source for all persisted iteration outputs. A
backtest run produces an in-memory `results` dict plus the source DataFrame,
and `save_iteration` fans them out to the canonical files under
`iterations/Vn/`:

- `trades.csv` — ML-ready full log (every `LOG_SCHEMA` field).
- `trader_log.csv` — minimal human-readable 7-column trades-only view.
- `daily_ledger.csv` — one row per calendar day (including no-trade days).
- `equity_curve.csv` — bar-by-bar equity path.
- `monte_carlo.json` — IID bootstrap risk metrics + win-rate permutation test.
- `metadata.json` — config snapshot plus run-specific extras.

The Monte Carlo layer is required by the project contract — do not skip it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


# ── 1. write_trades_csv ───────────────────────────────────────────────────────

def write_trades_csv(trades: List[Dict[str, Any]], path: Path) -> None:
    """Dump the full ML-ready trade list to CSV.

    All `LOG_SCHEMA.md` fields must already be populated by the backtest
    engine — this function does not compute derived columns.

    Args:
        trades: List of per-trade dicts produced by `run_backtest`.
        path: Destination CSV path.
    """
    pd.DataFrame(trades).to_csv(path, index=False)


# ── 2. write_trader_log ───────────────────────────────────────────────────────

def write_trader_log(trades: List[Dict[str, Any]], path: Path) -> None:
    """Write a minimal 7-column human-readable trades-only log.

    Columns (in order): `entry_time`, `side`, `entry_fill_price`, `sl_price`,
    `tp_price`, `net_pnl_dollars`, `cumulative_net_pnl_dollars`. Cumulative
    PnL is computed in traversal order of the input list.

    Args:
        trades: List of per-trade dicts produced by `run_backtest`.
        path: Destination CSV path.
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
    """Write one ledger row per calendar day in the partition's date range.

    No-trade days are preserved so per-day stats (Sharpe, drawdown) see
    the correct denominator. Output columns: `date`, `trades_count`,
    `pnl_day_dollars`, `equity_eod`.

    Args:
        trades: List of per-trade dicts (empty list is valid).
        df: Partition DataFrame used for the backtest — provides the
            calendar range via its `time` column.
        path: Destination CSV path.
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
    """Dump the bar-by-bar equity path returned by `run_backtest` to CSV.

    Args:
        equity_curve: List of per-bar dicts with at least `time` and
            `equity` keys.
        path: Destination CSV path.
    """
    pd.DataFrame(equity_curve).to_csv(path, index=False)


# ── 5. write_metadata ─────────────────────────────────────────────────────────

def write_metadata(cfg, extra: dict, path: Path) -> None:
    """Serialise config parameters + run-specific extras to JSON.

    Captures every upper-case, non-callable attribute on `cfg` (skipping
    dunders and functions), merges in `extra`, and writes with
    `default=str` to stringify non-JSON values (e.g. Timestamps, Paths).

    Args:
        cfg: Config module whose upper-case attributes are captured.
        extra: Extra keys to merge into the output (e.g. `version`,
            `n_trades`, `final_equity`).
        path: Destination JSON path.
    """
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
    """Bootstrap Monte Carlo risk summary on observed trade PnL.

    Draws `cfg.MC_N_SIMS` simulated equity paths via IID bootstrap of
    `net_pnl_dollars` (each path has the same number of trades as the
    observed sample), then reports the max-drawdown distribution
    (p50/p90/p95/p99/worst), single-trade VaR/CVaR at 5%, and
    risk-of-ruin probability (`drawdown ≥ cfg.MC_RUIN_THRESHOLD`).
    A permutation test against the break-even win rate is embedded under
    the `permutation_test` key. Seeded via `cfg.MC_SEED`.

    Args:
        trades: List of per-trade dicts; empty list is handled gracefully.
        cfg: Config exposing `MC_N_SIMS`, `MC_SEED`, `MC_BOOTSTRAP`,
            `MC_RUIN_THRESHOLD`.

    Returns:
        Dict with the summary stats described above. Returns a stub dict
        with an `error` field when there are no trades.
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

    mc_result = {
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

    # Permutation test on win rate
    mc_result["permutation_test"] = _permutation_win_rate_test(
        trades, n_sims=n_sims, seed=cfg.MC_SEED + 1, rng=rng
    )

    return mc_result


def _permutation_win_rate_test(
    trades: List[Dict[str, Any]],
    n_sims: int,
    seed: int,
    rng,
) -> dict:
    """One-sided binomial permutation test against break-even win rate.

    H0: strategy wins at `1 / (1 + avg_RR)` — no edge beyond chance at
    the observed average planned R:R. H1: win rate exceeds break-even.
    Under H0, simulate `n_sims` samples of `n` Bernoulli trials at
    `break_even_wr` and report the fraction that match or exceed the
    observed win rate — that fraction is the one-tailed p-value.

    Args:
        trades: List of per-trade dicts with `label_win` and `rr_planned`.
        n_sims: Number of null simulations to draw.
        seed: Seed recorded in the output (the `rng` drives the draws).
        rng: NumPy `Generator` used for the binomial draws.

    Returns:
        Dict with observed and null statistics plus `significant_05` /
        `significant_01` convenience flags.
    """
    n = len(trades)
    observed_wins = sum(t["label_win"] for t in trades)
    observed_wr   = observed_wins / n

    avg_rr        = float(np.mean([t["rr_planned"] for t in trades]))
    break_even_wr = 1.0 / (1.0 + avg_rr)   # win rate needed to break even at this RR

    # Simulate under null: n Bernoulli(break_even_wr) trials, n_sims times
    null_wins = rng.binomial(n, break_even_wr, size=n_sims)
    null_wrs  = null_wins / n

    p_value = float((null_wrs >= observed_wr).sum() / n_sims)

    return {
        "n_trades":            n,
        "observed_wins":       observed_wins,
        "observed_win_rate":   round(observed_wr, 6),
        "avg_rr_planned":      round(avg_rr, 4),
        "break_even_win_rate": round(break_even_wr, 6),
        "null_win_rate_mean":  round(float(null_wrs.mean()), 6),
        "null_win_rate_p95":   round(float(np.percentile(null_wrs, 95)), 6),
        "p_value_one_tailed":  round(p_value, 6),
        "n_sims":              n_sims,
        "seed":                seed,
        "significant_05":      bool(p_value < 0.05),
        "significant_01":      bool(p_value < 0.01),
        "notes": (
            "H0: strategy wins at break-even rate (1/(1+avg_RR)); "
            "H1: win rate > break-even (one-tailed); "
            "simulated under H0 via binomial draws"
        ),
    }


# ── 7. save_iteration ─────────────────────────────────────────────────────────

def save_iteration(version: str, results: dict, df: pd.DataFrame, cfg) -> Path:
    """Write every iteration artifact for one version `Vn` in one call.

    Creates (or reuses) `iterations/Vn/` and writes `trades.csv`,
    `trader_log.csv`, `daily_ledger.csv`, `equity_curve.csv`,
    `monte_carlo.json`, and `metadata.json`. Monte Carlo results are
    stamped with `version` to override any placeholder from `cfg`.

    Args:
        version: Iteration label, e.g. ``"V1"``.
        results: The dict returned by `run_backtest`, containing at
            least `trades`, `equity_curve`, `n_trades`, `final_equity`.
        df: The partition DataFrame used for the backtest (needed for
            the calendar range in the daily ledger).
        cfg: Config module passed through to Monte Carlo and metadata.

    Returns:
        `Path` to the iteration output directory.
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
