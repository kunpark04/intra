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


# ── 6. Monte Carlo primitives (shared with notebook) ──────────────────────────

STARTING_EQUITY_DEFAULT = 50_000.0
RISK_FRAC_DEFAULT = 0.05


def apply_sizing(
    pnl_base,
    risk_base,
    policy: str,
    equity0: float = STARTING_EQUITY_DEFAULT,
    risk_frac: float = RISK_FRAC_DEFAULT,
) -> np.ndarray:
    """Return per-trade dollar PnL (ordered) under the named sizing policy.

    `pnl_base` and `risk_base` are the $500-fixed baseline series
    (`actual_pnl`, `dollar_risk`). The policy-invariant r-multiple is
    `r_t = pnl_base[t] / risk_base[t]`.

    - `fixed_dollars_500`: returns `pnl_base` unchanged.
    - `pct5_compound`: equity compounds as
      `eq_{t+1} = eq_t * (1 + risk_frac * r_t)` from `equity0`, and
      per-trade PnL is the diff of that equity path.
    """
    pnl = np.asarray(pnl_base, dtype=float)
    risk = np.asarray(risk_base, dtype=float)
    if policy == "fixed_dollars_500":
        return pnl
    if policy != "pct5_compound":
        raise ValueError(f"unknown policy: {policy}")
    r = np.where(risk > 0, pnl / risk, 0.0)
    equity = equity0 * np.cumprod(1.0 + risk_frac * r)
    return np.diff(np.concatenate([[equity0], equity]))


def mc_policy_samples(
    pnl_base,
    risk_base,
    policy: str,
    n_sims: int = 10_000,
    seed: int = 42,
    equity0: float = STARTING_EQUITY_DEFAULT,
    risk_frac: float = RISK_FRAC_DEFAULT,
    dtype=np.float32,
):
    """Vectorized IID bootstrap of trade order under `policy`.

    Bootstrap matrices default to float32 (samples_pnl, equity_paths) and
    int32 (idx). This halves memory vs float64/int64 defaults: a 10k × 24k
    matrix drops from ~2 GB to ~1 GB. Precision is still ~7 significant
    digits — ample for Sharpe/DD/VaR quantile statistics at 4-decimal output.
    Pass dtype=np.float64 to restore the old behaviour.

    Returns:
        samples_pnl (n_sims, n): per-trade $PnL under the policy.
        equity_paths (n_sims, n+1): equity paths with column 0 = equity0.
    """
    rng = np.random.default_rng(seed)
    # Cast source arrays to the bootstrap dtype (float32 by default) so that
    # fancy-indexed intermediates like `pnl[idx]` — shape (n_sims, n) — don't
    # get materialised in float64 before the .astype downcast. At 10k × 44k
    # trades that hidden intermediate is 3.5 GB vs 1.75 GB at float32.
    pnl = np.asarray(pnl_base, dtype=dtype)
    risk = np.asarray(risk_base, dtype=dtype)
    n = len(pnl)
    if n == 0:
        return np.empty((0, 0), dtype=dtype), np.empty((0, 0), dtype=dtype)
    idx = rng.integers(0, n, size=(n_sims, n), dtype=np.int32)
    if policy == "fixed_dollars_500":
        samples_pnl = pnl[idx]  # already dtype (float32 default)
    elif policy == "pct5_compound":
        r = np.where(risk > 0, pnl / risk, 0.0)
        growth = 1.0 + risk_frac * r[idx]
        equity = equity0 * np.cumprod(growth, axis=1)
        samples_pnl = np.diff(
            np.concatenate([np.full((n_sims, 1), equity0), equity], axis=1)
        ).astype(dtype, copy=False)
    else:
        raise ValueError(f"unknown policy: {policy}")
    equity_paths = np.concatenate(
        [np.full((n_sims, 1), equity0, dtype=dtype),
         (equity0 + np.cumsum(samples_pnl, axis=1)).astype(dtype, copy=False)],
        axis=1,
    )
    return samples_pnl, equity_paths


def monte_carlo(
    pnl_base,
    risk_base=None,
    policy: str = "fixed_dollars_500",
    years_span: float | None = None,
    start_equity: float = STARTING_EQUITY_DEFAULT,
    risk_frac: float = RISK_FRAC_DEFAULT,
    n_sims: int = 10_000,
    seed: int = 42,
    ruin_threshold: float = 0.5,
    ruin_basis: str = "peak",
    var_source: str = "bootstrap",
) -> dict:
    """Unified policy-aware IID bootstrap Monte Carlo summary.

    - Bootstraps `n_sims` paths of trade order via `mc_policy_samples`.
    - Reports max-drawdown distribution in both % and $, VaR/CVaR on the
      per-trade $PnL under the policy, risk-of-ruin probability
      (dd_% ≥ `ruin_threshold`*100), and — when `years_span` is given —
      annualized Sharpe p50 + 95% CI.
    - Under `pct5_compound`, the bootstrap Sharpe is computed on per-trade
      log-returns `log1p(risk_frac * r)`, which is scale-invariant under
      compounding. Under `fixed_dollars_500`, it's on $PnL directly.

    Args:
        pnl_base: Per-trade $PnL at the $500-fixed baseline sizing.
        risk_base: Per-trade $-at-risk at $500-fixed. Required under
            `pct5_compound`; may be None or ones under `fixed_dollars_500`.
        policy: 'fixed_dollars_500' or 'pct5_compound'.
        years_span: Span of the bar series in years (drives `trades_per_year`
            and Sharpe annualization). If None, Sharpe CI is omitted.
    """
    if policy == "pct5_compound" and risk_base is None:
        raise ValueError("pct5_compound requires risk_base (r-multiples undefined without it)")
    pnl = np.asarray(pnl_base, dtype=float)
    risk = np.ones_like(pnl) if risk_base is None else np.asarray(risk_base, dtype=float)
    n = len(pnl)
    if n == 0:
        return {"n_sims": n_sims, "n_trades": 0, "note": "empty"}

    samples_pnl, equity_paths = mc_policy_samples(
        pnl, risk, policy,
        n_sims=n_sims, seed=seed, equity0=start_equity, risk_frac=risk_frac,
    )
    peak = np.maximum.accumulate(equity_paths, axis=1)
    dd_dollars = (peak - equity_paths).max(axis=1)
    dd_pct = np.nan_to_num((peak - equity_paths) / peak, nan=0.0).max(axis=1) * 100

    wins = (samples_pnl > 0).mean(axis=1)
    wr_ci = (float(np.percentile(wins, 2.5)), float(np.percentile(wins, 97.5)))

    if var_source == "observed":
        var5 = float(np.percentile(pnl, 5))
        tail_obs = pnl[pnl <= var5]
        cvar = float(tail_obs.mean()) if tail_obs.size else var5
    elif var_source == "bootstrap":
        var5 = float(np.percentile(samples_pnl.reshape(-1), 5))
        tail = samples_pnl[samples_pnl <= var5]
        cvar = float(tail.mean()) if tail.size else var5
    else:
        raise ValueError(f"unknown var_source: {var_source}")

    if ruin_basis == "peak":
        ruin_prob = float((dd_pct >= ruin_threshold * 100).mean())
    elif ruin_basis == "start_equity":
        ruin_prob = float((dd_dollars >= ruin_threshold * start_equity).mean())
    else:
        raise ValueError(f"unknown ruin_basis: {ruin_basis}")

    out = {
        "n_sims": n_sims, "n_trades": int(n), "policy": policy,
        "win_rate": round(float(wins.mean()), 4),
        "wr_ci_95": (round(wr_ci[0], 4), round(wr_ci[1], 4)),
        "dd_p50_pct": round(float(np.percentile(dd_pct, 50)), 2),
        "dd_p90_pct": round(float(np.percentile(dd_pct, 90)), 2),
        "dd_p95_pct": round(float(np.percentile(dd_pct, 95)), 2),
        "dd_p99_pct": round(float(np.percentile(dd_pct, 99)), 2),
        "dd_worst_pct": round(float(dd_pct.max()), 2),
        "dd_p50_dollars": round(float(np.percentile(dd_dollars, 50)), 2),
        "dd_p90_dollars": round(float(np.percentile(dd_dollars, 90)), 2),
        "dd_p95_dollars": round(float(np.percentile(dd_dollars, 95)), 2),
        "dd_p99_dollars": round(float(np.percentile(dd_dollars, 99)), 2),
        "dd_worst_dollars": round(float(dd_dollars.max()), 2),
        "var_5pct_trade": round(var5, 2),
        "cvar_5pct_trade": round(cvar, 2),
        "risk_of_ruin_prob": round(ruin_prob, 6),
    }

    if years_span is not None and years_span > 0:
        tpy = n / years_span
        if policy == "pct5_compound":
            r_full = np.where(risk > 0, pnl / risk, 0.0)
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n, size=(n_sims, n), dtype=np.int32)
            sim_vals = np.log1p(risk_frac * r_full[idx]).astype(np.float32, copy=False)
        else:
            sim_vals = samples_pnl
        mu = sim_vals.mean(axis=1)
        sig = sim_vals.std(axis=1, ddof=1) if n > 1 else np.zeros(n_sims)
        sharpe_boot = np.where(sig > 0, (mu / sig) * np.sqrt(tpy), 0.0)
        out["trades_per_year"] = round(tpy, 1)
        out["sharpe_p50"] = round(float(np.percentile(sharpe_boot, 50)), 4)
        out["sharpe_ci_95"] = (
            round(float(np.percentile(sharpe_boot, 2.5)), 4),
            round(float(np.percentile(sharpe_boot, 97.5)), 4),
        )
        out["sharpe_pos_prob"] = round(float((sharpe_boot > 0).mean()), 4)
    return out


def run_monte_carlo(trades: List[Dict[str, Any]], cfg) -> dict:
    """Legacy iteration-artifact MC wrapper.

    Extracts per-trade $PnL from `trades`, delegates to the unified
    vectorized `monte_carlo` under `fixed_dollars_500` (equivalent to the
    prior behavior), and reshapes the output back to the nested
    `iterations/Vn/monte_carlo.json` schema. Adds the win-rate
    permutation test under `permutation_test`.
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

    summary = monte_carlo(
        pnl_base=pnls, risk_base=None, policy="fixed_dollars_500",
        years_span=None, start_equity=starting_equity,
        n_sims=cfg.MC_N_SIMS, seed=cfg.MC_SEED,
        ruin_threshold=cfg.MC_RUIN_THRESHOLD,
        ruin_basis="start_equity",  # legacy artifact convention
        var_source="observed",       # legacy VaR/CVaR on observed pnls, not bootstrap
    )
    ruin_threshold_dollars = starting_equity * cfg.MC_RUIN_THRESHOLD

    mc_result = {
        "version":          getattr(cfg, "_VERSION", "unknown"),
        "n_trades":         summary["n_trades"],
        "n_sims":           summary["n_sims"],
        "seed":             cfg.MC_SEED,
        "bootstrap_method": cfg.MC_BOOTSTRAP,
        "max_drawdown": {
            "p50":   summary["dd_p50_dollars"],
            "p90":   summary["dd_p90_dollars"],
            "p95":   summary["dd_p95_dollars"],
            "p99":   summary["dd_p99_dollars"],
            "worst": summary["dd_worst_dollars"],
        },
        "var_trade_pnl":     summary["var_5pct_trade"],
        "cvar_trade_pnl":    summary["cvar_5pct_trade"],
        "risk_of_ruin_prob": summary["risk_of_ruin_prob"],
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

    # Permutation RNG matches legacy behavior: advance a generator seeded
    # with MC_SEED past the same N_SIMS*n draws the bootstrap would have
    # consumed, then hand it to the permutation test. Preserves byte-for-byte
    # reproducibility with the pre-refactor loop-based run_monte_carlo.
    rng = np.random.default_rng(cfg.MC_SEED)
    _ = rng.choice(pnls, size=(cfg.MC_N_SIMS, len(pnls)), replace=True)
    mc_result["permutation_test"] = _permutation_win_rate_test(
        trades, n_sims=cfg.MC_N_SIMS, seed=cfg.MC_SEED + 1, rng=rng,
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
