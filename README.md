# MNQ 1-Minute Z-EMA Mean Reversion Bot

A **backtest-first** algorithmic trading system for **MNQ futures** on **1-minute bars**, implementing an explosive mean-reversion strategy: Z-score band stretch + reversal confirmed by a sensitive EMA regime filter.

## Strategy Overview

**Type**: Mean reversion with EMA regime confirmation

**Instrument**: MNQ (Micro Nasdaq Futures) — $2/point per contract, tick = 0.25 pts

**Signal logic**:
- **Long**: Z-score ≤ −k (stretched below band) + reversal tick up + `ema_fast > ema_slow`
- **Short**: Z-score ≥ +k (stretched above band) + reversal tick down + `ema_fast < ema_slow`
- **Entry fill**: next-bar open after signal confirmation

**Indicators**:
| Indicator | Default |
|-----------|---------|
| Z-score window | 20 bars |
| Z-score band k | 1.5–2.5 (configurable) |
| EMA fast | 3 bars |
| EMA slow | 6 bars |

**Risk model**:
- Starting equity: $2,000
- Risk per trade: 5% of equity ($100 at start)
- Minimum R:R: 1:2
- One position at a time

See [`STRATEGY.md`](STRATEGY.md) for the full specification.

## Folder Structure

```
intra/
  data/
    NQ_1min.csv          # raw 1-min bars (not committed — too large)
    NQ_1min_gaps.csv     # session-break reference
  src/
    config.py            # all tunable parameters
    data_loader.py       # CSV → canonical DataFrame
    indicators.py        # Z-score, EMA, ATR
    strategy.py          # signal generation
    risk.py              # position sizing
    backtest.py          # bar-by-bar simulation loop
    reporting.py         # artifact writers
    io_paths.py          # path constants
  scripts/
    run_backtest.py      # CLI entry point
  notebooks/
    01_backtest_and_log.ipynb    # run backtest, write artifacts
    02_plotly_exploration.ipynb  # interactive charts
  iterations/
    V1/                  # training backtest artifacts (one folder per version)
      metadata.json
      trades.csv         # ML-ready, full feature schema
      trader_log.csv     # human-readable summary
      daily_ledger.csv   # includes no-trade days
      equity_curve.csv
      monte_carlo.json   # distributional risk metrics
  evaluation/            # single final hold-out run
    metadata.json        # includes source_iteration: "Vn"
    trades.csv
    trader_log.csv
    daily_ledger.csv
    equity_curve.csv
    monte_carlo.json
  STRATEGY.md            # strategy specification
  LOG_SCHEMA.md          # ML-ready log schema
  CLAUDE.md              # implementation contract for Claude Code
  lessons.md             # failure log and prevention rules
```

## Data

The primary input is `data/NQ_1min.csv` (NQ 1-minute OHLCV bars, ~233 MB — not committed).

Expected columns: `Time`, `Open`, `High`, `Low`, `Latest` (close), `Volume`, `session_break`

The loader normalizes these to: `time`, `open`, `high`, `low`, `close`, `volume`, `session_break`

**Train/test split**: chronological 80/20 — no shuffling. All iteration runs use training bars only; evaluation is a single final hold-out run.

## Output Artifacts

Each training iteration `Vn` produces:

| File | Description |
|------|-------------|
| `trades.csv` | ML-ready trade log with full feature + label schema |
| `trader_log.csv` | Human-readable trades-only summary |
| `daily_ledger.csv` | Calendar coverage including no-trade days |
| `equity_curve.csv` | Equity path per bar |
| `plotly_price_indicators.html` | Interactive price + indicator chart |
| `monte_carlo.json` | Bootstrapped risk metrics (max DD, VaR, CVaR, risk-of-ruin) |
| `metadata.json` | All parameters + data fingerprint |

See [`LOG_SCHEMA.md`](LOG_SCHEMA.md) for exact column definitions and ML leakage rules.

## Monte Carlo Risk Metrics

Each iteration runs a bootstrap Monte Carlo (seeded, reproducible) on trade outcomes to produce distributional estimates:

- Max drawdown distribution (p50, p90, p95, p99)
- VaR / CVaR on trade or daily returns
- Risk-of-ruin probability (equity ≤ ruin threshold)

## Setup

```bash
pip install pandas numpy plotly ipykernel nbformat pyarrow numba
```

Run a backtest (from repo root):

```bash
python scripts/run_backtest.py
```

Or use the notebooks interactively — they must be run from the repo root and write artifacts to `iterations/` automatically.

## Performance Design

- **Parquet** for large intermediate data (bars + indicators, ML datasets)
- **Numba** for performance-critical loops (bar-by-bar simulation, Monte Carlo)
- Pure-Python fallback retained for environments without Numba

## References

- [`STRATEGY.md`](STRATEGY.md) — full signal, entry/exit, and risk specification
- [`LOG_SCHEMA.md`](LOG_SCHEMA.md) — exact column schema for all output logs
- [`CLAUDE.md`](CLAUDE.md) — implementation contract, folder layout, non-negotiable constraints
- [`lessons.md`](lessons.md) — persistent log of tooling failures and prevention rules
