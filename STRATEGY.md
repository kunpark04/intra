## Purpose

Defines the **trading strategy specification** for the MNQ 1-minute bot: indicators, signals, confirmation logic, entries/exits, and risk sizing rules.

Implementation details (folder layout, artifacts, logging, notebooks) live in [`CLAUDE.md`](CLAUDE.md).

## Instrument & economics

- Price path: provided dataset is **NQ** 1-minute bars.
- PnL + sizing: use **MNQ** economics:
  - **$2 per full index point per contract**
  - tick size: 0.25 points → $0.50

## Data conventions

From CSV:

- `Latest` is treated as **close**.
- Canonical bar fields: `time`, `open`, `high`, `low`, `close`, `volume`, `session_break`.

## Indicators

All computed on **close** unless explicitly stated.

### Z-score

Rolling window length (v1 fixed): `z_window_N = 20`.

- `z = (close - rolling_mean(close, N)) / rolling_std(close, N)`
- `ddof`: specify and keep consistent (`0` or `1`); must be documented in config.

Bands:

- upper band: `z >= +k`
- lower band: `z <= -k`

where `k` is configurable (suggested 1.5–2.5).

### EMAs (sensitive)

Two EMAs on close (v1 fixed):

- fast EMA length: `ema_fast = 3`
- slow EMA length: `ema_slow = 6`

Example pairs: 5/13 or 3/8 (configurable).

## Signal logic (v1 default)

Strategy type: **mean reversion** with **EMA regime confirmation**.

### Primary mean-reversion condition

Long setup:

- stretch condition: `z <= -k` on the signal bar or recent bar (implementation can use a small state machine)
- reversal condition (must choose a precise rule and keep it fixed per version):
  - minimum default: `zscore_delta = z_t - z_{t-1} > 0`
  - optional add: `close_t > close_{t-1}`

Short setup: symmetric:

- stretch: `z >= +k`
- reversal: `zscore_delta < 0` (and optionally `close_t < close_{t-1}`)

### EMA confirmation (regime filter)

Long allowed only if:

- `ema_fast > ema_slow` at entry (or at signal bar; define precisely)

Short allowed only if:

- `ema_fast < ema_slow`

Note: “EMA cross” can be implemented either as a strict inequality regime filter or as “cross within last m bars.” For v1, use the regime filter unless explicitly changed in a new iteration.

## Entries, stops, targets

### Entry price model

Choose **one** deterministic fill model for backtests and document it:

- Recommended: **enter at next bar open** after signal confirmation.

The logs must store both:

- `entry_signal_price` (the price used for signal evaluation)
- `entry_fill_price` (the actual fill price in the simulation)

### Stop-loss (SL)

Stop placement method is configurable, but must translate to a stop distance in points:

Examples allowed:

- swing-based: beyond recent swing high/low (define lookback)
- volatility-based: `ATR * multiplier`
- fixed points

Also enforce tick rounding to 0.25 points.

### Take-profit (TP)

Minimum target distance:

- `target_distance_points >= 2 * stop_distance_points`

TP price is derived from entry and side:

- long: `tp = entry + target_distance_points`
- short: `tp = entry - target_distance_points`

### Intrabar SL/TP collision rule

If both SL and TP are touched within the same 1-minute bar, the backtester must use a deterministic rule (and log it):

- record `label_hit_tp_first` per the chosen rule.

## Risk management & sizing

### Risk budget

- Starting equity: **$2000**
- Risk per trade: **5% of equity at entry**
  - `max_risk_allowed_dollars = equity_before * 0.05`

### Contracts

Let:

- `stop_distance_points` be the planned stop distance in points
- MNQ dollars per point per contract = 2

Per-contract risk:

- `risk_per_contract_dollars = stop_distance_points * 2`

Contracts:

- `contracts = floor(max_risk_allowed_dollars / risk_per_contract_dollars)`
- If `contracts < 1`, skip the trade.

### One position at a time

v1 assumes one open position at a time (no pyramiding).

## Outputs required from strategy layer

At minimum the strategy/backtest pipeline must be able to produce:

- bar-level dataframe with indicators (EMA fast/slow, Z-score, optional ATR)
- trade records (closed trades) suitable for:
  - ML-ready `trades.csv` (see [`LOG_SCHEMA.md`](LOG_SCHEMA.md))
  - simple `trader_log.csv`
- daily ledger including no-trade days

## Known future extensions (not in v1)

- Live volume bubbles
- DOM/Level2 features
- Execution adapters

