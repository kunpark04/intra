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

### EMAs

Two EMAs on close. **Each indicator lives in its own file** under `src/indicators/`.

- fast EMA length: `ema_fast = 8`
- slow EMA length: `ema_slow = 21`
- `alpha = 2 / (span + 1)`; warm-up: `ema[0] = close[0]`

Crossover columns computed in `src/indicators/pipeline.py`:
- `ema_cross_up`   — bool: fast crossed **above** slow this bar
- `ema_cross_down` — bool: fast crossed **below** slow this bar

### Z-score (optional filter)

Rolling window length: `z_window_N = 20`.

- `z = (close - rolling_mean(close, N)) / rolling_std(close, N)`
- `ddof = 0`; fixed for V1

Bands (used only when `USE_ZSCORE_FILTER = True`):

- upper band: `z >= +k`
- lower band: `z <= -k`

where **default `k = 2.5`**.

**Baseline (V1)**: `USE_ZSCORE_FILTER = False` — trade every EMA crossover.

## Signal logic (v1 default)

Strategy type: **EMA crossover trend-following** with optional Z-score filter.

### Primary signal: EMA crossover

Long signal (`signal = 1`):
- `ema_fast` crosses **above** `ema_slow` (`ema_cross_up == True`)
- Optionally: `zscore <= -Z_BAND_K` when `USE_ZSCORE_FILTER = True`

Short signal (`signal = -1`):
- `ema_fast` crosses **below** `ema_slow` (`ema_cross_down == True`)
- Optionally: `zscore >= +Z_BAND_K` when `USE_ZSCORE_FILTER = True`

**Baseline V1**: `USE_ZSCORE_FILTER = False` — every crossover generates a signal.

### Exit conditions (whichever hits first)

1. **Stop-loss hit**: `low <= sl_price` (long) or `high >= sl_price` (short)
2. **Take-profit hit**: `high >= tp_price` (long) or `low <= tp_price` (short)
3. **Opposite signal**: opposite EMA crossover closes the position at next-bar open

## Entries, stops, targets

### Entry price model

Choose **one** deterministic fill model for backtests and document it:

- Recommended: **enter at next bar open** after signal confirmation.

The logs must store both:

- `entry_signal_price` (the price used for signal evaluation)
- `entry_fill_price` (the actual fill price in the simulation)

### Stop-loss (SL)

`STOP_METHOD` is configurable per iteration. V1 baseline: **fixed**.

- **`"fixed"`** (V1 default): `stop_distance = STOP_FIXED_PTS = 30` points
  - long: `sl = entry - 30`; short: `sl = entry + 30`
- **`"atr"`**: `stop_distance = ATR * ATR_MULTIPLIER`
- **`"swing"`**: beyond recent swing low/high ± `SWING_BUFFER_TICKS` ticks

All distances tick-rounded to 0.25 points.

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

