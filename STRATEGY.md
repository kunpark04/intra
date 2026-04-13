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

Current spans (V3+):

- fast EMA length: `EMA_FAST = 10`
- slow EMA length: `EMA_SLOW = 20`

V1/V2 used 8/21. Config is authoritative; always read from `src/config.py`.

- `alpha = 2 / (span + 1)`; warm-up: `ema[0] = close[0]`

Crossover columns computed in `src/indicators/pipeline.py`:
- `ema_cross_up`   — bool: fast crossed **above** slow this bar
- `ema_cross_down` — bool: fast crossed **below** slow this bar

### Z-score

Rolling window length: `Z_WINDOW = 20`.

- `z = (close - rolling_mean(close, N)) / rolling_std(close, N)`
- `ddof = 0`

Bands (used in signal logic; see below):

- upper band: `z >= +k`
- lower band: `z <= -k`

Default: **`k = 2.5`** (`Z_BAND_K` in config).

## Signal modes (`SIGNAL_MODE` config)

The strategy supports two signal modes, selectable via `SIGNAL_MODE` in `src/config.py`.

### `"ema_crossover"` — V1/V2 baseline

Primary signal: EMA crossover fires the entry.

Long signal (`signal = 1`):
- `ema_fast` crosses **above** `ema_slow` (`ema_cross_up == True`)
- Optionally: `zscore <= -Z_BAND_K` when `USE_ZSCORE_FILTER = True`

Short signal (`signal = -1`):
- `ema_fast` crosses **below** `ema_slow` (`ema_cross_down == True`)
- Optionally: `zscore >= +Z_BAND_K` when `USE_ZSCORE_FILTER = True`

**V1**: `USE_ZSCORE_FILTER = False` — every crossover generates a signal.
**V2**: `USE_ZSCORE_FILTER = True`, tighter `STOP_FIXED_PTS = 20`, `MIN_RR = 3.0`.

### `"zscore_reversal"` — V3+ (active default)

Primary signal: **z-score crossing back through the band** (mean-reversion recovery).
Confirming filter: **EMA direction** must agree at the signal bar.

Long signal:
- z-score crosses **up** through the lower band: `z_prev <= -k AND z > -k`
- AND `ema_fast > ema_slow` (bullish regime)

Short signal:
- z-score crosses **down** through the upper band: `z_prev >= k AND z < k`
- AND `ema_fast < ema_slow` (bearish regime)

**Rationale**: EMA crossover is a lagged momentum indicator; by the time it fires, extreme z-scores have already recovered, creating a fundamental incompatibility. Using z-score crossing as the primary trigger with EMA as the regime filter resolves this timing mismatch.

Current config (V3): `EMA_FAST = 10`, `EMA_SLOW = 20`, `Z_BAND_K = 2.5`.

## Exit conditions (whichever hits first)

1. **Stop-loss hit**: `low <= sl_price` (long) or `high >= sl_price` (short)
2. **Take-profit hit**: `high >= tp_price` (long) or `low <= tp_price` (short)
3. **Opposite signal**: opposite signal closes the position at next-bar open (if `EXIT_ON_OPPOSITE_SIGNAL = True`)

## Entries, stops, targets

### Entry price model

**Enter at next bar open** after signal bar confirmation (`FILL_MODEL = "next_bar_open"`).

The logs must store both:

- `entry_signal_price` (the price used for signal evaluation)
- `entry_fill_price` (the actual fill price in the simulation)

### Stop-loss (SL)

`STOP_METHOD` is configurable per iteration.

- **`"fixed"`** (V1/V2/V3 default): `stop_distance = STOP_FIXED_PTS`
  - V1: 30 pts; V2/V3: 20 pts (`STOP_FIXED_PTS = 20`)
  - long: `sl = entry - stop_distance`; short: `sl = entry + stop_distance`
- **`"atr"`**: `stop_distance = ATR * ATR_MULTIPLIER`
- **`"swing"`**: beyond recent swing low/high ± `SWING_BUFFER_TICKS` ticks

All distances tick-rounded to 0.25 points.

### Take-profit (TP)

Minimum target distance:

- `target_distance_points >= MIN_RR * stop_distance_points`

Current: `MIN_RR = 3.0` (V2/V3). V1 used `MIN_RR = 2.0`.

TP price:
- long: `tp = entry + target_distance_points`
- short: `tp = entry - target_distance_points`

### Intrabar SL/TP collision rule

If both SL and TP are touched within the same 1-minute bar:

- `SAME_BAR_COLLISION = "tp_first"` → assume TP hit first (optimistic; documented)
- Records `label_hit_tp_first = 1` in trades.csv.

## Risk management & sizing

### Risk budget

- Starting equity: **$50,000**
- Risk per trade: **5% of equity at entry** (`RISK_PCT = 0.05`)
  - `max_risk_allowed_dollars = equity_before * 0.05`

### Contracts

Let:

- `stop_distance_points` be the planned stop distance in points
- MNQ dollars per point per contract = 2 (`MNQ_DOLLARS_PER_POINT`)

Per-contract risk:

- `risk_per_contract_dollars = stop_distance_points * 2`

Contracts:

- `contracts = floor(max_risk_allowed_dollars / risk_per_contract_dollars)`
- If `contracts < 1`, skip the trade.

### One position at a time

v1/v2/v3: one open position at a time (no pyramiding).

## Track A: Entry quality scoring

Each trade receives an `entry_score` in [0, 1] computed from 5 components. This is **analytical only** (not an ML feature yet). Weights are configurable via `SCORE_W_*` in `src/config.py`.

| Component | Config var | Description |
|-----------|------------|-------------|
| Z-score stretch | `SCORE_W_ZSCORE = 0.25` | `abs(zscore_entry) / Z_BAND_K`, clipped to [0,1] |
| Volume confirmation | `SCORE_W_VOLUME = 0.20` | `(volume_zscore + 2) / 4`, clipped to [0,1] |
| EMA trend strength | `SCORE_W_EMA = 0.20` | `abs(ema_spread) / SCORE_EMA_NORM` |
| Bar body conviction | `SCORE_W_BODY = 0.20` | `bar_body / bar_range` |
| Session quality | `SCORE_W_SESSION = 0.15` | RTH=1.0, extended=0.6, overnight=0.3 |

NaN-safe: missing components are excluded and score is re-normalised by the sum of present weights.

Also tracked per-trade: `mfe_mae_ratio = mfe_points / abs(mae_points)` — measures how cleanly price moved in the trade direction (higher = cleaner entry, less adverse excursion).

## Statistical validation

Monte Carlo bootstrap on trade-level `net_pnl_dollars` (IID resampling, 10,000 sims by default) provides distributional risk estimates per iteration.

A **permutation test on win rate** is embedded in the Monte Carlo output:

- H0: strategy wins at exactly the break-even rate for its average planned R:R (`break_even_wr = 1 / (1 + avg_rr)`)
- H1: win rate > break-even (one-tailed)
- Simulated under H0 via binomial draws; p-value = fraction of simulations ≥ observed win rate
- `significant_05` and `significant_01` flags indicate edge at those significance levels

## Outputs required from strategy layer

At minimum the strategy/backtest pipeline must produce:

- bar-level dataframe with indicators (EMA fast/slow, Z-score, optional ATR)
- trade records (closed trades) suitable for:
  - ML-ready `trades.csv` (see [`LOG_SCHEMA.md`](LOG_SCHEMA.md))
  - simple `trader_log.csv`
- daily ledger including no-trade days

## Parameter sweep (implemented)

`scripts/param_sweep.py` randomly samples parameter combos within predefined
`--range-mode`s and runs the backtest core on each, writing one row per closed
trade to `data/ml/ml_dataset_v{N}.parquet`. Range modes `v4`–`v10` encode
successive data-driven refinements (see [`CLAUDE.md`](CLAUDE.md) for the
version history). The sweep reuses all strategy logic in this file; it only
varies the numeric parameters and the sampled z-score formulation.

## Known future extensions (not yet implemented)

- ML pipeline (Track B): train classifier on `data/ml/ml_dataset_v{N}.parquet` features to predict `label_win` and per-setup optimal R:R
- Live volume bubbles
- DOM/Level2 features
- Execution adapters
