# MNQ 1-Min Z-EMA Bot — V1 Implementation Plan

**Goal**: Build a complete, runnable V1 backtest system that produces all required artifacts in `iterations/V1/`.

**Primary spec sources** (read before each phase):
- `CLAUDE.md` — folder layout, constraints, artifact requirements
- `STRATEGY.md` — signal logic, entry/exit rules, risk sizing
- `LOG_SCHEMA.md` — exact column schemas for all output logs

---

## Phase 0: Documentation Discovery ✅ (completed by orchestrator)

**Sources read**: `CLAUDE.md`, `STRATEGY.md`, `LOG_SCHEMA.md`, `data/NQ_1min.csv` (header)

### Allowed APIs / confirmed facts

- CSV actual columns: `Symbol, Time, Open, High, Low, Latest, Change, %Change, Volume, Open Int, synthetic, session_break`
- Canonical rename map: `Time→time, Open→open, High→high, Low→low, Latest→close, Volume→volume, session_break→session_break`; drop unused columns
- MNQ economics: `$2/point/contract`, `tick=0.25pts=$0.50`
- Z-score: `z = (close - rolling_mean(N)) / rolling_std(N)`, `N=20`, `ddof=0`
- EMA fast=3, slow=6 (configurable)
- Entry fill: **next-bar open** after signal bar
- SL/TP collision rule (same bar): must define and document deterministically
- Contracts: `floor(equity*0.05 / (stop_distance_pts * 2))`, min=1
- Train split: `floor(0.8 * T)` bars; test = remainder
- Monte Carlo: bootstrap trade `r_multiple` / `net_pnl_dollars`, seeded, reproducible
- Required Monte Carlo metrics: `max_drawdown p50/p90/p95/p99`, `var/cvar`, `risk_of_ruin_prob`

### Anti-patterns identified

- Do NOT use `ddof=1` without documenting — pick one and fix it in config
- Do NOT shuffle train/test — chronological split only
- Do NOT run evaluation on training data — single final hold-out only
- Do NOT fill entry at signal-bar close — use next-bar open
- **Swing stop buffer**: 1 tick beyond swing (long: `swing_low - TICK_SIZE`, short: `swing_high + TICK_SIZE`)
- **Volume Z-score window**: 20 bars (same as Z_WINDOW)
- **Synthetic bars**: filter out (`synthetic == True`) before any processing — no signals, no fills

---

## Phase 1: Core Infrastructure

**Files to create**: `src/__init__.py`, `src/io_paths.py`, `src/config.py`, `src/data_loader.py`, `requirements.txt`

### Tasks

1. **`requirements.txt`**
   - `pandas`, `numpy`, `plotly`, `ipykernel`, `nbformat`, `pyarrow`, `numba`

2. **`src/io_paths.py`**
   - Single source of truth for all paths, relative to repo root
   - Functions: `iteration_dir(version: str) -> Path`, `evaluation_dir() -> Path`
   - Constants: `DATA_DIR`, `ITERATIONS_DIR`, `EVALUATION_DIR`

3. **`src/config.py`** — all tunable parameters as a dataclass or dict
   ```python
   # Strategy
   Z_WINDOW = 20
   Z_BAND_K = 2.5
   EMA_FAST = 3
   EMA_SLOW = 6
   ZSCORE_DDOF = 0

   # Execution
   FILL_MODEL = "next_bar_open"
   SAME_BAR_COLLISION = "tp_first"  # or "sl_first" — must be documented

   # Risk
   STARTING_EQUITY = 2000.0
   RISK_PCT = 0.05
   MNQ_DOLLARS_PER_POINT = 2.0
   TICK_SIZE = 0.25
   MIN_RR = 2.0

   # Stop placement
   STOP_METHOD = "swing"     # or "atr"
   SWING_LOOKBACK = 5        # bars for swing high/low
   SWING_BUFFER_TICKS = 1    # ticks beyond swing: long stop = swing_low - 1*TICK_SIZE
   ATR_WINDOW = 14
   ATR_MULTIPLIER = 1.5
   VOLUME_ZSCORE_WINDOW = 20 # same window as Z_WINDOW

   # Train/test
   TRAIN_RATIO = 0.8

   # Monte Carlo
   MC_N_SIMS = 10_000
   MC_SEED = 42
   MC_BOOTSTRAP = "iid"      # or "block"
   MC_RUIN_THRESHOLD = 0.5   # 50% equity drawdown = ruin
   ```

4. **`src/data_loader.py`**
   - `load_bars(csv_path: Path) -> pd.DataFrame`
   - Reads CSV, renames columns to canonical names, drops unused cols
   - Parses `time` as datetime, sorts ascending
   - Returns df with: `time, open, high, low, close, volume, session_break`
   - `split_train_test(df, train_ratio) -> (train_df, test_df)` — chronological, index-based

### Verification

- `python -c "from src.data_loader import load_bars; df = load_bars('data/NQ_1min.csv'); print(df.columns.tolist(), len(df))"`
- Confirm columns are exactly `['time','open','high','low','close','volume','session_break']`
- Confirm train+test row counts sum to total

---

## Phase 2: Indicators

**File to create**: `src/indicators.py`

### Tasks

1. **`compute_zscore(close: np.ndarray, window: int, ddof: int) -> np.ndarray`**
   - Rolling mean + std, return z-score array (NaN for first `window-1` bars)
   - Use NumPy or Numba-friendly loops (avoid pandas inside if using Numba)

2. **`compute_ema(close: np.ndarray, span: int) -> np.ndarray`**
   - Standard EMA with `alpha = 2 / (span + 1)`
   - Warm-up: first value = close[0], then EMA formula

3. **`compute_atr(high, low, close: np.ndarray, window: int) -> np.ndarray`**
   - True range = max(high-low, |high-prev_close|, |low-prev_close|)
   - ATR = rolling mean of TR over `window` bars

4. **`add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame`**
   - Applies all three; adds columns: `zscore`, `ema_fast`, `ema_slow`, `atr`
   - Also adds `ema_spread = ema_fast - ema_slow`

5. **Numba plan**: write core loops with `@numba.njit` where array ops replace pandas rolling. Keep a pandas fallback path behind a `USE_NUMBA` flag in config.

### Verification

- Plot `zscore` on a 100-bar sample — confirm it oscillates around 0
- Confirm `ema_fast` is more reactive than `ema_slow` visually
- Check NaN counts at head match expected warm-up lengths

---

## Phase 3: Strategy Signal Generation + Risk

**Files**: `src/strategy.py`, `src/risk.py`

### Tasks — `src/strategy.py`

1. **`generate_signals(df: pd.DataFrame, cfg) -> pd.DataFrame`**
   - Add columns: `signal` (`1`=long, `-1`=short, `0`=flat), `signal_bar_idx`
   - Long condition (all must be true on bar `t`):
     - `zscore[t] <= -k` (stretch)
     - `zscore[t] - zscore[t-1] > 0` (reversal)
     - `ema_fast[t] > ema_slow[t]` (regime)
   - Short condition: symmetric
   - Only emit signal when flat (no open position state here — backtest handles that)
   - Do NOT look ahead; use only `t` and `t-1` values

### Tasks — `src/risk.py`

1. **`compute_stop_distance(bar, side, df, cfg) -> float`**
   - If `STOP_METHOD == "swing"`: find recent swing high (for shorts) or low (for longs) in last `SWING_LOOKBACK` bars
   - If `STOP_METHOD == "atr"`: `stop_distance = atr[bar] * ATR_MULTIPLIER`
   - Round to nearest tick: `round(dist / TICK_SIZE) * TICK_SIZE`
   - Enforce minimum 1 tick

2. **`compute_contracts(equity, stop_distance_pts, cfg) -> int`**
   - `max_risk = equity * RISK_PCT`
   - `risk_per_contract = stop_distance_pts * MNQ_DOLLARS_PER_POINT`
   - `contracts = floor(max_risk / risk_per_contract)`
   - Return 0 if contracts < 1 (trade skipped)

3. **`compute_tp(entry, stop_distance_pts, side, cfg) -> float`**
   - `target_distance = stop_distance_pts * MIN_RR`
   - Long: `tp = entry + target_distance`; Short: `tp = entry - target_distance`
   - Round to tick

### Verification

- Unit test: at equity=2000, stop=4pts → risk_per_contract=$8 → contracts=floor(100/8)=12
- Confirm signals never fire on NaN indicator bars (warm-up guard)

---

## Phase 4: Backtest Engine

**File**: `src/backtest.py`

### Tasks

1. **`run_backtest(df: pd.DataFrame, cfg, version: str) -> dict`**
   - Returns `{"trades": [...], "equity_curve": [...], "metadata": {...}}`

2. **Bar-by-bar loop** (Numba-friendly; operate on NumPy arrays):
   ```
   state: flat | in_trade
   for each bar t:
     if flat:
       if signal[t] in {1,-1}:
         entry_fill = open[t+1]   # next-bar fill
         compute SL, TP, contracts
         if contracts >= 1:
           open position record
     if in_trade:
       check if high[t] >= tp (long) or low[t] <= tp (long) → TP hit
       check if low[t] <= sl (long) or high[t] >= sl (short) → SL hit
       same-bar collision: per SAME_BAR_COLLISION config
       if exit: record trade, update equity, go flat
   ```

3. **Trade record** matches full `LOG_SCHEMA.md` ML schema — all columns populated

4. **Equity curve**: one row per bar with `bar_idx, time, equity`

5. **Deterministic collision rule** (same bar, both SL and TP touched):
   - `"tp_first"`: assume TP hit → `label_hit_tp_first = True`
   - `"sl_first"`: assume SL hit → `label_hit_tp_first = False`
   - Document chosen rule in `metadata.json`

6. **MAE/MFE**: track intrabar adverse/favorable excursion during hold

### Verification

- Run on a 200-bar toy slice and manually verify 2–3 trades match expected PnL
- Confirm `equity_after` monotonically follows cumulative net PnL
- Confirm no trades opened during NaN indicator warm-up

---

## Phase 5: Reporting + Monte Carlo

**File**: `src/reporting.py`

### Tasks

1. **`write_trades_csv(trades, path)`** — ML-ready, all LOG_SCHEMA columns
2. **`write_trader_log(trades, path)`** — minimal 7-column human summary
3. **`write_daily_ledger(trades, df, path)`** — one row per calendar day, 0-fill no-trade days
4. **`write_equity_curve(equity_curve, path)`**
5. **`write_metadata(cfg, meta, path)`** — all params + data fingerprint (row count)

6. **`run_monte_carlo(trades, cfg) -> dict`**
   - Extract `r_multiple` or `net_pnl_dollars` per trade
   - Bootstrap `MC_N_SIMS` paths with `np.random.default_rng(MC_SEED)`
   - Per path: compute equity curve, max drawdown, final equity
   - Report:
     - `max_drawdown`: p50, p90, p95, p99, worst
     - `var` (5th percentile of trade PnL distribution)
     - `cvar` (mean of trade PnLs below VaR threshold)
     - `risk_of_ruin_prob`: fraction of paths where equity drops by `MC_RUIN_THRESHOLD`
   - Return dict matching `monte_carlo.json` schema in LOG_SCHEMA.md

7. **`save_iteration(version, trades, equity_curve, daily_ledger, mc_results, cfg, df)`**
   - Orchestrates writing all artifacts to `iterations/Vn/`

### Verification

- `monte_carlo.json` has all required keys
- `daily_ledger.csv` has one row per calendar day in the partition range
- `trader_log.csv` has exactly N rows matching trade count

---

## Phase 6: Scripts + Notebooks

### Tasks

1. **`scripts/run_backtest.py`**
   ```python
   # Usage: python scripts/run_backtest.py --version V1
   # Runs full pipeline: load → indicators → signals → backtest → reporting
   # Writes all artifacts to iterations/V1/
   ```
   - `argparse` with `--version` (default `V1`), `--data` (default `data/NQ_1min.csv`)
   - Runs on training partition only

2. **`notebooks/01_backtest_and_log.ipynb`**
   - Cells: load config → load data → add indicators → run backtest → write artifacts
   - Win/loss row shading on trader_log display (green/red)
   - Must run from repo root without path edits

3. **`notebooks/02_plotly_exploration.ipynb`**
   - Candlestick (or close line) + EMA fast/slow overlaid
   - Z-score subplot with `±k` horizontal lines
   - Trade entry/exit markers on price chart
   - `rangeslider=True`
   - Saves `iterations/Vn/plotly_price_indicators.html`

### Verification

- `python scripts/run_backtest.py --version V1` completes without error
- `iterations/V1/` contains all 7 required artifacts
- HTML chart opens in browser and is interactive

---

## Phase 7: Final Verification

### Checklist

- [ ] All 7 iteration artifacts exist in `iterations/V1/`
- [ ] `monte_carlo.json` has: `n_trades`, `n_sims`, `seed`, `max_drawdown.p50/p90/p95/p99`, `var`, `cvar`, `risk_of_ruin_prob`
- [ ] `trades.csv` has all columns from LOG_SCHEMA.md (no missing fields)
- [ ] `daily_ledger.csv` covers every calendar day in training partition
- [ ] `metadata.json` has: version, all params, train split indices, data fingerprint, `SAME_BAR_COLLISION` rule, `bootstrap_method`
- [ ] No trade opened during indicator warm-up bars (first `max(Z_WINDOW, EMA_SLOW)` bars)
- [ ] `contracts >= 1` for every logged trade
- [ ] `rr_planned >= 2.0` for every logged trade
- [ ] Plotly HTML renders correctly
- [ ] `run_backtest.py --version V2` creates `iterations/V2/` without touching V1

### Anti-pattern grep checks

```bash
grep -r "shuffle" src/          # must be 0 results
grep -r "test_df" src/backtest  # must be 0 (backtest only uses train)
grep -r "iloc\[0\]" src/        # watch for accidental lookahead
```

---

## Execution Order Summary

| Phase | Files Created | Est. Complexity |
|-------|--------------|-----------------|
| 1 | `requirements.txt`, `src/__init__.py`, `src/io_paths.py`, `src/config.py`, `src/data_loader.py` | Low |
| 2 | `src/indicators.py` | Medium |
| 3 | `src/strategy.py`, `src/risk.py` | Medium |
| 4 | `src/backtest.py` | High |
| 5 | `src/reporting.py` | Medium |
| 6 | `scripts/run_backtest.py`, `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb` | Medium |
| 7 | Verification only | — |

Each phase is self-contained and can be executed in a fresh chat context by providing the relevant spec files (`CLAUDE.md`, `STRATEGY.md`, `LOG_SCHEMA.md`) and this plan.
