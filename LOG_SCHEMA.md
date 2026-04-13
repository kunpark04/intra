## Purpose

Defines the **exact output schema** for generated logs, with emphasis on:

- ML training compatibility (feature/label separation to avoid leakage)
- deterministic backtesting auditability
- consistent naming across `iterations/Vn/` and `evaluation/`

Folder and artifact requirements are in [`CLAUDE.md`](CLAUDE.md).

## Logs produced

Per training iteration version `Vn`:

- `iterations/Vn/trades.csv` (ML-ready)
- `iterations/Vn/trader_log.csv` (simple)
- `iterations/Vn/daily_ledger.csv` (includes no-trade days)
- `iterations/Vn/monte_carlo.json` (Monte Carlo risk summary + permutation test)
- `iterations/Vn/analysis.ipynb` (7-cell analysis notebook)

Final hold-out evaluation:

- `evaluation/trades.csv`
- `evaluation/trader_log.csv`
- `evaluation/daily_ledger.csv`
- `evaluation/monte_carlo.json` (optional; same schema)

Parameter sweep (Track-B training data):

- `data/ml/ml_dataset_v{N}.parquet` — one row per closed trade across all sampled combos
- `data/ml/ml_dataset_v{N}_manifest.json` — per-combo status (enables resume)

See [§5](#5-parameter-sweep-ml-dataset-ml_dataset_vnparquet) for its schema.

## 1) ML-ready trade log (`trades.csv`)

### Core identity + timing

- `trade_id`
- `version` (e.g. `V3`)
- `source_iteration` (always set; also stored in metadata)
- `symbol`
- `entry_time`, `exit_time`, `entry_date`
- `day_of_week`, `time_of_day_hhmm`
- `signal_bar_index`, `entry_bar_index`, `exit_bar_index`
- `session_break_flag`

### Direction + execution

- `side` (`long`/`short`)
- `entry_signal_price`, `entry_fill_price`
- `exit_signal_price`, `exit_fill_price`
- `slippage_entry_points`, `slippage_exit_points`, `slippage_total_points`
- `spread_at_entry_points` (placeholder if not modeled yet)
- `exit_reason` (`stop`, `take_profit`, `end_of_data`, later `time_stop`/`manual`)

### Risk + sizing

- `sl_price`, `tp_price`
- `stop_distance_points`, `target_distance_points`
- `rr_planned`
- `contracts`
- `position_notional`
- `risk_dollars`
- `max_risk_allowed_dollars`
- `risk_utilization`
- `commission_dollars`, `fees_dollars`

### PnL + path metrics

- `gross_pnl_dollars`, `net_pnl_dollars`
- `r_multiple`
- `hold_bars`, `hold_minutes`
- `mae_points`, `mfe_points`
- `mae_dollars`, `mfe_dollars`
- `mfe_mae_ratio` — `mfe_points / abs(mae_points)`; NaN when `abs(mae_points) < 0.25`; higher = cleaner entry (Track A diagnostic)
- `equity_before`, `equity_after`

### Signal-context features at entry (for ML)

- `zscore_entry`, `zscore_prev`, `zscore_delta`
- `z_band_k` (the band threshold `k` active at trade entry — feature for ML)
- `ema_fast`, `ema_slow`, `ema_spread`
- `close_price`, `open_price`, `high_price`, `low_price`
- `bar_range_points`, `bar_body_points`
- `volume`, `volume_zscore`
- `atr_points`
- `distance_to_ema_fast_points`, `distance_to_ema_slow_points`

### Track A: Entry quality

- `entry_score` — composite entry quality score in [0, 1]; 5 components (z-score stretch, volume, EMA spread, bar body ratio, session quality); NaN-safe, weight-normalised. **Post-trade diagnostic only** (not a feature for ML; computed at entry bar but depends on scoring weights which may change).

### Labeling + QA flags

- `label_win`
- `label_hit_tp_first`
- `data_quality_flag`
- `rule_violation_flag`

### Leakage rule (critical)

When preparing ML datasets:

- **Features**: only columns known at decision time (entry bar close / next open).
- **Labels**: outcomes (`net_pnl_dollars`, `r_multiple`, `label_win`) and post-trade diagnostics (`mae/mfe`, `mfe_mae_ratio`, `entry_score`) must not be used as input features.

Recommended follow-on artifact (optional):

- `trades_schema.json` mapping each column to:
  - dtype
  - `feature` vs `label` vs `post_trade_diagnostic`

## 2) Trader log (simple) (`trader_log.csv`)

Purpose: a minimal, human-readable trades-only list (no no-trade days).

One row per closed trade, columns:

- `entry_time`
- `side`
- `entry_fill_price`
- `sl_price`
- `tp_price`
- `net_pnl_dollars`
- `cumulative_net_pnl_dollars`

Styling rule in notebooks:

- win rows green, loss rows red, 0 neutral.

## 3) Daily ledger (`daily_ledger.csv`)

Purpose: calendar coverage, including **no-trade days** (one row per day in the partition's date range).

Minimum fields (final set can be extended):

- `date`
- `trades_count`
- `pnl_day_dollars` (0 if none)
- `equity_eod`
- optional: `max_drawdown_to_date`

Trade-specific columns may be blank/NaN on no-trade days (by design).

## Metadata (`metadata.json`)

Each `iterations/Vn/metadata.json` should include:

- version id (`Vn`) and optional `name`/description
- all strategy parameters (EMA lengths, z window, k, stop method, fill model, commission/slippage assumptions)
- Monte Carlo configuration (n_sims, seed, bootstrap method)
- train/test split indices and date ranges
- data file used and a simple fingerprint (row count; optional hash)

`evaluation/metadata.json` should include:

- `source_iteration: "Vn"` (the refined version being evaluated)
- identical parameters to the iteration version being tested

## Monte Carlo summary (`monte_carlo.json`)

Purpose: robust distributional risk estimates for a given iteration or final evaluation.

Minimum fields (can extend):

- `version` (e.g. `V3`) and `source_iteration` (for evaluation)
- `n_trades`, `n_sims`, `seed`
- `bootstrap_method` (e.g. iid, block)
- `max_drawdown` distribution summary: `p50`, `p90`, `p95`, `p99`, `worst`
- `var_trade_pnl` / `cvar_trade_pnl` (5th percentile, trade-level horizon)
- `risk_of_ruin_prob` (ruin = max drawdown ≥ `MC_RUIN_THRESHOLD * starting_equity`)
- `ruin_definition` (human-readable string)
- `notes` (freeform)

### Permutation test sub-object (`permutation_test`)

Embedded inside `monte_carlo.json` under the `"permutation_test"` key.

Tests whether the observed win rate is statistically greater than the break-even win rate implied by the strategy's average planned R:R.

Fields:

| Field | Description |
|-------|-------------|
| `n_trades` | Number of observed trades |
| `observed_wins` | Count of winning trades (`label_win == 1`) |
| `observed_win_rate` | Observed win fraction |
| `avg_rr_planned` | Mean `rr_planned` across all trades |
| `break_even_win_rate` | `1 / (1 + avg_rr_planned)` — H0 null win rate |
| `null_win_rate_mean` | Mean of simulated null win rates |
| `null_win_rate_p95` | 95th percentile of simulated null win rates |
| `p_value_one_tailed` | Fraction of simulations ≥ observed win rate |
| `significant_05` | bool — p-value < 0.05 |
| `significant_01` | bool — p-value < 0.01 |
| `n_sims` | Number of binomial simulations |
| `seed` | RNG seed used |
| `notes` | Description of H0 and test method |

Interpretation: `significant_05 = true` means the strategy's win rate cannot be explained by random chance at the 5% level given its R:R profile.

Store alongside Monte Carlo so ML pipelines can join `Vn` performance + risk characteristics.

## 5) Parameter-sweep ML dataset (`ml_dataset_v{N}.parquet`)

Produced by `scripts/param_sweep.py`. One row per **closed trade** across all
sampled combos in the run. Used as Track-B ML training data.

Row layout = three concatenated groups, in order:

### (a) Combo metadata — identifies the parameter set that produced the trade

Keys in `_COMBO_META_KEYS` (source of truth in `scripts/param_sweep.py`):

- `combo_id` — integer index into the sampled combo list (seed-reproducible)
- `z_band_k`, `z_window`, `volume_zscore_window`
- `ema_fast`, `ema_slow`
- `stop_method` ∈ {`fixed`, `atr`, `swing`}, `stop_fixed_pts`, `atr_multiplier`, `swing_lookback`
  - **dtype invariant**: nullable numeric columns are always `float` when non-`None` (never `int`)
- `min_rr`, `exit_on_opposite_signal`, `use_breakeven_stop`, `max_hold_bars`, `zscore_confirmation`
- Z-score formulation dimensions: `z_input`, `z_anchor`, `z_denom`, `z_type`, `z_window_2`, `z_window_2_weight`
- V5+ entry filters: `volume_entry_threshold`, `vol_regime_lookback`, `vol_regime_min_pct`, `vol_regime_max_pct`, `session_filter_mode`, `tod_exit_hour`

### (b) Per-trade features (known at decision time — safe ML inputs)

Keys in `_TRADE_FEATURE_KEYS`:

- `zscore_entry`, `zscore_prev`, `zscore_delta`
- `volume_zscore`, `ema_spread`
- `bar_body_points`, `bar_range_points`, `atr_points`
- `parkinson_vol_pct`, `parkinson_vs_atr`
- `time_of_day_hhmm`, `day_of_week`
- `distance_to_ema_fast_points`, `distance_to_ema_slow_points`
- `side`

### (c) Labels / outcomes (must NOT be used as features)

Keys in `_LABEL_KEYS`:

- `label_win` — 0/1 classification target
- `net_pnl_dollars` — regression target (dollars)
- `r_multiple` — regression target (R units)

### Dtype consistency rule (Parquet append safety)

PyArrow rejects concat when the same column has mixed dtypes across batches.
Every `range_mode` branch in `_sample_combos` must use the same Python type
for a given key — in particular, nullable stop/lookback columns are always
`float(...)` when non-`None`. The pre-sweep reviewing gate (see
[`CLAUDE.md`](CLAUDE.md#pre-sweep-gate-mandatory)) enforces this.
