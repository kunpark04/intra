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
- `iterations/Vn/monte_carlo.json` (Monte Carlo risk summary)

Final hold-out evaluation:

- `evaluation/trades.csv`
- `evaluation/trader_log.csv`
- `evaluation/daily_ledger.csv`
- `evaluation/monte_carlo.json` (optional; same schema)

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

### Labeling + QA flags

- `label_win`
- `label_hit_tp_first`
- `data_quality_flag`
- `rule_violation_flag`

### Leakage rule (critical)

When preparing ML datasets:

- **Features**: only columns known at decision time (entry bar close / next open).
- **Labels**: outcomes (`net_pnl_dollars`, `r_multiple`, `label_win`) and post-trade diagnostics (`mae/mfe`) must not be used as input features.

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

Purpose: calendar coverage, including **no-trade days** (one row per day in the partition’s date range).

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
- `var` / `cvar` (define horizon + unit in metadata; trade-level or daily)
- `risk_of_ruin_prob` (include ruin definition in metadata)
- `notes` (freeform)

Store this alongside the logs so later ML pipelines can join `Vn` performance + risk characteristics.

