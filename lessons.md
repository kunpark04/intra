## Purpose

Persistent log of lessons learned and mistakes encountered while building/running this repo, so Claude Code (and humans) do not repeat them.

## How to use

- Before running new tooling commands (installs, notebook execution, backtest scripts), scan recent entries.
- When something fails, add a new entry immediately with the fix and a prevention rule.

## Entry template

### YYYY-MM-DD Short_title

- **What I ran**:
- **What happened (error summary)**:
- **Root cause**:
- **Fix**:
- **Prevention rule**:
- **Related files/commands**:

---

### 2026-04-11 sweep_oom_rolling_rank_and_no_del

- **What I ran**: V9 parameter sweep (`--range-mode v9`, 6000 combos).
- **What happened**: 1,896/6000 combos (31.6%) failed with `ArrayMemoryError`. Two failure modes: (A) allocations of shape `(n_bars, z_window)` float64 (e.g. `(2049955, 40)` = 625 MiB) crashing `_rolling_rank`; (B) later, even tiny allocations like `(3, 2049994)` bool = 5.87 MiB failed due to heap fragmentation.
- **Root cause**:
  1. `_rolling_rank` in `zscore_variants.py` used `np.lib.stride_tricks.as_strided` with overlapping strides, then `wins < current` forced numpy to materialize the full `(n, window)` float64 array — up to 782 MiB per `quantile_rank` combo.
  2. No `del df_ind / df_sig / results` between combos; Python GC couldn't keep up over 6000 iterations, fragmenting the heap until even small contiguous allocations failed.
- **Fix**:
  1. Replaced `_rolling_rank` with pandas `rolling().rank(method='average')` → O(window) memory, equivalent math: `(rank_1based - 0.5) / window`.
  2. Added `del df_ind` after `generate_signals`, `del df_sig` after `_run_backtest_light`, extracted needed fields then `del results`.
  3. Added `gc.collect()` every 50 combos.
- **Prevention rule**: Any rolling window function that uses `as_strided` with overlapping strides and then performs element-wise comparisons will materialise an `(n, window)` intermediate array. For 2M-bar series, even window=30 → 469 MiB. Always prefer pandas rolling or a Numba loop for ranked/comparison-based rolling ops. Also: always `del` large DataFrames inside sweep loops and call `gc.collect()` periodically.
- **Related files/commands**: `src/indicators/zscore_variants.py:_rolling_rank`, `scripts/param_sweep.py` main loop.

---

### 2026-04-11 swing_lookback_int_vs_float_type_crash

- **What I ran**: Launched V8 parameter sweep with `--range-mode v8`.
- **What happened**: Sweep crashed at combo 100 (first batch flush) with `pyarrow.lib.ArrowTypeError: Unable to merge: Field swing_lookback has incompatible types: double vs int64`.
- **Root cause**: V8 branch assigned `swing_lookback = int(rng.integers(...))`, while all prior branches used `float(rng.integers(...))`. When a PyArrow batch contains only `None` values for a nullable column, pandas infers it as `float64`. A subsequent batch where the same column holds `int` values produces `int64`. `pa.concat_tables` rejects the type mismatch.
- **Fix**: Changed V8 branch to `float(rng.integers(...))` to match all other branches.
- **Prevention rule**: Every nullable numeric column (`swing_lookback`, `stop_fixed_pts`, `atr_multiplier`, etc.) must use the **same Python type** (`float`) for non-`None` values across **every** `range_mode` branch. Never use bare `int(...)` for these fields. The pre-sweep reviewing agent must explicitly grep every branch for these columns and compare types.
- **Related files/commands**: `scripts/param_sweep.py` — all `range_mode` branches, `swing_lookback` assignment.

---

### 2026-04-11 sweep_log_naming_off_by_one

- **What I ran**: Added log naming code using `enumerate(sys.argv[1:], 1)` to derive sweep log filename from `--output` arg.
- **What happened**: V7 sweep wrote to `sweep_run--output.log` instead of `sweep_run_v7.log`.
- **Root cause**: `enumerate(sys.argv[1:], 1)` yields `(_i, _arg)` where `_i` is 1-based and indexes `sys.argv[1:]`. When `_arg == "--output"`, `_i` points to the flag itself in `sys.argv`, not the value after it. `sys.argv[_i]` reads `"--output"` rather than `"data/ml_dataset_v7.parquet"`.
- **Fix**: Replaced enumeration with a simple `range(1, len(sys.argv) - 1)` loop and access `sys.argv[_j + 1]` (the value after the flag).
- **Prevention rule**: When parsing `sys.argv` manually to find flag+value pairs, always use `sys.argv[j+1]` for the value. Never use `enumerate(sys.argv[1:], start)` and then index back into `sys.argv[start]` — the index math is confusing. Prefer `for j in range(1, len(sys.argv)-1): if sys.argv[j] == "--flag": value = sys.argv[j+1]`.
- **Related files/commands**: `scripts/param_sweep.py` `__main__` block.

---

### 2026-04-10 nbconvert_cwd_flag_not_recognized

- **What I ran**: `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.cwd=<path> notebooks/01_backtest_and_log.ipynb`
- **What happened**: `ModuleNotFoundError: No module named 'src'` — kernel launched from wrong cwd; `sys.path.insert(0, ".")` adds `.` relative to the kernel's cwd, not the repo root.
- **Root cause**: `--ExecutePreprocessor.cwd` is not a recognized config option in nbconvert 7.x / nbclient 0.10.x. The flag is silently ignored.
- **Fix**: Use `nbclient.NotebookClient` directly in a Python one-liner with the `cwd` kwarg:
  ```python
  import nbformat
  from nbclient import NotebookClient
  with open('notebooks/01_backtest_and_log.ipynb') as f:
      nb = nbformat.read(f, as_version=4)
  NotebookClient(nb, timeout=600, kernel_name='python3', cwd='<repo_root>').execute()
  with open('notebooks/01_backtest_and_log.ipynb', 'w') as f:
      nbformat.write(nb, f)
  ```
- **Prevention rule**: Never use `--ExecutePreprocessor.cwd` with nbconvert. Always use `nbclient.NotebookClient(cwd=...)` for in-place notebook execution with a specific working directory.
- **Related files/commands**: `notebooks/01_backtest_and_log.ipynb`, `notebooks/02_plotly_exploration.ipynb`

### 2026-04-10 reviewing_agent_catches_critical_bugs

- **What I ran**: Implemented `zscore_variants.py` + updated `param_sweep.py` for V4 sweep without running a reviewing agent first
- **What happened**: 4 CRITICAL bugs found only when reviewing agent was finally spawned:
  1. Cache key missing `ema_fast`/`ema_slow` — EMA-anchor combos silently served wrong z-scores
  2. `returns` combos recorded wrong `z_anchor` in ML Parquet metadata (price-space anchor stored despite override)
  3. `_rolling_rank` tie handling: all-equal window produced z=±3 instead of 0
  4. `returns+atr` unit mismatch (returns ~1e-4, ATR ~20pts) — permanently zero trades, wasted sweep time
- **Root cause**: Logical errors in new code are invisible to the author who wrote the design. A fresh agent with no prior assumptions finds them systematically.
- **Fix**: Spawned reviewing agent with explicit checklist (math, edge cases, cache keys, metadata accuracy, compatibility rules). Fixed all 4 before proceeding.
- **Prevention rule**: Every logical code change (new indicator, sweep parameter, formula, cache) must have a reviewing agent spawned before marking complete. The prompt must include an explicit severity-tagged checklist — not just "check for bugs." See `CLAUDE.md` Reviewing Agent Protocol section.
- **Related files**: `src/indicators/zscore_variants.py`, `scripts/param_sweep.py`

