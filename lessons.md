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

### 2026-04-16 scripts_reorg_regex_blind_spots

- **What I ran**: Bulk rename of 49 scripts into 7 content-based subfolders
  (`scripts/X.py` → `scripts/<group>/X.py`) via `_scripts_reorg.py patch`, followed
  by `git mv` and AST-parse verification across all 55 files.
- **What happened**: AST parsing passed on first attempt, but a follow-up sweep
  found two dependency references the regex missed:
  1. `scripts/runners/run_filter_compare_v3.py:33` — a bash wrapper string
     `python3 scripts/filter_backtest_${job}_v3.py` where `${job}` is a shell
     variable. The regex was anchored on literal filenames and could not match
     a variable-substituted path, so the remote invocation would have failed
     with `No such file or directory` on sweep-runner-1.
  2. `scripts/analysis/build_combo_features_ml1.py:64` — a docstring comment
     `Replicates scripts/adaptive_vs_fixed_backtest_v1.build_features` (no `.py`
     suffix). Patcher regex required `\.py` terminator.
- **Root cause**: Automated rename tools are only as good as the patterns they
  match. Two failure modes surfaced:
  1. Runtime-constructed paths (bash variable substitution, string formatting,
     f-string interpolation) look like partial matches to regex and are
     silently skipped.
  2. Natural-language prose (docstrings, comments) can contain path-like
     references without file extensions that don't match strict
     `name\.py(?![A-Za-z0-9_])` patterns.
- **Fix**: Manually patched both sites after a broad grep `scripts/(adaptive_|
  build_|filter_|...)` across `.py` and other text files.
- **Prevention rule**: After any bulk rename, run a broad grep of *just the
  renamed basenames* across all text files — never trust that the patterning
  regex covered every invocation style. Include bash `$VAR` substitutions,
  f-string fragments, and prose/docstring references in the post-rename audit.
  Also: if old and new names share a prefix (e.g. `adaptive_rr_heldout` →
  `adaptive_rr_heldout_v2`), the rename regex **must** use negative lookahead
  `(?![A-Za-z0-9_])` to avoid double-patching (`_v2_v2`) on re-runs.
- **Related files/commands**: removed `_scripts_reorg.py` helper, commits
  `be7645e`, `2d88a49`.

---

### 2026-04-16 path_depth_tracking_after_file_moves

- **What I ran**: Moved 49 scripts from `scripts/` flat root into `scripts/<group>/`
  subfolders via `git mv`. The scripts use two path-resolution idioms:
  `Path(__file__).resolve().parents[N]` and `Path(__file__).parent.parent[...]`.
- **What happened**: After the move, the `parents[N]` index and
  `.parent.parent` chain length no longer pointed to the repo root — each
  moved file was now one directory deeper. If the moves had been committed
  without fixing these, 24 files would have been looking for data at
  `scripts/data/...` instead of `<repo>/data/...`.
- **Root cause**: Both idioms encode the number of directory hops from the
  file to the repo root. When a file moves deeper, the hop count must increase
  by exactly the number of new directory levels.
- **Fix**:
  - 24 files: `parents[1]` → `parents[2]` (all moved into `scripts/<group>/`).
  - 16 files: `.parent.parent` → `.parent.parent.parent`.
  - 11 files: `sys.path.insert(..., REPO / "scripts")` retargeted to
    `REPO / "scripts" / "models"` (or `"analysis"` for shap_audit_v2) so that
    sibling-module imports like `import adaptive_rr_model_v2` still resolve.
- **Prevention rule**: Before any `git mv` of Python files into a new
  subfolder, grep for `parents\[\d+\]`, `\.parent\.parent`, and
  `sys\.path\.insert.*scripts` in every file being moved. Count the new
  directory depth delta (usually +1 per move) and patch atomically with the
  move — never commit "moves" and "path fixes" as separate steps.
- **Related files/commands**: `_scripts_reorg.py`, commit `be7645e`.

---

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

---

### 2026-04-17 eval_notebook_oom_at_7g_memorymax

- **What I ran**: `scripts/runners/run_eval_notebooks_remote.py` — remote execution of `top_performance.ipynb` (6-section refactor with compounding sizing + ML#2 trade cache). Wrapper launched via `systemd-run --scope -p MemoryMax=7G -p CPUQuota=400%`.
- **What happened**: Notebook died ~3 min in with `DeadKernelError: Kernel died, restarting`. `dmesg` on sweep-runner-1 showed `oom-kill: ... python3 ... total-vm=8.2G RSS=7.2G` under a 7G cgroup cap. Node RAM is 9.7G total.
- **Root cause**: The new 6-section layout bootstraps 10,000 MC sims × 2 sizing policies × 2 sections (§3 + §6), each producing a `(n_sims, n_trades)` resample matrix. ML#2 stack (V3 booster + isotonic calibrators) also holds the full feature frame in memory for scoring. Aggregate peak RSS exceeds the old 7G headroom that was adequate for the flat-$500 pre-refactor notebook.
- **Fix**: Bumped `MemoryMax=7G` → `MemoryMax=9G` in the runner's wrapper command. Leaves ~700 MiB for the OS.
- **Prevention rule**: When adding new bootstrap / MC sections or new ML inference pipelines to a notebook, recompute the remote memory budget. Rule of thumb on sweep-runner-1 (9.7G RAM): MC-heavy notebooks need ≥9G cap; pure backtest notebooks can stay at 7G. Always check `dmesg | grep -i oom` on first failure before increasing any timeout — OOM masquerades as kernel timeouts.
- **Related files/commands**: `scripts/runners/run_eval_notebooks_remote.py` WRAPPER_SH.

---

### 2026-04-17 sharpe_under_compounding_must_use_log_returns

- **What I ran**: Added `pct5_compound` sizing policy to `top_performance.ipynb` via `_build_v2_notebooks.py`. First pass used the same `metrics_from_pnl(pnl_dollars, years) → mean/std(pnl) × sqrt(tpy)` Sharpe formula for both sizing policies.
- **What happened (error summary)**: Logical review agent flagged as WARN: Sharpe on raw $-PnL under compounding is scale-dependent. A strategy that multiplies every $-PnL by a growing equity factor inflates both mean and std by the same factor *per trade*, but the ratio is path-dependent because later trades see larger absolute dollars. This biases the Sharpe upward in compounded regimes vs. fixed-$ regimes — not comparable across the two sizing columns.
- **Root cause**: Sharpe under compounding should be computed on **per-trade log returns** `log(1 + 0.05 * r_t)`, which are additive and scale-invariant. Dollar PnL is not.
- **Fix**: Extended `metrics_from_pnl(pnl, years_span, policy, r=None, start_equity=...)`:
  - `policy == 'fixed_dollars_500'`: unchanged (mean/std of $-PnL).
  - `policy == 'pct5_compound'`: `log_ret = np.log1p(RISK_FRAC * r)`; `sharpe = log_ret.mean() / log_ret.std(ddof=1) * sqrt(trades_per_year)`.
  - Same split in `monte_carlo`'s `sharpe_boot` branch (bootstrap r-multiples, compute log-return Sharpe per sim).
  - Updated all 7 call sites to pass both `policy` and `r = pnl_base / risk_base`.
- **Prevention rule**: Whenever introducing a compounding sizing policy (position size scales with equity), Sharpe must switch to log-returns. Dollar-PnL Sharpe is only valid under flat-dollar sizing. If a reviewer can't tell at a glance which basis is used, add a docstring line naming the policy → metric convention. Same rule applies to Sortino, Calmar ratios when they're wired in.
- **Related files/commands**: `scripts/evaluation/_build_v2_notebooks.py` — `metrics_from_pnl`, `monte_carlo`, all 7 cell-source constants invoking them.

---

### 2026-04-17 cache_key_must_include_all_content_source_mtimes

- **What I ran**: Added ML#2 trade cache to `top_performance.ipynb` to skip the 15–25 min rebuild between iterations. First pass keyed on `(top_strategies.json mtime, V3 booster mtime, cache version)`.
- **What happened**: Logical review agent flagged as WARN: cache would serve stale results if any of (a) pooled per-R:R isotonic calibrators, (b) per-combo calibrators, or (c) `final_holdout_eval_v3_c1_fixed500.py` eval script were touched without bumping the booster. All three feed directly into ML#2 trade dicts.
- **Root cause**: The cache key reflected only the "lead artifact" (booster) and the "lead input" (strategies JSON), not the full set of files whose contents change the cached output. Any silent update to calibrators or eval code would produce a cache hit with stale trades.
- **Fix**:
  - Expanded `_cache_key` to include mtimes of: `V3_CALIBRATORS`, `V3_PER_COMBO_CALIBRATORS`, and `scripts/evaluation/final_holdout_eval_v3_c1_fixed500.py`.
  - Bumped `_ML2_CACHE_VERSION = 'v1' → 'v2'` to force a one-shot rebuild of any existing cache.
- **Prevention rule**: A cache key must include an identity signal (name + mtime or content hash) for **every** file whose contents affect the cached output. When in doubt, over-include — a false miss costs one rebuild; a false hit silently corrupts downstream results. Always pair any schema/code change to the cached function with a cache-version bump.
- **Related files/commands**: `scripts/evaluation/_build_v2_notebooks.py` — `run-ml2` cell source.

---

### 2026-04-17 remote_git_credentials_empty_blocks_push_workflow

- **What I ran**: Attempted to convert the evaluation-notebook artifact pipeline from SFTP-pull to git-based: remote wrapper would `git commit -am '…' && git push origin master`, local side would `git pull --ff-only`.
- **What happened (error summary)**: Remote `git push` failed with `remote: Invalid username or token. Password authentication is not supported for Git operations.` Remote had created a local-only commit (`93b0391`) that couldn't reach origin. Local side had no way to pull the artifacts.
- **Root cause**: `~/.git-credentials` on sweep-runner-1 is 0 bytes. GitHub removed password auth in 2021; only PATs and SSH deploy keys work. The VM was never configured with either.
- **Fix** (interim): Reverted `_pull_eval_nbs.py` + `run_eval_notebooks_remote.py` to SFTP-based transfer (paramiko `sftp.get` for `top_performance.ipynb` + `top_trade_log.xlsx`). Hard-reset the orphan commit on remote: `git reset --hard origin/master`.
- **Prevention rule**: Before proposing a git-based remote artifact workflow, verify remote auth works: `ssh <host> 'cd /root/intra && git ls-remote origin &>/dev/null && echo OK || echo NEEDS_AUTH'`. If NEEDS_AUTH, either (a) install a fine-grained PAT in `~/.git-credentials` with `credential.helper=store`, or (b) add an SSH deploy key and switch origin to `git@github.com:...`. Until one of these lands, SFTP-pull is the only option for fetching remote-generated artifacts. Never assume `git push` works on a VM without testing it.
- **Related files/commands**: `scripts/runners/run_eval_notebooks_remote.py`, `scripts/runners/_pull_eval_nbs.py`.

---

### 2026-04-20 always_ablate_vs_trivial_baselines

- **What I ran**: Pool B validation gauntlet — three council-mandated ablations against the shipped Pool B + V4 pipeline (Sharpe p50 2.13). Step 2 (rank stability via ±90/±180d train-window shifts), Step 3 (Pool B + V3 filter), Step 4 (full post-gate pool + V4, no ML#1 top-K).
- **What happened (result summary)**: Steps 2 and 3 resolved ambiguously — Jaccard 0.653 is in the "noisy but real" band, and Pool B + V3 was Sharpe 1.78 / DD 123.7% (close enough to Pool B + V4 that a council without Step 4 would have remained in HOLD). It was Step 4 — the *simplest possible baseline*, with ML#1's top-K concentration removed entirely — that produced the decisive signal: Sharpe 0.00, DD worst 280%, ruin prob 43.45% on 24,542 trades. Without that arm, no evidence existed that ML#1's selection gate was load-bearing; a future simplification proposal could have retired ML#1 with no way to detect the catastrophic regression.
- **Root cause** (of the near-miss, not of a bug): I initially framed the gauntlet around the subtle questions (rank stability, bundle attribution) and the "kill-ML#1" baseline was the fifth priority, nearly cut for execution time. The subtle questions produced subtle answers. Only the trivial ablation produced a signal big enough to decide on.
- **Fix**: Ran the full-pool baseline anyway (plan §Step 4). Hit three upstream failures (CellTimeoutError at 7200s, s3 OOM at 9G cgroup, then skip-s3 wrapper succeeded) — a strong hint that the full-pool regime is *operationally* as well as *statistically* unstable. Total cost of the Step 4 arm: ~2 hours of remote execution and three iteration cycles.
- **Prevention rule**: **For any architectural-change decision, the cheapest and most decisive evidence usually comes from the most trivial baseline** — "what if we removed this component entirely?" — not from the subtle ablations that perturb parameters within a working pipeline. When designing a validation gauntlet, include at least one "remove component X" arm for every load-bearing component the user is thinking about removing. Order the arms so the trivial baselines run first: a strong signal there can short-circuit the whole gauntlet (either confirming SHIP or triggering HOLD without needing the subtle arms). Corollary: the council verdict section should explicitly enumerate "what was the simplest possible baseline we tested?" before finalizing.
- **Related files/commands**: `tasks/ship_decision.md`, `evaluation/v12_full_pool_net_v4_2k/s6_mc_combined_ml2_net.ipynb`, plan `C:\Users\kunpa\.claude\plans\sleepy-roaming-kettle.md` §Step 4.

---

### 2026-04-20 skip_gross_suite_for_new_eval_variants

- **What I did**: Added the `v12_top50_raw_sharpe_v4_no_gcid` variant to `scripts/evaluation/_build_v2_notebooks.py` by copying the shipped `v12_top50_raw_sharpe_v4` block — which uses `_build_variant(gross_dir, net_dir, ...)` to emit a paired 6-notebook gross suite + 6-notebook net suite. That's 12 notebooks per variant, half of which run under zero-friction assumptions.
- **What happened (user correction)**: User pushed back: "Is the gross suite relevant if it does not account for frictions?" Since April 2026 (`_ml2_net_ev_mask` wiring, `pct5_compound` removal, single-policy MC), ship decisions read only `s6_net` Sharpe — gross Sharpe never enters the pass/fail criterion. Gross is at best a decomposition diagnostic (did combo-id memorization drive lift via trade-count or via selection quality?), at worst a misleading "upside" figure that creeps into write-ups (e.g. commit 734d3a3 quoting "Sharpe p50 2.77 gross"). The compute cost — two extra MC-heavy notebooks (s3, s6) per variant, both OOM-risky on the 9G cap — is real.
- **Root cause**: I copy-pasted the old variant's structure without re-asking whether gross was load-bearing for *this* specific question. The builder's `_build_variant` API defaults to gross+net paired, which is convention rot from Phase 6 (pre-friction-aware) — every new variant inherits it by muscle memory.
- **Fix**: Added `_build_net_variant(net_dir, setup_net_src, title_tag="")` helper alongside the legacy `_build_variant`. Docstring and a bullet in CLAUDE.md "Reporting & styling requirements" flag `_build_net_variant` as the preferred API for new variants from 2026-04-20 forward. Left the v4_no_gcid call on the legacy API (already mid-execution on remote at the time of correction); it's explicitly marked as "LAST dual gross+net variant" in a code comment.
- **Prevention rule**: When adding a new variant to `_build_v2_notebooks.py`, the default is `_build_net_variant`. Only use the legacy `_build_variant` if the user explicitly requests a gross+net comparison — e.g. for a research note decomposing where ML#2's lift comes from. Before adding any new paired emit, check: "Does the ship-decision rubric care about gross?" If the answer is no (it always is, post-April 2026), emit net-only. Corollary: when quoting results in write-ups, cite only the net metric. Gross numbers are a trap — they set expectations the production pipeline cannot meet.
- **Related files/commands**: `scripts/evaluation/_build_v2_notebooks.py` (`_build_net_variant` helper + docstring), `CLAUDE.md` §Reporting & styling requirements (net-only policy bullet).

---

### 2026-04-20 plot_mc_sims_has_no_n_sims_kwarg

- **What I ran**: Patched the v4_no_gcid s3 notebooks (gross + net) to pin `n_sims=2000` for 9G MemoryMax compliance, mirroring commit 2105b45's shipped V4 patch. My Python patcher applied `YEARS_SPAN)` → `YEARS_SPAN, n_sims=2000)` indiscriminately to every `monte_carlo` and `plot_mc_*` call — 5 matches per notebook.
- **What happened (error summary)**: Remote nbclient crashed on the first MC call in s3_gross: `TypeError: plot_mc_sims() got an unexpected keyword argument 'n_sims'`. `set -e` in the bash wrapper killed the whole suite at notebook 3 of 12. s4–s6 gross and all 6 net notebooks did not execute.
- **Root cause**: Commit 2105b45 patched exactly 4 functions: `monte_carlo`, `plot_mc_pnl`, `plot_mc_sharpe`, `plot_mc_dd`. It deliberately skipped `plot_mc_sims` — which takes `n_paths=_MC_SIM_PATHS` (the per-path sample count for the sample-paths spaghetti plot), not `n_sims` (the bootstrap matrix width). These are different axes: `n_paths` controls how many equity paths are drawn, `n_sims` controls how many bootstrap samples go into the distribution. My grep-and-replace conflated them. The signature difference is visible at `scripts/evaluation/_top_perf_common.py:618` vs `:648/:664/:692`.
- **Fix**: Reverted the `plot_mc_sims(..., n_sims=2000)` call in both s3 files back to `plot_mc_sims(..., YEARS_SPAN)`. Re-uploaded via the main runner (not just sftp patch — the runner handles screen-kill + relaunch atomically). Resumed from notebook 1; s1/s2 are cheap so re-running them costs under a minute.
- **Prevention rule**: When replicating a historical commit's patch pattern on new files, **read the commit diff explicitly** before running a grep-and-replace. Count functions: 2105b45 patched 4 calls, not 5. Any kwarg-based patch script must enumerate target function names, not just match on trailing patterns (`YEARS_SPAN)`). Corollary for any future `n_sims` pinning: the whitelist is `{monte_carlo, plot_mc_pnl, plot_mc_sharpe, plot_mc_dd}` — `plot_mc_sims` takes `n_paths` and is always left alone.
- **Related files/commands**: `scripts/evaluation/_top_perf_common.py:618` (`plot_mc_sims` signature), commit 2105b45, `evaluation/v12_topk_top50_raw_sharpe_v4_no_gcid/s3_mc_combined.ipynb` (target of revert).

