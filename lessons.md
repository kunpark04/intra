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

### 2026-04-26 nbclient_kernel_startup_timeout_on_windows_py313

- **What I ran**: `python <<'PYEOF' ... NotebookClient(nb, timeout=300, startup_timeout=180, kernel_name="python3").execute() ... PYEOF` to re-execute `evaluation/probe5_combo865_backfill/analysis.ipynb` after a sizing-change rerun of the backfill artifacts.
- **What happened**: First call: `RuntimeError: Kernel didn't respond in 60 seconds`. Second call (with `startup_timeout=180` and `asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())` set before any nbclient/zmq import): `RuntimeError: Kernel didn't respond in 180 seconds`. `ipykernel 7.2.0` was installed; `python3` kernel spec was registered (verified via `jupyter kernelspec list`). The `RuntimeWarning: Proactor event loop does not implement add_reader family of methods required for zmq` appeared on the first call but NOT on the second (after the policy fix), so the warning was suppressed but the underlying timeout was unchanged — the warning is informational, not the actual root cause.
- **Root cause**: Some interaction between `nbclient.NotebookClient`'s async kernel-manager startup handshake and Python 3.13's tightened Windows asyncio behavior. The kernel subprocess spawns but the `wait_for_ready` ZMQ handshake never completes within timeout. Exact failure mode was not isolated this session (may relate to socket binding, AV interaction, or asyncio loop affinity); the workaround was confirmed before deeper RCA was prioritized.
- **Fix**: `jupyter nbconvert --to notebook --inplace --execute <path> --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.startup_timeout=180`. Different invocation chain; same notebook output (628 KB embedded matplotlib outputs); ran cleanly on the same machine seconds after the `nbclient` direct call had timed out twice. cwd defaults to the notebook's directory under nbconvert, so no chdir injection is needed (in contrast to `scripts/exec_analysis.py`'s pattern).
- **Prevention rule**: **On Windows + Python 3.13, prefer `jupyter nbconvert --to notebook --inplace --execute` over `nbclient.NotebookClient` direct execution for re-running notebooks.** The repo's existing `scripts/exec_analysis.py` uses the `nbclient` path and is now confirmed-vulnerable; if it starts hanging in future sessions, switch its core to `subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--inplace", "--execute", path, "--ExecutePreprocessor.timeout=300"], check=True)`. Setting `WindowsSelectorEventLoopPolicy` before `nbclient` import is *not* a sufficient fix — confirmed empirically.
- **Related files/commands**: `scripts/exec_analysis.py` (vulnerable, not yet patched), `evaluation/probe5_combo865_backfill/analysis.ipynb` (re-executed via nbconvert this session), `tasks/probe5_combo865_paper_trade_preregistration.md` §1.A (analyst pre-sign note that mentions the re-execution).

---

### 2026-04-26 bash_run_in_background_python_path_127

- **What I ran**: `python scripts/paper_trade/backfill_combo865.py --end "2026-04-24 15:00:00"` via the Bash tool with `run_in_background=true`. Same script + args had succeeded ~30s earlier when auto-backgrounded by the harness without the `run_in_background` flag.
- **What happened**: Exit 127 (command not found) with empty stdout/stderr. Retried with `py` launcher: same exit 127. Retried via PowerShell: exit 255 with empty output. The PATH-cache theory was hard to validate because every failure mode was silent.
- **Root cause**: Unclear. The `python` binary IS on PATH globally — the auto-backgrounded call worked, and a `python --version` that completed in the *same* shell-tool round-trip later returned `Python 3.13.12` cleanly. Hypotheses: (a) miniconda env activation script not sourced for `run_in_background=true` subshells, (b) PATH inheritance differs between Bash-auto-background and explicit-background code paths, (c) transient Windows process-spawn race. No deterministic repro pattern was isolated.
- **Fix**: Prefix the invocation with `python --version &&`: e.g. `python --version && python scripts/...`. Empirically this restored PATH resolution for the subsequent `python` call in the same shell. Worked on first retry this session.
- **Prevention rule**: **If a Bash `run_in_background=true` invocation of a Python script fails with exit 127 (or exit 255 from PowerShell) with empty output, retry with `python --version && python ...` chained.** Cheap workaround; root cause unknown. If the workaround stops working, escalate to `echo "$PATH" && which python && python ...` to surface what the subshell actually sees, since the failure mode is otherwise silent.
- **Related files/commands**: bash tool invocations bwu1olba4 (workaround success), bkf5q70jn / b70wangis / bzb05w81z / b1s05cc8m (failure variants).

---

### 2026-04-25 exit_reason_engine_codes_are_1_indexed

- **What I ran**: Wrote `scripts/paper_trade/backfill_combo865.py` to re-run combo-865 from the 1h test partition start through the latest available bar. The bar-by-bar engine cores (Cython / Numba / NumPy) emit per-trade `exit_reason` as an integer code; I converted via `["sl","tp","opposite_signal","max_hold","tod_exit","end_of_data"][reason]` — a 0-indexed list lookup.
- **What happened**: Code-logic-reviewer cross-tabulated `exit_reason` against `r_multiple` on the produced `trades.csv` and found 104 rows tagged `tp` all had `r_multiple = -1.0` (impossible — a take-profit must be at +MIN_RR), and 118 rows tagged `opposite_signal` were uniformly at +RR. Combo-865 has `EXIT_ON_OPPOSITE_SIGNAL=False`, so code 3 cannot legally fire — the label inversion was provable on its face. Every Sharpe/dollars number was numerically correct (those don't read the label) but the entire exit-reason breakdown column was scrambled.
- **Root cause**: Engine cores emit **1-indexed** exit codes. Canonical mapping at `scripts/param_sweep.py:945`: `_EXIT_REASON_MAP = {1: "stop", 2: "take_profit", 3: "opposite_signal", 4: "end_of_data", 5: "time_exit"}`. A 0-indexed list lookup shifts every label by one slot — `code 1 (stop)` reads `"sl"` (correct by accident), `code 2 (take_profit)` reads `"opposite_signal"`, etc. Coincident-correctness on `code 1` is the worst kind of bug because it makes the lookup look sane on cursory inspection.
- **Fix**: Replaced the 0-indexed list with `_EXIT_REASON_MAP.get(int(reason), f"code_{int(reason)}")` mirroring `param_sweep.py:945` verbatim. Post-fix cross-tab: 117 `take_profit` rows uniformly at `r_multiple = +1.8477` (= MIN_RR), 103 `stop` rows uniformly at `-1.0000`. Coherent.
- **Prevention rule**: **Engine exit codes are 1-indexed across all three core tiers (Cython/Numba/NumPy).** Never define a 0-indexed list lookup for `exit_reason`. Always import or copy the canonical dict from `scripts/param_sweep.py:945`. **Empirical post-condition for any new trade-converter code**: `df.groupby("exit_reason")["r_multiple"].agg(["min","max","mean","count"])` — `stop` rows must be uniformly at -1.0, `take_profit` rows at +MIN_RR exactly. The one-line groupby diagnostic catches label inversion that pure code-reading misses, because it cross-checks logic against arithmetic.
- **Related files/commands**: `scripts/param_sweep.py:945` (canonical map), `src/cython_ext/backtest_core.pyx` (Cython core emit), `src/backtest._backtest_core` (Numba core emit), `src/backtest._backtest_core_numpy` (NumPy fallback emit), `scripts/paper_trade/backfill_combo865.py:124-127` (post-fix module-level constant), `tasks/_agent_bus/20260425-0738/code-logic-reviewer.md` (review C1).

---

### 2026-04-25 reproducibility_drift_is_a_bug_not_edge_case

- **What I ran**: Reproduction script for combo-865's 220-trade Probe 2 result on the 1h test partition. Engine returned **222 trades**. I dismissed the 2-trade drift as "expected from edge cases — `entry_timing_offset` k-shift / `cooldown_after_exit_bars` boundary handling differences between this script and the canonical `param_sweep` path" and shipped the artifacts with that framing.
- **What happened**: Code-logic-reviewer's first pass diagnosed: "k-shift and cooldown are bit-identical to `param_sweep._process_one_combo:1207-1213`. The drift is not from edge cases — it's from running on the wrong partition." Root cause was using `TEST_PARTITION_START = "2024-10-22 05:07:00"` (the 1-min boundary from `src.config.TRAIN_END_FROZEN`) on 1h bars, where the actual 80/20 split lands at `2024-10-30 21:00:00`. The script processed **152 extra 1h bars (~6 calendar days)** before the real 1h test partition began; those bars produced exactly the 2 extra trades.
- **Root cause**: I rationalized a quantitative discrepancy with a qualitative excuse ("edge cases") instead of diagnosing. K-shift and cooldown are deterministic and source-identical between this script and `param_sweep`, so attributing drift to them was unfalsifiable hand-waving. The real cause was a single off-by-partition error that the simplest possible diagnostic — "what timestamp is the boundary actually at?" — would have caught immediately.
- **Fix**: Hardcoded `TEST_PARTITION_START = pd.Timestamp("2024-10-30 21:00:00")` from `floor(0.8 × 41,900) = 33,520 → df["time"].iloc[33520]`. Re-ran: exactly 220 trades, WR 53.18%, TP/SL r_multiple uniform. Lock confirmed by three independent paths: (a) window length recomputation, (b) per-reason r_multiple uniformity, (c) sizing arithmetic.
- **Prevention rule**: **Any reproducibility delta against a signed reference is a bug until mechanically proven otherwise.** Never accept "expected from edge cases" or similar qualitative excuses to close out a discrepancy. The diagnostic must produce either (a) an exact match, or (b) a named, mechanically traceable cause. Specifically for backtest reproduction: when a re-run differs from a signed reference by N trades, the **minimal first diagnostic** is to recompute `floor(train_ratio × len(bars))` against the same parquet and verify the partition boundary timestamp matches the reference's claim. Cost: 30 seconds. Catches a non-trivial fraction of "off-by-something" reproduction failures.
- **Related files/commands**: `src/config.py:87` (`TRAIN_END_FROZEN` — 1m boundary), `src/data_loader.py:46-72` (`split_train_test`), `scripts/paper_trade/backfill_combo865.py:111-116` (post-fix 1h boundary), `tasks/_agent_bus/20260425-0738/code-logic-reviewer.md` (review C3).

---

### 2026-04-25 partition_boundaries_are_timeframe_dependent

- **What I ran**: Same backfill script as the prior lesson. Used `TEST_PARTITION_START` mechanically derived from `src.config.TRAIN_END_FROZEN = "2024-10-22 05:07:00"` (a literal string in the config module).
- **What happened**: That timestamp is the chronological 80/20 split boundary computed on the **1-minute** bar series. On 1h bars (each row = 60 1m bars), the same 80/20 split lands at `floor(0.8 × 41,900) = 33,520`, which corresponds to `2024-10-30 21:00:00`. The two boundaries differ by **8 calendar days / 152 1h bars / ~488 1-minute bars**. Probe 2's verdict text says "test partition: 2024-10-22 → 2026-04-08" — but that's narrative shorthand referring to the 1m split; the 1h-engine test partition really starts on 2024-10-30.
- **Root cause**: `src.data_loader.split_train_test(df, train_ratio)` computes the boundary against whatever bar series the caller passes. Same `train_ratio = 0.8` → different right-edge timestamp depending on bar timeframe. `TRAIN_END_FROZEN` is a frozen **1-min** boundary; cross-applying it to a 1h script silently includes 6+ days of training-fold data in what the script labels "test." Verdicts written for human readers use 1m-shorthand timestamps because that's the canonical project boundary, but the *engine-level* boundary on a non-1m timeframe is silently different.
- **Fix**: Documented the 1h boundary explicitly via a comment block in `backfill_combo865.py:108-114` and hardcoded `pd.Timestamp("2024-10-30 21:00:00")`. Future per-timeframe scripts must either (a) recompute the boundary dynamically from `floor(train_ratio × len(df))` against the actual loaded parquet, or (b) maintain a per-timeframe lookup table with explicit naming (e.g. `TEST_START_1H`, `TEST_START_15M`).
- **Prevention rule**: **Test-partition boundaries are timeframe-dependent.** `src.config.TRAIN_END_FROZEN` is the 1m boundary; do not cross-apply to 15m / 1h / 4h scripts. When writing code that consumes a non-1m bar parquet, either compute the boundary dynamically (`df["time"].iloc[int(train_ratio * len(df))]`) or hardcode the timeframe-specific timestamp with a comment naming which bar series it came from. **Probe verdict text's date ranges are 1m-shorthand narrative, not 1h-engine ground truth** — they look authoritative but they're not directly usable in non-1m scripts.
- **Related files/commands**: `src/config.py:87` (1m `TRAIN_END_FROZEN`), `src/data_loader.py:46-77` (`split_train_test`), `scripts/paper_trade/backfill_combo865.py:108-114` (post-fix 1h boundary with comment), `tasks/probe2_verdict.md` (1m-shorthand range), `tasks/_agent_bus/20260425-0738/code-logic-reviewer.md` (review C3).

---

### 2026-04-24 artifact_drift_after_retraction_cascade

- **What I ran**: Stats-ml-logic-reviewer audit of the probe pipeline after the 2026-04-23 TZ-bug retraction cascade had fixed Probes 3, 4, and Scope D at the script level. Reviewer returned 0 critical, 3 warn, 4 info findings. One of the WARN findings checked the machine-readable aggregates that downstream automation would consume.
- **What happened**: WARN 1 flagged `data/ml/probe3/readout.json` as **stale** — it still reported `F=0`, `branch=PAPER_TRADE` from the pre-TZ-fix run, while every human-readable artifact (`probe3_verdict.md` Amendment 2, CLAUDE.md Signal family status, `memory/project_probe3_combo865_pass.md`) had already been updated to `F=1`, `COUNCIL_RECONVENE`. No runtime error, no failing test, no visible contradiction in isolation — the JSON was internally consistent, just frozen against the obsolete verdict. Any automation (council pipeline, ship gate, dashboard) that read this JSON as ground truth would silently consume the retracted conclusion while every human-reader artifact said otherwise.
- **Root cause**: Multi-step text retraction (Probe 3 Amendment 1 → Amendment 2 → CLAUDE.md rewrite → memory updates → cross-probe cascade doc → ship_decision.md banner) optimized for "make the retractions visible to human readers first." The per-gate input scripts (`_probe3_1h_ritual.py`, `_probe3_15m_nc.py`) were rerun and produced correct post-TZ-fix outputs. But the *aggregator* one level downstream (`_probe3_readout.py`, which consumes the per-gate outputs and emits `readout.json`) was not re-invoked — no script or hook signals to the author that "upstream re-runs mean this aggregator is now stale." The cascade's mental model stopped at "text artifacts I'm editing" and did not extend to "machine outputs downstream of the scripts I already re-ran."
- **Fix**: Re-ran `tasks/_probe3_readout.py` against the post-TZ-fix per-gate inputs, regenerating `data/ml/probe3/readout.json` with `F=1`, `branch=COUNCIL_RECONVENE`. Committed with the rest of the WARN/INFO fixes as `40bc05d`. Added this lesson so the pattern is surfaced at the start of future retractions.
- **Prevention rule**: **After any retraction cascade that touches multiple gate scripts, enumerate every downstream aggregate (readout JSON, manifest, summary parquet, cached notebook output) and re-run the aggregator — even if the upstream scripts individually produced correct outputs.** Stale aggregates are silent: no error, no failing test, no visible contradiction in isolation. They only diverge against the text narrative, which is exactly where automated consumers don't look. Class name for future audits: **"artifact drift."** Practical mechanical check at the end of every retraction: `git ls-files "data/ml/<probe>/*.json" "data/ml/<probe>/*.parquet"` → for each file, compare its mtime against the last-touched commit on the upstream scripts; any aggregate older than its inputs is suspect. Run this check *before* declaring the retraction complete, not after an external reviewer catches it.
- **Related files/commands**: `tasks/_probe3_readout.py` (aggregator), `data/ml/probe3/readout.json` (regenerated artifact), `tasks/probe3_verdict.md` Amendment 2, `tasks/_agent_bus/pipeline_review_2026-04-24/stats-ml-logic-reviewer.md` WARN 1, commit `40bc05d`.

---

### 2026-04-24 classifier_coverage_drift_under_framing_inversion

- **What I ran**: Stats-ml-logic-reviewer audit of Scope D (per-combo session-dominance classification) after the TZ retraction cascade. `tasks/_scope_d_readout.py` emits a `_dominance_label` per combo — a categorical readout of which session bucket (SES_0 = full day, SES_1 = RTH, SES_2a/2b = GLOBEX overnight sub-windows, SES_3 = RTH-ex-lunch) drove that combo's Sharpe.
- **What happened**: WARN 2 flagged that the `_dominance_label` if/elif chain only branched on `SES_2a` and `SES_2b` dominance — the two overnight sub-windows that Scope D was originally designed to discriminate. Under the **original** (buggy) ET-as-UTC framing, the interesting finding was "edge concentrates in GLOBEX overnight," so the classifier author encoded only the overnight sub-case. Under the **corrected** CT-aware framing, the label remapping inverted: combo-1298's dominant bucket is now `SES_1` (RTH, Sharpe **4.41** on 77 trades — more than 3× the absolute gate), but the classifier had no `elif SES_1 ...` branch, so it fell through to the default "weak." A 4×-gate-clear combo was reported as having no session-level edge.
- **Root cause**: Classifier authors encode the *categories their current hypothesis makes salient*. Scope D's original hypothesis was "overnight drives the edge," so the classifier branched on the overnight categories. When a label remapping (CT→ET fix) reassigned which bucket each trade belonged to, the *data* stayed numerically identical, but the *categorical distribution* over buckets inverted. The classifier's category space — frozen to the pre-inversion framing — no longer spanned the observed dominance pattern, and the missing category silently defaulted to a semantic opposite ("weak" when the truth was "strongly dominated by RTH"). Different failure mode from artifact drift (the script ran cleanly and produced a well-formed label) — here the defect is in *semantic coverage*, not in staleness.
- **Fix**: Extended `_dominance_label` in `tasks/_scope_d_readout.py` to treat SES_1 (RTH) dominance as a first-class category with the same threshold logic already used for SES_2a/2b. Re-ran the aggregator; combo 1298 now labels as `SES_1 (RTH) dominates`, consistent with its 4.41 Sharpe on RTH-only trades. Committed in `40bc05d`.
- **Prevention rule**: **Whenever a finding's narrative axis inverts (e.g., "overnight dominates" → "RTH dominates"), re-audit the category space of every classifier that operated on the original framing before trusting its outputs on the inverted data.** Categorical outputs carry implicit coverage assumptions: branches exist for categories the author expected, and anything outside collapses to a default that may not be truthful under the new framing. Concrete check when framing inverts: grep the classifier for the *previously dominant* category strings (`SES_2a`, `SES_2b`) and verify the *newly dominant* category (`SES_1`) has branch parity — same threshold, same label-severity, same downstream treatment. **Corollary**: classifier scripts over categorical labels should make their default branch **loud** (`raise ValueError(f"unexpected dominance pattern: {buckets}")` or at minimum a `logger.warning`) rather than silently returning a neutral sentinel like "weak" — so coverage gaps become visible defects, not invisible ones. Class name for future audits: **"coverage drift under framing inversion."**
- **Related files/commands**: `tasks/_scope_d_readout.py:_dominance_label`, `data/ml/scope_d/readout.json`, `tasks/scope_d_brief.md` Amendment, `tasks/_agent_bus/pipeline_review_2026-04-24/stats-ml-logic-reviewer.md` WARN 2, commit `40bc05d`.

---

### 2026-04-23 tz_bug_in_session_decomposition

- **What I ran**: Probe 3 §4.3 (15m negative control) + §4.4 (1h session/exit ritual), Probe 4 §4.4 session decomposition, and Scope D post-hoc SES_2 sub-window split. All used `tasks/_probe*.py` and `tasks/_scope_d_readout.py` scripts that shared a common ET-minute computation pattern.
- **What happened**: stats-ml-logic-reviewer independent re-derivation showed every combo's SES_1 (RTH) and SES_2 (overnight) session assignments were inverted. Under corrected TZ: combo-865 RTH Sharpe 0.64 → **2.64**; combo-1298 RTH Sharpe −0.17 → **4.41**; combo-664 overnight Sharpe 1.65 → **1.17**. Probe 3 §4.3 went 0/16 PASS → 8/16 FAIL (rescue fires), F-count 0 → 1. Probe 4 §5 row 2 (session-purity) no longer fires — SES_1 passes for 1298, SES_2 fails. Both signed verdicts (Probe 3 PAPER_TRADE at `b68fe62`, Probe 4 SESSION_CONFOUND at `c419391`) reroute to COUNCIL_RECONVENE under corrected TZ.
- **Root cause**: `data/NQ_1h.parquet` and `data/NQ_1min.csv` timestamps are naive **Central Time** (Barchart vendor export — see `scripts/data_pipeline/update_bars_yfinance.py:37`), NOT UTC. The four buggy scripts all did `ts.dt.tz_localize("UTC")` then `.tz_convert("America/New_York")`. CT-naive localized as UTC then converted to ET shifts every timestamp by ~5–6 hours, which inverts the RTH vs overnight label for the majority of bars. The numerical Sharpes/Welch-t/absolute gates reproduce at 4-decimal precision because they don't touch session labels — the bug lives entirely in the session bucket assignment.
- **How it was born**: `tasks/_agent_bus/probe4_2026-04-22/code-logic-reviewer-scripts.md:55` asserted "The source dataset's `Time` column is UTC" as a premise in the pass-4 review of Probe 4. `tasks/_agent_bus/probe4_2026-04-22/coding-agent-pass4.md:36` implemented `tz_localize("UTC")` matching that premise. The code-logic-reviewer then affirmed "DST-aware via IANA zone; no fixed UTC-offset arithmetic" — a correct DST observation built on a false source-TZ premise. The bug propagated to `_scope_d_readout.py` via copy-paste from `_probe4_readout.py`, and `_probe3_{1h_ritual,15m_nc}.py` had the same pattern even earlier.
- **Fix**: All 4 scripts (`_probe3_1h_ritual.py:186`, `_probe3_15m_nc.py:207`, `_probe4_readout.py:129`, `_scope_d_readout.py:133`) now use `ts.dt.tz_localize("America/Chicago", ambiguous="infer", nonexistent="shift_forward")`. Variable renamed `ts_utc → ts_ct` for semantic clarity. Misleading `# Times are UTC-naive` comment replaced with explicit CT reference + vendor marker pointer.
- **Prevention rule**: Before any new script does `tz_localize`/`tz_convert` on bar data, grep `scripts/data_pipeline/update_bars_yfinance.py` for `LOCAL_TZ` — the vendor-authoritative TZ marker lives there. Source-TZ verification is a **separate** audit step from DST-handling verification; a DST-aware pattern applied to the wrong base TZ is still wrong. The reviewing agent protocol for TZ-sensitive changes must include an explicit source-TZ check against the data-pipeline marker, not just DST-correctness.
- **Also discovered (not fixed — Option B)**: `scripts/param_sweep.py:1567` builds `bar_hour` from the raw CSV hour (= CT hour), so the engine's `SESSION_FILTER_MODE` (`src/strategy.py:83-93`) and `tod_exit_hour` operate on CT hours. Combos {865, 1298, 664} all use `session_filter_mode=0` so engine session filter is inactive for them, but any historical Probe 1 combo with `session_filter_mode != 0` ran with a ~1h-shifted filter from the ET-intent that comments ("core US: 9-15h") imply. Did NOT correct the engine — any shift would invalidate past sweep caches. Documented instead: see `memory/feedback_tz_source_ct.md`.
- **Related files/commands**: `tasks/_probe3_1h_ritual.py`, `tasks/_probe3_15m_nc.py`, `tasks/_probe4_readout.py`, `tasks/_scope_d_readout.py`, `scripts/data_pipeline/update_bars_yfinance.py:37`, `scripts/param_sweep.py:1567`, `src/strategy.py:83-93`, `src/backtest.py:471-472`, `memory/feedback_tz_source_ct.md`, `memory/project_tz_bug_cascade.md`, `tasks/probe3_verdict.md` Amendment 2, `tasks/probe4_verdict.md` Amendment 2, `tasks/scope_d_brief.md` Amendment.

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

---

### 2026-04-21 v11_has_no_mfe_variant

- **What I ran**: Launched `scripts/analysis/build_combo_overlap_labels.py` remotely with `--mfe-parquet data/ml/mfe/ml_dataset_v11_mfe.parquet`, expecting the canonical v2–v10 layout to carry forward to v11.
- **What happened (error summary)**: `FileNotFoundError: combo-features parquet not found` fired first (that arg path was also wrong — see corollary below). After fixing that, the MFE path failed silently because the file does not exist — v11 was never re-run through the dedicated MFE pass that produced `data/ml/mfe/ml_dataset_v{N}_mfe.parquet` for v2–v10.
- **Root cause**: v11 is the friction-aware sweep (CLAUDE.md, `COST_PER_CONTRACT_RT`) — it emits `gross_pnl_dollars`, `friction_dollars`, `mfe_points`, `mae_points`, `entry_bar_idx` **inline** at sweep time, so there is no separate MFE re-run pass and no `/mfe/` variant. The v11 trade-row parquet lives at `data/ml/originals/ml_dataset_v11.parquet`. Secondary issue: the v12 combo-features parquet is at `data/ml/ml1_results_v12/combo_features_v12.parquet`, not `data/ml/combo_features_v12.parquet` — ML#1 outputs live under `ml1_results_v{N}/` per project convention.
- **Fix**: Made two edits to `scripts/analysis/build_combo_overlap_labels.py` (commit `f1822fd`): (a) `DEFAULT_MFE_PARQUET` now points at `data/ml/originals/ml_dataset_v11.parquet`; (b) `_derive_mfe_version_prefix` uses `re.search(r"(v\d+)", name)` instead of string-slicing on a required `_mfe` suffix, so both naming conventions are tolerated. Also updated the plan doc to reflect the correct paths.
- **Prevention rule**: When a script takes a parquet path from one sweep family and joins against another, **do not assume the directory layout is uniform across sweep versions**. v11+ inlines MFE/friction; v2–v10 needed re-runs. Before writing any join script, `ls data/ml/*/ml_dataset_v{N}*.parquet` for all N and check whether the expected file exists. Corollary: when writing filename-parsing code, use a regex (`r"(v\d+)"`) rather than a suffix-anchored slice — sweeps evolve naming conventions and the brittle parser will crash on the next convention shift. Corollary 2: ML#1 artifacts live under `data/ml/ml1_results_v{N}/`, not the root of `data/ml/`. When a script consumes combo-features, default to the `ml1_results_v{N}/` path.
- **Related files/commands**: `scripts/analysis/build_combo_overlap_labels.py:68-71` (defaults), `scripts/analysis/build_combo_overlap_labels.py:201-212` (regex parser), CLAUDE.md §Parameter sweep (v11 inline-MFE note), commit `f1822fd`.

---

### 2026-04-21 auc_parity_is_not_a_ship_gate

- **What I ran**: Phase 3 V3 combo-agnostic ship-blocker audit per `tasks/plan_v3_audit_and_ranker_null.md`. V3 was refit with `global_combo_id` stripped; Phase 2 anti-leak verification PASSED with OOF AUC **0.8293** — near-identical to shipped V3's 0.8293. The plan explicitly treated this AUC preservation as a positive signal that V3's learning was not fundamentally dependent on the leak channel.
- **What happened (finding)**: Phase 3 s6_net MC on the identical Pool B top-50 revealed catastrophic collapse — Sharpe p50 dropped from shipped V3's +1.78 to +0.31, ruin rose from 6.93% to 53.62%, pos_prob fell from 98.7% to 64%. Same AUC, entirely different ship outcome. V3's combo-ID feature was a per-combo P(win) re-weighter that biased calibration upward for the 50 shipped combos — a leak that lives in calibration, not in AUC ranking.
- **Root cause**: AUC is the probability that a random positive is ranked above a random negative across the *entire trade population* (9.99M expanded rows). A small subset of combos (the top-50, a tiny fraction of 588,235 training combos) can have heavily biased absolute P(win) values without moving AUC by more than a rounding digit. The ship pipeline filters on an absolute-threshold E[R] cutoff against those same 50 combos, so the calibration bias leaks directly into filter pass-rates. AUC is blind to this because it's scale-invariant.
- **Fix**: Added this lesson. Amended `CLAUDE.md` §ML#2 production stack to note AUC parity is a precondition, not evidence of ship readiness. New memory `project_v3_combo_id_leak_confirmed.md` encodes the AUC insufficiency finding.
- **Prevention rule**: For any future combo-agnostic refit or leak-removal attempt, **treat OOF AUC parity as a PRECONDITION that unlocks deeper ship-blocker tests, not as ship evidence in itself**. The mandatory deeper test is a pool-filtered Monte Carlo on the identical ship target (Pool B top-50 + s6_net in this project) with identical kernel parameters. If Sharpe p50, sharpe_ci_95, pos_prob, and ruin_prob all survive against the baseline, only then consider the refit ship-safe. Corollary: when a LightGBM model uses a high-cardinality categorical feature (`global_combo_id` has ~588k levels) **AND** the eval basket is a small subset of training combos (50 of 588k), suspect per-combo memorization by default — AUC cannot detect this class of leak.
- **Related files/commands**: `tasks/v3_no_gcid_audit_verdict.md` (full verdict), `data/ml/adaptive_rr_v3_no_gcid/metrics_v3.json` (AUC evidence), `evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s6_mc_combined_ml2_net.ipynb` (MC evidence), `scripts/models/adaptive_rr_model_v3.py:64-69` (feature list with `global_combo_id`).

---

### 2026-04-21 check_universe_ceiling_before_ablation

- **What I ran**: Drafted `tasks/root_ablation_criterion.md` (V3-no-memory refit) to test whether stripping the `prior_wr_*` memorization channels would preserve V3-no-gcid's +0.31 Sharpe uplift. Pre-registered Branch 1 / Branch 2 kill criteria per Option C discipline. Left **UNSIGNED** and ran the Council-directed chairman sweep on friction first.
- **What happened (finding)**: The chairman sweep, together with the prior `tasks/_sharpe_distribution.py` output over `combo_features_v12.parquet`, showed that the **gross Sharpe ceiling across 13,814 v11 combos is 1.108** (one combo, v11_23634, gross 1.108 → net 0.601 at $5 RT). Even at zero friction, the universe contains at most ~1 combo above the Sharpe ≥ 1.0 ship bar. An ML#2 filter of any design can only reweight trades *within* that gross distribution; it cannot manufacture signal that isn't there. The ablation's Branch 1 (PASS) was therefore foreclosed before the refit would have run. I marked the criterion `SUPERSEDED` without executing it.
- **Root cause**: I spent three days designing a careful pre-registered ablation to test whether the ML#2 filter's uplift would survive memorization stripping — without first asking whether the underlying gross distribution contained enough signal for any filter to lift Pool B above ship. The ceiling check is ~20 lines of pandas and a ten-minute read; the ablation plan was hours of design and a 3-hour remote refit. I inverted the order.
- **Fix**: Added `Phase A` in `tasks/todo.md` to formally retire the criterion. Added Phase B/C for the actual load-bearing next probe (cross-instrument viability of the strategy family).
- **Prevention rule**: **Before pre-registering any ablation that tests whether an ML filter / reweighter survives a leak-removal treatment, first compute the gross (zero-friction) Sharpe distribution across the candidate universe under the same eligibility gate as production. If the count of combos with gross Sharpe ≥ (ship bar + friction headroom) is small single digits or zero, the ablation is foreclosed — no filter design can recover what the gross distribution does not contain.** The check is cheap (single pandas pass); the ablation is expensive (design + remote compute + interpretation discipline). Order matters.
- **Related files/commands**: `tasks/root_ablation_criterion.md` (SUPERSEDED header), `tasks/_sharpe_distribution.py` (ceiling evidence), `memory/project_friction_constant_unvalidated.md` (chairman sweep findings), `data/ml/ml1_results_v12/combo_features_v12.parquet` (source data for ceiling check).

---

### 2026-04-21 family_falsification_needs_preregistered_spec

- **What I ran**: Probe 1 — `tasks/probe1_preregistration.md` (signed `d0ee506`), remote sweeps on sweep-runner-1 covering NQ at 15min (3000 combos) and 1h (1500 combos) with a $5/contract RT friction model and three microstructure axes. Then ran `tasks/_probe1_gross_ceiling.py` to compute per-combo gross Sharpe and count combos ≥ 1.3 at ≥ 50 trades.
- **What happened (finding)**: The bar-timeframe axis produced a **9×** increase in N_1.3 (1min = 1 → 15m = 9 → 1h = 4) and a **2.05×** increase in the gross Sharpe ceiling (1.108 → 2.272), but neither timeframe reached the pre-registered gate of N_1.3 ≥ 10 combos. Preregistration §3 tie-breaking is explicit: "`N_1.3 = 9` on both timeframes → Branch A (sunset)." Branch A fires. Strategy family Z-score mean-reversion on NQ/MNQ is falsified across 1min, 15min, and 1h.
- **Root cause**: None — this is the intended use of a preregistration. The lesson is about *process*: because the rule was written and signed before the sweep, a borderline result (N_1.3 = 9, one combo short) could be read mechanically without the temptation to move the goalposts. The earlier ceiling-check lesson (2026-04-21 check_universe_ceiling_before_ablation) established the discipline. This one establishes its follow-through.
- **Fix (process)**: (1) Branch A verdict recorded in `tasks/probe1_verdict.md`. (2) CLAUDE.md updated with a 🛑 falsification banner noting the 1min / 15m / 1h results and the §7.6 "terminal" status that blocks intermediate-timeframe workarounds (30min, 2h, 5min) from being added opportunistically. (3) Fresh LLM Council spawned on the next-fork decision (signal-family swap vs session-structure probe vs project sunset).
- **Prevention rule**: **Family-level falsification is a decision, not an observation — it requires a preregistered decision rule, not just a ceiling readout.** When an earlier probe (the 1min ceiling check) suggests a family is underperforming, do not declare family-level sunset from that single observation. Instead, pre-register the axis along which the family could credibly improve (here: bar timeframe), write a mechanical decision rule with a numerical threshold, have the user sign it, run the experiment, and read the rule. The preregistration does two things a single-experiment readout cannot: (a) it commits to a gate before seeing the data, protecting against motivated interpretation when the result is borderline (e.g. 9 vs 10); (b) it forecloses post-hoc workarounds like "let's also try 30min" (§7.6 "terminal" clause), which would otherwise extend a dying investigation indefinitely.
- **Related files/commands**: `tasks/probe1_preregistration.md` (§3 decision rule + §7 irrevocable commitments), `tasks/probe1_verdict.md` (Branch A verdict), `tasks/_probe1_gross_ceiling.py` (readout), `tasks/probe1_15m_gross_sharpe.csv` + `tasks/probe1_1h_gross_sharpe.csv` (per-combo dumps), `CLAUDE.md` (Signal family status bullet).

