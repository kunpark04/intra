---
from: code-logic-reviewer
run_id: probe4_2026-04-22
timestamp: 2026-04-23T00:00:00Z
for: [user]
reviewing:
  - tasks/_probe4_run_combo.py
  - tasks/_probe4_readout.py
  - tasks/_run_probe4_remote.py
scope_reviewed:
  - tasks/_probe4_run_combo.py:1-295
  - tasks/_probe4_readout.py:1-304
  - tasks/_run_probe4_remote.py:1-223
  - scripts/param_sweep.py:121, 886, 1567, 1181, 972-973 (session_filter_mode propagation path)
  - src/strategy.py:82-93 (engine-side session filter implementation)
  - tasks/_probe3_1h_ritual.py:89, 124-140, 180-193, 229-271 (Probe 3 precedent comparison)
critical_count: 1
warn_count: 3
info_count: 2
launch_recommendation: BLOCK
cross_references:
  - tasks/_agent_bus/probe4_2026-04-22/coding-agent-scripts.md
  - tasks/probe4_preregistration.md
---

## Goal Understanding

Probe 4 is a property test — it asks whether the cross-TF-coherence heuristic that admitted combo-865 to Probe 2 is genuine edge or a Stage-1 selection artifact. The three scripts under review must, on the 1h test partition 2024-10-22 → 2026-04-08:

1. Run combo-1298 (Probe 1 rank-1, 1h gross Sharpe 2.272) and combo-664 (Probe 1 rank-6, 1h gross Sharpe 1.200, fails 1.3 pre-gate) with their frozen parameter dicts, each decomposed by session filter into SES_0=all, SES_1=RTH (09:30–16:00 ET), SES_2=GLOBEX overnight (6 engine runs total).
2. Apply §4.1/§4.2 absolute gates (net_sharpe ≥ 1.3 AND n_trades ≥ 50 AND net $/yr ≥ $5,000), the §4.3 Welch-t primary gate on per-trade net PnL (one-tailed, H1: 1298 > 664, threshold t ≥ 2.0), and the §5 ordered branch routing (INCONCLUSIVE / SESSION_CONFOUND / PROPERTY_VALIDATED / COUNCIL_RECONVENE).
3. Run entirely on sweep-runner-1 via paramiko orchestrator with remote git-sync, no SFTP-patching of `src/`.

Correct behavior means: the 6 runs genuinely produce RTH-only / GLOBEX-only entry subsets in SES_1 / SES_2; the Welch-t and absolute-gate arithmetic match the prereg formula exactly; branch routing fires in §5 table order.

## Scope Reviewed

- **`tasks/_probe4_run_combo.py`** (~295 LOC) — per-(combo, session) backtest invocation, metric aggregation, optional SES_0 per-trade parquet emission.
- **`tasks/_probe4_readout.py`** (~304 LOC) — 6-JSON + 2-parquet aggregation, Welch-t computation, §5 ordered branch routing.
- **`tasks/_run_probe4_remote.py`** (~223 LOC) — paramiko orchestrator (git stash + reset --hard origin/master, .so wipe, wrapper write, screen launch, 30-min poll at 60s cadence, SFTP pull).
- **Engine propagation spot-check** in `scripts/param_sweep.py` (line 886: `ns.SESSION_FILTER_MODE = int(combo.get("session_filter_mode", 0))`), `src/strategy.py:82-93` (engine-side filter body), and `scripts/param_sweep.py:1567` (`bar_hour` construction).
- **Probe 3 precedent** in `tasks/_probe3_1h_ritual.py:124-140, 180-193, 229-271` for the session-filter mechanism that Probe 3 actually used.

## Findings

### CRITICAL (must fix before launch)

#### C1 — `session_filter_mode` override does NOT match prereg SES_1/SES_2 semantics. Probe 4 will silently produce UTC-hour-of-day subsets, not RTH/GLOBEX subsets.

- **Location**: `tasks/_probe4_run_combo.py:123` (`c["session_filter_mode"] = ses_mode`); engine consumer at `scripts/param_sweep.py:886` → `src/strategy.py:82-93`.
- **Issue**: The engine's `SESSION_FILTER_MODE` branches (in `src/strategy.py`) do **UTC hour-of-day** filtering against the `bar_hour` column, which is built from a **timezone-naive** timestamp at `scripts/param_sweep.py:1567`:
  ```python
  hour_np = pd.DatetimeIndex(time_np).hour.to_numpy(dtype=np.int64)
  ```
  The `time` column is loaded timezone-naive from `data/NQ_1min.csv` (see `src/data_loader.py:36-43` — no `tz_localize`/`tz_convert` anywhere). The source dataset's `Time` column is **UTC**, so `bar_hour` is the UTC hour. The engine's branch mapping is:
  - `SESSION_FILTER_MODE == 1` → `hour >= 7 & hour < 20` (UTC 07:00–20:00 = ET 02:00/03:00–15:00/16:00)
  - `SESSION_FILTER_MODE == 2` → `hour >= 9 & hour < 16` (UTC 09:00–16:00 = ET 04:00/05:00–11:00/12:00 — **NOT RTH**)
  - `SESSION_FILTER_MODE == 3` → `hour >= 20 | hour < 7` (UTC 20:00–06:00 = ET ~15:00/16:00–01:00/02:00 — overnight-ish but not GLOBEX-ex-RTH)

  The preregistration §4.4 specifies:
  - SES_1 = RTH 09:30–16:00 ET
  - SES_2 = GLOBEX overnight (excluding RTH)

  Passing `ses_mode=1` to the engine therefore filters to "UTC 7–19, which is roughly ET pre-noon" — **not RTH**. Passing `ses_mode=2` to the engine filters to "UTC 9–15, which is ET 04:00–11:00" — **this is almost entirely GLOBEX overnight, the inverse of RTH**. The two sub-sessions are roughly swapped and neither boundary is RTH's 09:30 ET half-hour cut.

  Probe 3 knew this and avoided it by leaving `session_filter_mode=0` in the engine and applying a **post-hoc ET-minute filter on the SES_0 trade table** (see `tasks/_probe3_1h_ritual.py:124-140` `_ses_rth_only`/`_ses_overnight`, and `_load_test_bars_with_et_minutes` at line 180, which computes `et_min = ts_et.hour*60 + ts_et.minute` after `tz_convert("America/New_York")`). Probe 4's `_build_engine_combo` inverts this design and triggers the engine-side UTC-hour filter that does not match the prereg's session labels.
- **Why it matters**: Every SES_1 and SES_2 row in §4.4 session decomposition, plus the SES_2 arm of the §5 row-2 SESSION_CONFOUND branch, will be computed on **the wrong bar subset**. The probe will still produce numbers and a JSON-addressable branch, but the "RTH vs GLOBEX" reading will be meaningless. A real SESSION_CONFOUND signal (edge concentrated in overnight-GLOBEX) would be **missed entirely** because the engine's `ses_mode=2` filter excludes GLOBEX overnight. Consumers of the readout (verdict authors, council) will draw wrong conclusions from labeled-but-mis-filtered data.
- **Suggested fix**: Match Probe 3's pattern exactly — keep `session_filter_mode=0` on the engine side, run the engine **once per combo** on SES_0 only, then apply post-hoc ET-minute filters on the resulting trade table in `_probe4_readout.py` using `entry_bar_idx` → `et_minute` lookup into the 1h test partition. This reduces 6 engine runs to 2, saves ~8 min wall-clock, and is semantically correct. Compare `tasks/_probe3_1h_ritual.py:_apply_cells` (lines 229–271) for the exact pattern. The `_probe4_run_combo.py` invocations for `ses_mode ∈ {1, 2}` become obsolete; the readout gains an `_apply_session_filter(entry_bar_idx, et_min_arr, filter_fn)` helper that partitions `net_pnl_dollars` before re-running `_compute_cell_gate`.

  Alternate (not recommended) fix if the 6-run structure is preserved for traceability: keep `session_filter_mode=0` on all 6 engine runs (so all runs produce the SES_0 full-session trade set — identical outputs) and still apply the ET-minute post-hoc filter inside `_probe4_readout.py`. But this wastes 4 runs of compute and is just the Probe 3 pattern in disguise.

  **Either fix requires re-authoring the three scripts and the engine-combo-id encoding strategy.** Do not launch remote as-written.

  **Note on the handoff's open question #1**: coding-agent explicitly flagged "worth a reviewer spot-check that the override branch actually engages the RTH/GLOBEX filter logic at engine runtime". The answer is "it engages a filter but with the wrong timezone and wrong boundaries". The handoff was exactly right to flag this.

### WARN (fix or document)

#### W1 — `YEARS_SPAN_TEST = 1.4799` has ~7.5-day discrepancy from the stated partition (2024-10-22 → 2026-04-08 = 533 calendar days = 1.4593 years).

- **Location**: `tasks/_probe4_run_combo.py:56` (`YEARS_SPAN_TEST = 1.4799`); used at lines 164, 165.
- **Issue**: The constant was inherited from Probe 3 unchanged (coding-agent's handoff open question #2 acknowledges this). `1.4799 * 365.25 = 540.5 days`, which is 7.5 calendar days longer than the partition. The discrepancy likely traces back to a 1-min partition calculation or a calendar-vs-trading-day mismatch in the originating computation.
- **Why it matters**: The direction of the error is conservative (biases net_sharpe ~0.7% downward and net $/yr ~1.4% downward vs the true 1.4593-year window), so none of the absolute gates would be unduly relaxed. However, the Welch-t gate is **independent** of `YEARS_SPAN_TEST` (it's a per-trade statistic), so the primary gate is unaffected. Documentation consistency matters because the verdict will quote annualized numbers and a future auditor will compare `YEARS_SPAN_TEST` to the partition and notice. This is not CRITICAL because the exactly-identical biasing was accepted at the Probe 3 signing.
- **Suggested fix**: If the Probe 3 verdict and this prereg both locked `1.4799`, leave it and add a 1-line comment at line 56 citing Probe 3 continuity (e.g., `# Matches Probe 3 constant for result continuity; true calendar span is 533 days = 1.4593y. Bias direction conservative.`). Otherwise fix to `1.4593`. Prereg §6.1 forbids mid-flight threshold edits but this is an annualization constant not a gate threshold, so it could be corrected without signing violation — confirm with user first. **Do not silently fix.**

#### W2 — Wrapper `set -e` + 6 sequential engine invocations: if run #3 fails, readout step is skipped, leaving partial JSONs with no aggregator output.

- **Location**: `tasks/_run_probe4_remote.py:68` (`set -e`), wrapper loop lines 77-87 generates 6 invocations without `|| true`, readout at line 92.
- **Issue**: With `set -e`, any non-zero exit from `_probe4_run_combo.py` aborts the wrapper before the readout step. In a partial failure, sftp pulls some per-run JSONs + parquets but no `readout.json`. The orchestrator's failure mode on this path is: 30-min poll completes (screen exits on the failed run), readout.json sftp returns FileNotFoundError, verdict author sees partial artifacts with no aggregator. Probe 4 design is "6 runs → 1 readout"; partial runs have no defined interpretation.
- **Why it matters**: Probe 4 compute budget is ~12 min (per §8.4) and runs are ~2 min each. A failure on run 3 wastes runs 1-2 output artifacts with no summary. This is not a CRITICAL bug because the orchestrator does tail the log (line 187) and per-run JSONs are still produced, but the readout is the contract with the verdict step.
- **Suggested fix**: Either (a) remove `set -e` and rely on each `_probe4_run_combo.py` to raise on its own invariants, then always run the readout (which has its own FileNotFoundError handler at `_probe4_readout.py:69-73`); or (b) keep `set -e` but wrap each invocation in `|| { echo "FAIL"; exit 1; }` and unconditionally run the readout in a `trap` or `set +e` block at the end. Option (a) is simpler.

#### W3 — `MAX_WAIT_SECONDS = 30*60` (30 min) orchestrator timeout: no "screen still alive, kill it" handler, just a WARN log.

- **Location**: `tasks/_run_probe4_remote.py:61, 168-184`.
- **Issue**: Lines 182-184 log `"[WARN] 30-min abort threshold reached. Investigate manually"` when polling exceeds the cap, but the remote `screen` continues running and can collide with a second orchestrator invocation. Prior-probe runners followed this exact pattern, so it's not a regression, but it means a stuck screen persists silently on the remote. For Probe 4 with a 12-min expected wall-clock, this is very unlikely to trip, but if one of the 6 engine runs hangs (e.g., OOM with systemd-run OOMKilling being caught differently than expected), the user has to SSH manually.
- **Why it matters**: Minor. Probe 3 ran in the same way; the verdict process survived. Still worth flagging for documentation.
- **Suggested fix**: Either add a `screen -S probe4 -X quit` on timeout, or leave as-is and document "post-timeout remediation: `ssh root@195.88.25.157 'screen -S probe4 -X quit'`" in the script header.

### INFO

#### I1 — `_load_param_dict` falls back to parquet lookup, but the parquet path loads the full v11 1h parquet into memory just to find one combo's dict row.

- **Location**: `tasks/_probe4_run_combo.py:89-117`.
- **Issue**: The fallback at line 90 (`pd.read_parquet(V11_1H_PARQUET)`) loads the entire dataset (~tens of MB). Since the JSON is committed and lives alongside the preregistration (per handoff decision #3), the fallback is almost certainly dead code. If it triggers, the perf cost is acceptable (one-time per invocation) but worth noting.
- **Suggested fix** (optional): Replace the full-parquet read with a `pq.read_table(..., filters=[("combo_id", "=", combo_id)])` pushdown query if pyarrow version supports it, or just leave a comment that this path is a safety net and not expected to fire in production.

#### I2 — Comment-level inconsistency: `_probe4_readout.py:23` says `welch_gate_pass == False` on row 1, but the preregistration §5 row 1 says `Welch-t < 2.0` (which is also false). Not a bug — same semantics — but a future maintainer might read "welch_gate_pass == False" and miss the insufficient-n case also routes there.

- **Location**: `tasks/_probe4_readout.py:23, 151`.
- **Issue**: The code comment and prereg table disagree on phrasing. Code says `welch_gate_pass == False`, prereg says `Welch-t < 2.0`. Both evaluate the same in passing flow, but `welch_gate_pass == False` is more expansive (also covers insufficient-n → false). This is arguably correct (the insufficient-n defense feeds into row 1 routing), but worth a 1-line comment tying the two together for clarity.
- **Suggested fix**: At `_probe4_readout.py:151`, add inline `# welch_gate_pass == False includes the insufficient-n case (§4.3 defensive default).`

## Checks Passed

- **CRITICAL #1 — Welch-t formula**: Verified at `_probe4_readout.py:116-126`. Numerator is `mean_1298 - mean_664` (H1: 1298 > 664), variance uses `pnl_a.var(ddof=1)` (sample variance), denominator is `sqrt(var_a/n_a + var_b/n_b)` (correct pooled SE, not any other variant), gate is `t >= WELCH_T_GATE` (one-tailed, not absolute value).
- **CRITICAL #2 — Welch insufficient-n defense**: Verified at `_probe4_readout.py:105-115`. If either `n_a < 50` or `n_b < 50`, returns `welch_t: None` AND `welch_gate_pass: False` (never `null`). Zero-variance degenerate case at lines 118-125 also returns `welch_t: None` + `welch_gate_pass: False`. The dict is merged into `base` (which has `n_1298`/`n_664`/`mean_*`/`var_*` fields) so the full insufficient-n context is preserved.
- **CRITICAL #3 — §5 routing order**: Verified at `_probe4_readout.py:135-189`. Routing is implemented as 5 sequential `if ... return ...` blocks, **not** a dict or list-of-tuples. Row 1 (INCONCLUSIVE) → Row 2 (SESSION_CONFOUND) → Row 3 (PROPERTY_VALIDATED) → Row 4 (COUNCIL_RECONVENE both-pass) → Row 5 (catch-all COUNCIL_RECONVENE). Row 2 **precedes** Row 3 (as prereg requires). Row 4 **follows** Rows 2 and 3. First match wins via early `return`.
- **CRITICAL #4 — Branch strings**: Verified at lines 153, 161, 170, 179, 185. Exact string set is `{INCONCLUSIVE, SESSION_CONFOUND, PROPERTY_VALIDATED, COUNCIL_RECONVENE}`. No `PROPERTY_FALSIFIED`. Case-sensitive, no trailing whitespace.
- **CRITICAL #5 — Absolute gate arithmetic**: Verified at `_probe4_run_combo.py:159-178`. Net Sharpe formula is `(mean/std) * sqrt(n/years)` using `std(ddof=1)`. Net $/yr is `mean * n / years`. Sub-gate thresholds match prereg §4.1/§4.2 exactly (Sharpe ≥ 1.3, n ≥ 50, $/yr ≥ $5,000). `abs_pass = sharpe_pass AND n_trades_pass AND dollars_pass` (conjunction of all three). Friction is already baked into `net_pnl_dollars` by the v11 sweep engine (`COST_PER_CONTRACT_RT` = $5 RT, confirmed at `scripts/param_sweep.py:901`); sizing is fixed $500 risk per trade (v11 sweep convention, not configurable here).
- **CRITICAL #6 — Engine combo_id collisions**: Verified via arithmetic — 1298→{32980, 32981, 32982}, 664→{26640, 26641, 26642}. All 6 unique, all disjoint from Probe 3's 10_000-band (`10_000 + i` for param_nbhd, `10_200 + i` for 1h_ritual). No aliasing.
- **§5.2 narrow-miss handling**: Verified at `_probe4_readout.py:225-229` — `narrow_miss_flag` is set when row 5 fires AND `gate_1298_abs_pass == False` AND `welch_gate_pass == True`, with an explanatory note emitted at lines 267-272. Matches prereg §5.2: routes to COUNCIL_RECONVENE via row 5 (not promoted to PROPERTY_VALIDATED) and flagged explicitly for the verdict document.
- **Empty parquet schema**: Verified at `_probe4_run_combo.py:282-285`. Empty SES_0 per-trade parquet is written with `(net_pnl_dollars: float64, entry_bar_idx: int64)` schema. Readout only consumes `net_pnl_dollars` (line 85) so no concat-time type promotion hazard; insufficient-n branch at readout:105 fires cleanly on `n_a = 0 < 50`.

## Launch Recommendation

**BLOCK.**

Rationale: CRITICAL #1 (session-filter TZ mismatch) is a diagnostic correctness bug that silently renders SES_1 and SES_2 decomposition meaningless against the preregistration's labels. The probe will run, produce a JSON, and route to a branch — but every per-session number in §4.4 decomposition, including the SES_2 vs SES_1 gate in §5 row 2 (SESSION_CONFOUND), will be computed on the wrong bar subset. Remote launch as-authored would require re-running Probe 4 from scratch after fix. Fix by mirroring Probe 3's post-hoc ET-minute filter pattern (leave `session_filter_mode=0` on the engine; filter the SES_0 trade table post-hoc using `entry_bar_idx` → `et_min` lookup). This change reduces the 6-run design to 2 engine runs + readout post-processing, which is not a prereg edit (§4.4 specifies a session decomposition readout, not an engine-run count — coding-agent's decision #1 to use 6 runs is a design choice, not a prereg commitment).

The 5 other CRITICAL checks plus all §5 routing and Welch-t math pass cleanly. Once C1 is addressed and W1 (YEARS_SPAN_TEST discrepancy) is either documented or corrected with user sign-off, the suite is cleared for remote launch.

## Notes on the three coding-agent open questions

1. **`session_filter_mode` engine propagation** — coding-agent was right to flag this. The propagation path is correct (`combo dict` → `ns.SESSION_FILTER_MODE` → `src/strategy.py:83` — wire is intact), but the semantics at the consumer (UTC-hour-of-day, not ET-minute) **do not match the prereg's SES_1/SES_2 labels**. See C1 above. The "all 6 runs would produce identical SES_0/SES_1/SES_2 outputs" worst case does not materialize (the engine does filter, and SES_1/SES_2 outputs will differ), but they'll differ along the **wrong axis**, which is worse than identical outputs because the mismatch is silent.

2. **`YEARS_SPAN_TEST = 1.4799` reused unchanged** — flagged as W1. The constant is in one location (`_probe4_run_combo.py:56`), not duplicated. It participates in two compute sites (lines 164, 165) inside the same function, so there is no drift risk across scripts. The 7.5-day discrepancy biases numbers conservatively (lower Sharpe, lower $/yr). Recommend documenting rather than silently fixing, per Probe 3 continuity.

3. **Empty trades parquet on SES_0 zero-trades** — verified safe. Empty parquet has the canonical `(net_pnl_dollars: float64, entry_bar_idx: int64)` schema (lines 282-285). Readout's `_load_ses0_trades` extracts `df["net_pnl_dollars"].to_numpy()`, which returns an empty float64 array of size 0. `_welch_t(n_a=0, n_b=...)` trips the `n_a < WELCH_MIN_N` branch at line 105 → returns `welch_t: None`, `welch_gate_pass: False`. This is exactly the insufficient-n defense working as designed. No pa.concat is involved (readout loads each parquet independently). All good.
