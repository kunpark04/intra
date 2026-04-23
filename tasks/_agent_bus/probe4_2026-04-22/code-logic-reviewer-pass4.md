---
from: code-logic-reviewer
run_id: probe4_2026-04-22
timestamp: 2026-04-23T06:29:22Z
for: [user]
subject: Self-review pass on pass-4 fixes to tasks/_probe4_{run_combo,readout}.py + tasks/_run_probe4_remote.py
reviewing:
  - tasks/_probe4_run_combo.py
  - tasks/_probe4_readout.py
  - tasks/_run_probe4_remote.py
scope_reviewed:
  - C1 fix: engine session-filter dropped; post-hoc ET-minute partitioning moved to readout
  - W2 fix: wrapper set -e removed; run_status.json emitted; readout remains the fail-fast boundary
  - W3 fix: screen cleanup on timeout + KeyboardInterrupt + unexpected exit via try/except/finally
critical_count: 0
warn_count: 1
info_count: 2
launch_recommendation: PROCEED WITH FIXES
cross_references:
  - tasks/_agent_bus/probe4_2026-04-22/coding-agent-pass4.md
  - tasks/_agent_bus/probe4_2026-04-22/code-logic-reviewer-scripts.md (the pre-fix review I am responding to)
  - tasks/probe4_preregistration.md
  - tasks/_probe3_1h_ritual.py (canonical post-hoc ET filter precedent)
---

## Role-switch declaration

This is a self-review pass: the same merged agent that authored the pass-4 changes is now reviewing them. Language is deliberately adversarial to the implementation; findings are graded against the prereg and the original code-logic-reviewer-scripts.md BLOCK verdict.

## Findings on the three fixes

### C1 — session-filter TZ mismatch: FIXED

The implementation now:

1. Keeps `session_filter_mode=0` on every engine combo dict (`_probe4_run_combo.py::_build_engine_combo`). Verified: the field is present at value 0 so `_COMBO_META_KEYS` at `param_sweep.py:121` does not trip.
2. Runs the engine exactly once per combo (SES_0 baseline). Verified: `_run_probe4_remote.py::COMBO_IDS = [1298, 664]`, 2 engine invocations.
3. Partitions SES_0 trades into SES_1 (RTH) / SES_2 (GLOBEX) post-hoc in `_probe4_readout.py`, using `tz_convert("America/New_York")` on the 1h bar timeline. Verified: `_load_test_bars_with_et_minutes` is a line-for-line port of `_probe3_1h_ritual.py:180-193`, which is the canonical precedent named in both the original review and the pattern memory.

**DST correctness check**: `tz_convert("America/New_York")` hands off to IANA zone tables via pytz/zoneinfo, so EDT/EST transitions during the 2024-10-22 -> 2026-04-08 window (which spans three DST transitions) are handled transparently. There is no fixed UTC-offset arithmetic — the trap flagged in `feedback_dst_aware_session_filter.md` is avoided.

**Bar timeline alignment check**: Both the engine (`scripts/param_sweep.py:1459-1468`) and the readout (`_probe4_readout.py::_load_test_bars_with_et_minutes`) load `data/NQ_1h.parquet`, apply `split_train_test(df, 0.8)`, and `reset_index(drop=True)` on the test partition. The engine's `entry_bar_idx` indexes into that reset partition; the readout's `_et_minutes` array is built from the same reset partition. Indices line up exactly. Verified by reading both sides.

**Edge case — empty SES_1 after partitioning**: If combo-664's trades happen to land entirely overnight (unlikely but possible), SES_1 RTH has zero trades. `_compute_metrics(np.array([]))` returns `abs_pass=False` with n=0. `_route_branch` reads `ses_1_1298_metrics["abs_pass"]` which is False, so row 2 SESSION_CONFOUND's `(not ses1_pass)` condition fires correctly. No NaN propagation, no division-by-zero.

**Edge case — entry_bar_idx out of bounds**: `_partition_session_metrics` raises RuntimeError with a clear message if `idx.max() >= len(et_min_arr)`. This is a loud failure path; the readout terminates before writing a misleading verdict.

**Open question #1 from pass-3 handoff**: The prior pass-3 handoff explicitly flagged "worth a reviewer spot-check that the override branch actually engages the RTH/GLOBEX filter logic at engine runtime". The C1 fix addresses this by removing the engine-side override entirely — the concern is now moot. The original question sidestepped "does the engine do the right thing?" in favor of "don't trust the engine's semantics; build RTH/GLOBEX in the readout". That is the correct resolution.

### W2 — wrapper set -e: FIXED

The wrapper no longer uses `set -e`. Each engine invocation's exit code is captured into `data/ml/probe4/run_status.json`. The readout is invoked unconditionally at the wrapper's end.

**Fail-fast boundary check**: If `_probe4_run_combo.py --combo-id 1298` fails (non-zero exit), `combo1298_SES_0.json` will not be written (the script's final `json_out.write_text(...)` is unreachable on an earlier raise). The subsequent engine invocation for 664 runs regardless. The readout then fires; its `_load_run(1298)` raises `FileNotFoundError("Per-run readout missing: ...")` and the readout exits non-zero, emitting NO `readout.json`. From the orchestrator's perspective, the SFTP pull will report `[miss] readout.json` and the user sees a clear signal that the run failed. Fail-fast is preserved at the readout boundary, which is exactly what the original review asked for.

**Trailing-comma correctness in run_status.json**: The shell-generated file has the form `{\n  "combo1298": 0,\n  "combo664": 0\n}` (first iteration writes comma, last iteration omits comma per `is_last` flag). Rendered by hand and parsed mentally — valid JSON.

### W3 — orchestrator timeout screen ghost: FIXED

`_kill_screen` is called on the following exit paths:

1. 30-min timeout: explicit `_kill_screen(client)` call inside the polling block when `timed_out = True`, followed by `launched = False` to avoid double-kill via the `finally` path.
2. KeyboardInterrupt: `except KeyboardInterrupt` branch calls `_kill_screen` then re-raises. User gets partial log output and the screen is cleaned up.
3. Unexpected exception: `except Exception` branch calls `_kill_screen` then re-raises. Same contract.
4. Happy path: polling loop breaks when `screen -ls` reports 0 live sessions; the screen is already dead, so `_kill_screen` is NOT called (avoids noisy warnings on a clean exit). `finally: client.close()` still fires to close SSH.

The `|| true` tail on the `screen -S probe4 -X quit` command makes it idempotent — if the screen already exited, the quit command's non-zero return is suppressed.

**Gap: SSH disconnect during polling**: If the SSH connection drops mid-poll and paramiko raises an exception other than KeyboardInterrupt, the `except Exception` branch fires. It attempts `_kill_screen` on the same (now-possibly-dead) client, which will swallow the paramiko error inside `_kill_screen`'s `try/except`. The screen may continue running on the remote with no local visibility. This is a pre-existing orchestrator pattern (same in Probe 3), not a regression, but worth noting.

## Cross-reference check against pass-3 open questions

Pass-3 coding-agent handoff (`coding-agent-scripts.md`) flagged three open questions:

1. **`session_filter_mode` engine propagation** — addressed by C1 fix (engine-side filter is now bypassed entirely). The concern "worth a reviewer spot-check that the override branch actually engages the RTH/GLOBEX filter logic at engine runtime" is moot.
2. **`YEARS_SPAN_TEST = 1.4799` reused unchanged** — documented inline per `feedback_years_span_cross_tf.md`. Still must be disclosed in the verdict document.
3. **Empty trades parquet on SES_0 zero-trades** — verified safe. Readout's `_partition_session_metrics` short-circuits on `len(trades) == 0` before `idx.max()`.

## Remaining findings

### WARN

#### W-new — SES_2 semantics differ from Probe 3 (raised by coding-agent in the pass-4 handoff; reviewer agrees it's worth user confirmation)

- **Location**: `_probe4_readout.py::_ses_overnight_mask` — `return ~_ses_rth_mask(et_min)` (exact complement of RTH).
- **Issue**: Probe 3 `_probe3_1h_ritual.py:127-128` defined `_ses_overnight` as `(et_min >= 1080) | (et_min < 570)` = [18:00, 09:30] ET wraparound, which excludes the 16:00-18:00 ET late-afternoon window from BOTH sides (SES_1 ends at 16:00 ET, SES_2 starts at 18:00 ET). Probe 4's pass-4 implementation uses the exact complement so SES_1 ∪ SES_2 = SES_0 — the 16:00-18:00 ET bars fall into SES_2.
- **Why it matters**: The prereg §4.4 language is "SES_2 = overnight / GLOBEX (excluding RTH)" with no explicit start-of-overnight boundary. The complement-of-RTH reading is defensible. The trade-off:
  - **Complement-of-RTH (current pass-4)**: SES_1 ∪ SES_2 covers all SES_0 trades. §5 row 2's SESSION_CONFOUND comparison exhausts the SES_0 edge. A combo whose edge lives in the 16:00-18:00 ET window is counted as SES_2. Probe 4 numbers may not be directly comparable to Probe 3's reported overnight SES_2.
  - **Probe 3 wraparound**: SES_1 ∪ SES_2 < SES_0 (16:00-18:00 ET bars orphaned). §5 row 2 compares two subsets that together do not cover SES_0. Probe 4 numbers directly comparable to Probe 3's SES_2 overnight (the 3.45× per-trade ratio in the Probe 3 verdict).
- **Severity**: WARN, not CRITICAL, because the prereg wording supports either reading and combo-1298 / combo-664 are new combos not directly compared to combo-865's SES_2 metric. The verdict document must disclose which SES_2 definition was used. **Reviewer recommendation**: either get user confirmation on complement-of-RTH, or fall back to Probe 3's wraparound definition for continuity with Probe 3's §4.4 readout.
- **Suggested fix (if user prefers Probe 3 parity)**:
  ```python
  def _ses_overnight_mask(et_min: np.ndarray) -> np.ndarray:
      """SES_2 GLOBEX: 18:00 - 09:30 ET wraparound (matches Probe 3)."""
      return (et_min >= 1080) | (et_min < 570)
  ```
  Note: this reintroduces the 16:00-18:00 ET gap and must be disclosed in the verdict.

### INFO

#### I1-new — readout duplicates `_compute_metrics`

- **Location**: `_probe4_readout.py::_compute_metrics` (lines 169-206) vs `_probe4_run_combo.py::_aggregate_metrics` (lines 144-179). Two copies, ~15 lines each, identical semantics, identical constants (`YEARS_SPAN_TEST`, `SHARPE_GATE`, `NTRADES_GATE`, `DOLLARS_GATE`).
- **Issue**: Drift risk if a future edit updates one without the other.
- **Suggested fix**: non-blocking. Could be consolidated by importing from `_probe4_run_combo.py`, but that module's module-level imports include subprocess/param_sweep machinery that bloats the readout's startup. Acceptable as-is for a single-use probe script. Current convention in the Probe 3 scripts uses the same duplication pattern.

#### I2-new — `run_status.json` is never consumed

- **Location**: `_run_probe4_remote.py::_build_wrapper` writes `data/ml/probe4/run_status.json`; `_probe4_readout.py` never reads it.
- **Issue**: The file is diagnostic-only. If the user expects the readout to refuse based on exit codes (rather than missing JSON), the current design won't do that. The fail-fast path relies on the more natural condition ("JSON missing").
- **Suggested fix**: either (a) leave as-is (simplest; readout already fail-fasts on missing JSON — the original review's W2 asked for fail-fast at readout boundary which is achieved), or (b) add a 3-line check at readout start that reads run_status.json and raises on any non-zero entry. Current choice is (a). The user should confirm this satisfies the W2 intent.

## Checks passed (carry-over from pass-3 review, re-verified on pass-4)

- **Welch-t formula**: unchanged from pass-3. Numerator `mean_1298 - mean_664`, variance uses `ddof=1`, denominator `sqrt(var_a/n_a + var_b/n_b)`, gate `t >= 2.0` one-tailed. Re-verified at `_probe4_readout.py:217-258`.
- **Welch insufficient-n defense**: unchanged. `n < 50` on either side returns `welch_t=None`, `welch_gate_pass=False`. Zero-variance degenerate case guarded. Re-verified at `_probe4_readout.py:229-250`.
- **§5 routing order**: unchanged. Row 1 -> Row 2 -> Row 3 -> Row 4 -> Row 5, first match wins via early return. Re-verified at `_probe4_readout.py:261-308`.
- **Branch string set**: unchanged. `{INCONCLUSIVE, SESSION_CONFOUND, PROPERTY_VALIDATED, COUNCIL_RECONVENE}`.
- **§5.2 narrow-miss flagging**: unchanged at the readout's main body.
- **Absolute gate arithmetic**: unchanged. `net_sharpe = (mean/std) * sqrt(n/years)` with `ddof=1`; `net_$/yr = mean * n / years`; `abs_pass = sharpe AND n_trades AND dollars`.
- **Engine combo_id disjointness**: 1298 -> 21298, 664 -> 20664 (under the simplified `ENGINE_ID_BASE + CID` scheme). Both disjoint from Probe 3's 10_000 band. OK.
- **Empty parquet schema**: `(net_pnl_dollars: float64, entry_bar_idx: int64)` preserved on the zero-trades path in `_probe4_run_combo.py`.

## Launch recommendation

**PROCEED WITH FIXES.**

Rationale: All three target findings (C1 critical, W2 warn, W3 warn) are fixed and the fixes follow the canonical Probe 3 precedent. The only remaining item is the WARN-new on SES_2 semantics (complement-of-RTH vs Probe 3's wraparound), which is a defensible implementation choice rather than a correctness bug — but the user should confirm it before remote launch to avoid verdict-writing surprises.

**Caveat**: This is a self-review by the same merged agent that wrote the changes. The user may want a second pair of eyes (standalone reviewer re-run, or an LLM Council if the stakes warrant) before authorizing the remote launch of a signed preregistration's implementation path.
