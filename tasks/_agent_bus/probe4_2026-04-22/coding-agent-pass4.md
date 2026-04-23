---
from: coding-agent
run_id: probe4_2026-04-22
pass: 4
timestamp: 2026-04-23T06:29:22Z
for: [code-logic-reviewer, user]
task_summary: Fix C1 (session-filter TZ mismatch), W2 (wrapper set -e stranding), W3 (timeout screen ghost) from code-logic-reviewer-scripts.md. Drop 6-run design to 2 runs + post-hoc ET-minute partitioning in the readout.
files_changed:
  - tasks/_probe4_run_combo.py
  - tasks/_probe4_readout.py
  - tasks/_run_probe4_remote.py
files_created: []
context_loaded:
  - tasks/probe4_preregistration.md
  - tasks/_agent_bus/probe4_2026-04-22/code-logic-reviewer-scripts.md
  - tasks/_agent_bus/probe4_2026-04-22/coding-agent-scripts.md
  - tasks/_probe3_1h_ritual.py (lines 124-140, 180-193, 229-271 — canonical post-hoc ET pattern)
  - tasks/_probe4_run_combo.py (pre-edit state)
  - tasks/_probe4_readout.py (pre-edit state)
  - tasks/_run_probe4_remote.py (pre-edit state)
  - scripts/param_sweep.py (bar loading lines 1448-1475; session_filter propagation verification)
  - src/data_loader.py (split_train_test semantics, lines 46-76)
  - C:/Users/kunpa/.claude/agent-memory/coding-agent/feedback_engine_session_filter_utc_hour.md
  - C:/Users/kunpa/.claude/agent-memory/coding-agent/feedback_dst_aware_session_filter.md
  - C:/Users/kunpa/.claude/agent-memory/coding-agent/feedback_years_span_cross_tf.md
libraries_consulted: []
ready_for_review: true
---

## What I changed

### C1 — session-filter TZ mismatch (CRITICAL)

- `_probe4_run_combo.py::_build_engine_combo` — now forces `session_filter_mode=0` unconditionally. Dropped the `ses_mode` CLI argument (now a single `--combo-id`). Engine-side combo_id encoding simplified from `20_000 + CID*10 + SES` to `20_000 + CID` (2 disjoint engine ids; still disjoint from Probe 3's 10_000 band).
- `_probe4_run_combo.py::main` — emits a single SES_0 JSON + SES_0 per-trade parquet per combo. Empty-trades shim preserved (prior reviewer-verified edge case).
- `_probe4_readout.py` — added `_load_test_bars_with_et_minutes` mirroring Probe 3 exactly: `pd.read_parquet("data/NQ_1h.parquet")` -> `split_train_test(df, 0.8)` -> `test_part.reset_index(drop=True)` -> `ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")` -> `hour*60 + minute`. DST-aware via IANA zone; no fixed UTC-offset arithmetic.
- `_probe4_readout.py::_ses_rth_mask` / `_ses_overnight_mask` — RTH = `et_min ∈ [570, 960)`, GLOBEX = complement. Complement semantics means SES_1 ∪ SES_2 = SES_0 exactly (no dropped trades; §5 row 2 comparison well-covered).
- `_probe4_readout.py::_partition_session_metrics` — resolves each trade's `entry_bar_idx` -> bar `_et_minutes` -> mask. Guards `idx.max() >= len(et_min_arr)` with explicit RuntimeError. Handles `len(trades) == 0` via early return of empty metrics dicts.
- `_probe4_readout.py::main` — swapped 6-JSON ingest for 2 SES_0 JSONs + 2 SES_0 per-trade parquets; SES_1/SES_2 metrics now computed post-hoc. `_route_branch` signature updated to accept dict metrics rather than JSON payloads (functionally identical to pass-3).
- `_run_probe4_remote.py` — `RUNS` replaced with `COMBO_IDS = [1298, 664]`. Artifact pull list reduced proportionally (combo{cid}_SES_0 files only).

### W2 — wrapper `set -e` (WARN)

- `_run_probe4_remote.py::_build_wrapper` — dropped `set -e`. Each engine invocation's `$?` captured into `data/ml/probe4/run_status.json` (minimal shell-written JSON object, e.g. `{"combo1298": 0, "combo664": 0}`). Readout is called unconditionally at the wrapper's end. Fail-fast semantics preserved at the readout boundary: if a run failed before emitting its JSON, `_load_run` raises `FileNotFoundError` with a clear message and no verdict is written. If JSON exists but trades parquet is missing, `_load_ses0_trades` raises the same way.

### W3 — orchestrator timeout screen ghost (WARN)

- `_run_probe4_remote.py::_kill_screen` — new helper: `screen -S probe4 -X quit 2>&1 || true`, idempotent and harmless if the session already died.
- `_run_probe4_remote.py::main` — wrapped body in `try / except KeyboardInterrupt / except Exception / finally`. `_kill_screen` fires on: (a) 30-min timeout (explicit call inside the polling block), (b) KeyboardInterrupt, (c) any other exception. Happy path exits cleanly because the polling loop only breaks when the screen is already dead. `finally: client.close()` guarantees SSH cleanup in every exit path.

### W1 — NOT a code change (per scope)

- `_probe4_run_combo.py:YEARS_SPAN_TEST` — added 2-line comment documenting that `1.4799` is inherited from Probe 3 for cross-probe continuity, true 1h partition span is ~533 days = 1.4593 y, bias is ~1.3% conservative. Per inherited memory `feedback_years_span_cross_tf.md`: "must be documented, not silently fixed." **Verdict document must disclose this discrepancy and its conservative-bias direction.**

## Why (non-obvious)

- **Complement semantics for SES_2**: Prereg §4.4 says "SES_2 = overnight / GLOBEX (excluding RTH)". Probe 3's `_ses_overnight` used `(et_min >= 1080) | (et_min < 570)` = [18:00, 09:30] ET wraparound, which drops the 16:00-18:00 ET window entirely. Probe 4's prereg wording is looser and does not qualify the overnight window with an 18:00 ET start. I chose exact complement (`~_ses_rth_mask`) so SES_1 ∪ SES_2 = SES_0, which is the property needed for §5 row 2's `(SES_2 Sharpe - SES_1 Sharpe) > 1.0` comparison to exhaust the SES_0 edge. A Probe-3-style wraparound would leave the 16:00-18:00 ET window unaccounted for on both sides of the inequality. Flagged for reviewer under "Open questions".
- **Readout duplicates `_compute_metrics`**: ~15 lines identical to `_probe4_run_combo::_aggregate_metrics`. Chosen over a cross-module import because (a) project convention for these single-use probe scripts is self-contained modules; (b) the readout runs in environments where the run-combo module's subprocess machinery is dead weight. Both copies use identical thresholds (`YEARS_SPAN_TEST`, `SHARPE_GATE`, `NTRADES_GATE`, `DOLLARS_GATE`); any future edit must update both sites.
- **`session_filter_mode=0` kept in the combo dict** (not removed): `_COMBO_META_KEYS` in `param_sweep.py:121` requires the key. Removing it would trip the `_missing` check at `param_sweep.py:1488`.

## Docs cited

None — all changes mirror in-repo precedent (Probe 3 `_probe3_1h_ritual.py`); no external library lookups required.

## Open questions for the reviewer

1. **Complement vs wraparound SES_2 semantics** — see "Why" above. Probe 3's `_ses_overnight` was `(et_min >= 1080) | (et_min < 570)` = [18:00, 09:30] ET wraparound. Probe 4 prereg §4.4 says "GLOBEX (excluding RTH)"; I implemented exact complement (`~_ses_rth_mask`). Current choice captures all ex-RTH trades (the 16:00-18:00 ET gap goes into SES_2) and preserves §5 row 2 comparison coverage. Should this match Probe 3 exactly instead (loses 16:00-18:00 ET)? Current choice defensible on prereg wording.
2. **Wrapper `run_status.json` is diagnostic only** — the readout doesn't read it. The fail-fast path relies on `_load_run` raising when a run failed before emitting its JSON. If the user wants the readout to explicitly refuse on non-zero exit codes (even if a stale JSON somehow persists), say so and I'll add a check. Current behavior: readout refuses iff the JSON or the trades parquet is missing (i.e. the natural and sufficient condition).

## Ready-for-review

Yes. Three scripts AST-parse clean (`python -c "import ast; ast.parse(...)"`). No engine invocation performed. No remote calls. No commits. Self-review pass completed and emitted at `code-logic-reviewer-pass4.md` alongside this artifact. Awaiting user authorization to launch remote per `_run_probe4_remote.py`.
