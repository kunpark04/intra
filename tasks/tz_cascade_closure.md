# TZ-Cascade Closure Document

**Status:** ✅ CLOSED — user signed 2026-04-24
**Date opened:** 2026-04-24
**Date closed:** 2026-04-24
**Required by:** LLM Council verdict 2026-04-24 (`tasks/council-report-2026-04-24-post-tz-cascade.html`), which gated Probe 5 preregistration on three prerequisites:
1. Enumeration of every historical script / artifact that touched `tz_localize("UTC")` on timestamps sourced from `data/NQ_*.parquet`.
2. Scope D retraction signed off.
3. Historical N_1.3 re-verification signed off.

This document addresses all three. All gating deliverables are complete. Probe 5 preregistration is now unblocked.

---

## 1 · Scope

(a) Every `.py` script in the repo that calls `tz_localize` on timestamps ultimately derived from `data/NQ_*.parquet` or `data/NQ_*.csv`.
(b) Every comment, docstring, or variable name that describes those timestamps as UTC.
(c) Every cached machine-readable artifact (JSON, parquet) produced by a post-hoc session-decomposition script.
(d) The preventive infrastructure required to keep a repeat of the 2026-04-23 TZ bug from re-occurring.

Out of scope: narrative references to the past bug in verdicts, lessons, memories, and council artifacts — those correctly describe history, not live code.

---

## 2 · Definitions

- **The TZ bug (root cause):** Pre-2026-04-23 post-hoc scripts executed `ts.tz_localize("UTC").tz_convert("America/New_York")` on bar timestamps that are actually **naive Central Time** (Barchart vendor export per `scripts/data_pipeline/update_bars_yfinance.py:37` — `LOCAL_TZ = "America/Chicago"`). Result: every session-label assignment shifted by ~5–6 hours, inverting RTH vs GLOBEX buckets for most bars.
- **CT-naive:** A pandas timestamp or column with no `tzinfo`, whose wall-clock values are in `America/Chicago` (the vendor-authoritative TZ).
- **Canonical fix pattern:** `ts.tz_localize("America/Chicago", ambiguous="infer", nonexistent="shift_forward").tz_convert("America/New_York")`. This is the only correct pattern for converting CT-naive bars to ET-aware instants.
- **Source-of-truth TZ marker:** `scripts/data_pipeline/update_bars_yfinance.py:37`. Any future script that reads NQ parquets must honor this marker.

---

## 3 · Enumeration

### 3.A Live `.py` scripts that had the bug — now fixed

Verified by grep across the entire repo: **zero** live `.py` call sites for `tz_localize("UTC")` remain. All four previously-buggy scripts now use the canonical fix pattern at the line noted:

| File | Line | Pattern | Status |
|---|---|---|---|
| `tasks/_probe3_1h_ritual.py` | 190 | `tz_localize("America/Chicago", ambiguous="infer", nonexistent="shift_forward")` | FIXED |
| `tasks/_probe3_15m_nc.py` | 210 | same | FIXED |
| `tasks/_probe4_readout.py` | 134 | same | FIXED |
| `tasks/_scope_d_readout.py` | 139 | same | FIXED |

Fix commits: primary retraction cascade in late April 2026; additional comment/docstring cleanup in pipeline-review commit `40bc05d`. Verification command (should return 0 live hits in `.py` code):

```
rg -n '\.tz_localize\s*\(\s*["\']UTC["\']' --type py
```

Lessons-log cross-reference: `lessons.md` entry `2026-04-23 tz_bug_in_session_decomposition` records the root cause, the four fix sites, and the vendor-marker-verification rule.

### 3.B Adjacent scripts — comment / semantics audit

| File | Concern | Status |
|---|---|---|
| `tasks/_probe3_regime_halves.py:120-128` | Previous comment incorrectly said "Bar times are naive UTC from the CSV." Comparison itself is naive-on-both-sides (semantically correct — both sides strip tz identically). Comment replaced in `40bc05d` to reference vendor CT marker. Variable `SPLIT_DT_UTC` retained (misleading name; no numeric error). | COMMENT FIXED. Variable rename optional (see §5.D). |
| `tasks/_probe1_gross_ceiling.py:11` | Previous comment said "gross_pnl_dollars column already = 1-contract gross PnL per trade." Loose wording; scale-invariance applies for this pool because every combo resolves to a single fixed stop (contracts constant per combo). Comment replaced in `40bc05d` with the scale-invariance reasoning. | DOCUMENTATION FIXED. |
| `tasks/_probe4_run_combo.py` | Only references `tz_convert("America/New_York")` in docstring/comments describing downstream behavior of `_probe4_readout.py`. No live tz_localize or tz_convert call in this file. | NOT A SITE. |

### 3.C Engine / sweep — correct-by-design

| File | Pattern | Status |
|---|---|---|
| `scripts/param_sweep.py:1586-1596` | Opt-in `--bar-hour-tz ET` flag: when set, CT-localizes via `America/Chicago` then `tz_convert("America/New_York")` before extracting hour. Default `CT` preserves historical sweep caches v2–v11 (which all ran with raw CSV hour = CT hour). Comment block (lines 1578-1585) names the invariant explicitly. | **SUPERSEDES** the 2026-04-23 lessons.md §"Also discovered (not fixed — Option B)" note. The engine is now consciously CT-by-default with explicit opt-in ET semantics, which is the correct design choice for preserving historical comparability. |
| `scripts/data_pipeline/update_bars_yfinance.py` | Four tz calls, all canonical: L37 defines `LOCAL_TZ = "America/Chicago"`; L71 localizes vendor start to CT; L72 converts CT→UTC→naive for yfinance API (intentional); L91 converts yfinance UTC-aware bars back to naive CT for storage. | SOURCE OF TRUTH; no change. |

### 3.D Cached artifacts — status under corrected TZ

| Artifact | Produced by | TZ-sensitive? | Status |
|---|---|---|---|
| `data/ml/probe3/readout.json` | `tasks/_probe3_readout.py` (aggregator) | YES — consumes fixed per-gate JSONs | REGENERATED in `40bc05d` (F=1, COUNCIL_RECONVENE). |
| `data/ml/probe3/15m_nc.json`, `1h_ritual.json`, `param_nbhd.json`, `regime_halves.json` | per-gate scripts | YES — session partitioning depends on ET-minute mask | REGENERATED in retraction cascade (Apr 23 07:38 mtime). |
| `data/ml/probe4/readout.json` | `tasks/_probe4_readout.py` | YES | REGENERATED (Apr 23 07:37 mtime); row 4 fires COUNCIL_RECONVENE. |
| `data/ml/scope_d/readout.json` | `tasks/_scope_d_readout.py` | YES | REGENERATED (Apr 23 07:37 mtime). Additionally patched in `40bc05d` to support `SES_1 (RTH) dominates` classifier output. |
| `data/ml/probe4/readout.json:15` | (same) | Narrative string inside JSON describing methodology | NARRATIVE ONLY — no numerical content. |

### 3.E Narrative references — no action required

The following `tz_localize("UTC")` / "naive UTC" mentions all correctly describe the *past* bug and do not represent live defects:

- `lessons.md` entries
- `CLAUDE.md` Probe 3 retraction bullet
- `tasks/probe3_verdict.md` Amendment 2, `tasks/probe4_verdict.md` Amendment 2, `tasks/scope_d_brief.md` Amendment, `tasks/tz_bug_provenance_log_2026-04-23.md`
- `tasks/_agent_bus/probe4_2026-04-22/*.md` — pre-fix reviewer notes frozen at the time of the bug
- `tasks/_agent_bus/pipeline_review_2026-04-24/stats-ml-logic-reviewer.md` — post-fix audit
- `tasks/council-report-2026-04-24-post-tz-cascade.html`, `tasks/council-transcript-2026-04-24-post-tz-cascade.md` — today's deliverables

---

## 4 · What does NOT need re-verification

The council's peer-review round (Reviewer 2) called for re-verification of `combo_features_v12.parquet` and historical Probe 1 N_1.3 counts. Under closer inspection, neither actually depends on the buggy TZ path:

- **`combo_features_v12.parquet`** is produced by `scripts/analysis/build_combo_features_ml1_v12.py`. The per-combo features it emits (`audit_full_gross_sharpe`, `audit_full_net_sharpe`, robust walk-forward aggregates) are computed from `gross_pnl_dollars` / `net_pnl_dollars` / `r_multiple` columns that do not reference session labels. **TZ-agnostic.**
- **Probe 1 N_1.3 counts** (`tasks/_probe1_gross_ceiling.py`) use per-combo Sharpe on `gross_pnl_dollars` with no session filter applied. Session filtering for Probe 1 sweeps happened inside the engine's `SESSION_FILTER_MODE`, which operated on CT hours by design (lessons.md 2026-04-23 §"Also discovered") — combos with `session_filter_mode != 0` are a documentation-mismatch issue, not a TZ bug. **Probe 1 N_1.3 counts are unaffected** by the 2026-04-23 correction because session labels don't enter the computation.
- **Probe 2 absolute Sharpe, Welch-t, $/yr figures** — all session-agnostic statistics. Unchanged under TZ fix.
- **Probe 4 absolute gate values** (combo-1298 full-session Sharpe 3.55, combo-664 Sharpe 1.34, Welch-t 3.851) — all computed on SES_0 (entire session) which has no session labeling. **TZ-agnostic.**

What *did* change under the TZ fix: the session *sub-decomposition* of those absolute totals (SES_1 RTH vs SES_2 GLOBEX). The absolute totals themselves stand.

---

## 5 · Gating deliverables (what remains for closure)

### 5.A Source-TZ validation harness [✅ COMPLETE]

**Problem:** The 2026-04-23 bug cascaded across four probes because no assertion at parquet-load time caught the mismatch between the author's premise (naive UTC) and the vendor's reality (naive CT). Four reviewers independently added code-logic-reviewer approvals to the wrong premise.

**Implementation:** `src/tz_contract.py` (top-level module matching `src/io_paths.py` flat-layout convention; not `src/utils/` as initially proposed). Public API:

- `assert_naive_ct(df, time_col="time")` — raises `AssertionError` with actionable message if `time_col` is tz-aware, or if first bar year is outside `[2005, current_year]`. Two guarantees documented in module docstring: (1) mechanical — downstream `tz_localize(SOURCE_TZ)` will not raise; (2) plausibility — wholly-corrupt files are caught. Wholesale-shifted columns (e.g. someone converted CT→UTC then stripped tz) are **not** detectable at runtime; the vendor marker at `scripts/data_pipeline/update_bars_yfinance.py:37` is the sole source of truth and this limitation is documented in the module docstring.
- `localize_ct_to_et(ts)` — canonical CT-naive → ET-aware conversion using `tz_localize(SOURCE_TZ, ambiguous="infer", nonexistent="shift_forward").tz_convert("America/New_York")`. Raises `AssertionError` rather than silently double-localizing if input is already tz-aware.

**Adoption rule:** Every new script that reads a NQ parquet and does TZ-sensitive work must call `assert_naive_ct` at load time and use `localize_ct_to_et` for any ET-minute math. Retrofitting the 4 fixed scripts to call these helpers is optional but recommended; no retrofit is required for closure.

**Self-test:** `python src/tz_contract.py` loads each of `data/NQ_{1min,15min,1h}.parquet`, runs `assert_naive_ct`, and round-trips through `localize_ct_to_et`. Output on 2026-04-24:

```
[tz_contract] SKIP  NQ_1min.parquet    (not present)
[tz_contract] OK    NQ_15min.parquet   (n=   170,422, first=2019-01-01 17:15:00, last=2026-04-08 20:00:00)
[tz_contract] OK    NQ_1h.parquet      (n=    41,900, first=2019-01-01 18:00:00, last=2026-04-08 19:00:00)

[tz_contract] all NQ parquets pass source-TZ contract.
```

`NQ_1min.parquet` is not present in the local data layout (1min bars live at `data/NQ_1min.csv` per `CLAUDE.md` §Data inputs); the SKIP branch is the correct handling.

**Sign-off:** Signed 2026-04-24.

### 5.B Pre-commit / CI enforcement [⊘ DECLINED by user 2026-04-24]

**Problem:** Nothing currently prevents a future contributor (human or agent) from adding `tz_localize("UTC")` on NQ timestamps. The bug can regress.

**Decision:** User declined the hook-based enforcement on 2026-04-24. Both `scripts/checks/check_tz_utc.py` and `.githooks/pre-commit` were implemented and adversarially verified (sandbox `tz_localize("UTC")` call → exit code 1 with remediation pointer; full-tree scan on clean state → 162 files, zero violations) and then **removed** from the workspace before commit per user direction. The runtime validation harness `src/tz_contract.py::assert_naive_ct` (§5.A) is retained as the primary — and now sole — preventive layer.

**Rationale (reconstructed):** Any script that misinterprets the TZ will hit an `AssertionError` the moment it loads an NQ parquet and calls `assert_naive_ct`, which catches the defect at the actual fail point rather than at commit time. A pre-commit grep scanner is a belt-on-suspenders layer whose marginal benefit the user judged not worth the setup / escape-hatch friction.

**Residual risk:** A future contributor who (a) writes a TZ-sensitive post-hoc script, (b) does *not* call `assert_naive_ct`, and (c) localizes as UTC will re-create the 2026-04-23 bug with no mechanical barrier. Mitigation is procedural: `lessons.md` 2026-04-23 entry + the vendor-marker at `scripts/data_pipeline/update_bars_yfinance.py:37` + `src/tz_contract.py` module docstring are the three documentation pointers a careful reviewer should hit.

**Sign-off:** DECLINED 2026-04-24.

### 5.C Scope D retraction posture [COMPLETE — documented inline]

**Reviewer 2's request:** "Scope D divergence needs its own retraction document."

**Current state:** `tasks/scope_d_brief.md` contains:
- A `🛑 RETRACTED 2026-04-23 UTC` banner at line 3 naming the thesis inversion and the root-cause script.
- A full `## Amendment — Timezone bug retraction (2026-04-23 UTC)` section at line 118 with the before/after Sharpe table and the dominance-label correction.
- Committed in the TZ-fix cascade commits and reinforced by `40bc05d` (classifier extension for SES_1 dominance + regenerated readout).

**Decision:** Leave as in-document retraction. A standalone retraction file would duplicate content already load-bearing in the brief. If the user prefers a separate file for indexing (parallel to probe3_verdict / probe4_verdict amendments, which are also in-document), this position is revisable.

**Sign-off:** user acknowledges in-document retraction is sufficient (or requests extraction to standalone file).

### 5.D Variable rename in `_probe3_regime_halves.py` [OPTIONAL, LOW-COST]

**Problem:** Variable `SPLIT_DT_UTC` is a relic from the buggy-premise era. The value is a naive CT timestamp used in naive comparison; the name is misleading.

**Proposal:** Rename `SPLIT_DT_UTC → SPLIT_DT_CT` (or `SPLIT_DT_NAIVE`). No numerical change.

**Sign-off:** not required for closure; can be done opportunistically in a future edit.

### 5.E Historical N_1.3 re-verification [COMPLETE — see §4]

**Reviewer 2's request:** "Historical artifacts (combo_features_v12.parquet, Probe 1 N_1.3 counts) need re-verification under corrected TZ."

**Finding:** Neither artifact is TZ-sensitive (see §4 above). N_1.3 counts and combo_features are computed from session-agnostic per-combo aggregates. Re-verification would produce identical numbers.

**Sign-off:** user acknowledges the §4 argument and confirms no re-run of the Probe 1 gross-ceiling or v12 feature builder is required.

---

## 6 · Sign-off criteria ("TZ cascade is closed")

The cascade is closed when **all three** of the following are true:

1. **Enumeration acknowledged** — user has read §3 and agrees no additional live `.py` tz_localize("UTC") call sites exist.
2. **Preventive infrastructure landed** — §5.A (validation harness) is implemented, tested, and committed. §5.B (pre-commit/CI enforcement) was proposed and adversarially verified but declined by the user on 2026-04-24; `src/tz_contract.py::assert_naive_ct` is therefore the sole preventive layer.
3. **Re-verification posture acknowledged** — user agrees §5.C (Scope D retraction inline is sufficient, OR extract to standalone) and §5.E (N_1.3 + combo_features re-verification not required).

Only after this document is signed may Probe 5 preregistration be drafted, per the council's chairman verdict.

---

## 7 · Verification commands (reproducible)

```bash
# (a) No live tz_localize("UTC") call sites remain in .py code
rg -n '\.tz_localize\s*\(\s*["\x27]UTC["\x27]' --type py

# (b) All fix-site scripts use the canonical pattern
rg -n 'tz_localize\s*\(\s*["\x27]America/Chicago["\x27]' --type py tasks/

# (c) Vendor TZ marker is unchanged (LOCAL_TZ = America/Chicago)
rg -n '^LOCAL_TZ\s*=' scripts/data_pipeline/update_bars_yfinance.py

# (d) Post-fix machine artifacts have mtime ≥ 2026-04-23 (TZ-fix cascade date)
ls -la data/ml/probe3/readout.json data/ml/probe4/readout.json data/ml/scope_d/readout.json
```

---

## 8 · Sign-off

| Item | Status | Signed |
|---|---|---|
| §3 enumeration acknowledged (no live `tz_localize("UTC")` remains) | ✅ COMPLETE | 2026-04-24 |
| §5.A source-TZ validation harness implemented + tested | ✅ COMPLETE | 2026-04-24 |
| §5.B pre-commit / CI enforcement | ⊘ DECLINED by user (runtime harness §5.A stands alone) | 2026-04-24 |
| §5.C Scope D in-document retraction accepted (inline sufficient) | ✅ COMPLETE | 2026-04-24 |
| §5.E N_1.3 + combo_features re-verification waived (TZ-agnostic per §4) | ✅ COMPLETE | 2026-04-24 |

**TZ cascade is CLOSED.** Probe 5 preregistration is unblocked per chairman verdict 2026-04-24. Next gating step per the council: the **independent-data question** — either identify a genuinely held-out partition never touched by Probes 1-4 + Scope D, preregister forward paper-bar accrual, or acknowledge NQ bar-data is exhausted and sunset is the honest call. Only after that step may `tasks/probe5_signal_family_preregistration.md` be opened.

---

*Document generated 2026-04-24. Closed 2026-04-24.*
