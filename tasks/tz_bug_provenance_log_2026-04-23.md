# TZ-Bug Provenance Log — 2026-04-23 UTC

**Purpose**: Per Reviewer 1 of the Probe 3 COUNCIL_RECONVENE
(`tasks/council-transcript-2026-04-23-probe3-reconvene.md`), this document
records exactly when the Central-Time-vs-UTC timezone bug was noticed, by whom,
under what trigger, and the exact commits that touched affected files. Purpose:
establish whether the "correction" is admissible as a bug fix or should be
treated as an additional test-partition read.

The Reviewer 1 concern: *"If the TZ audit was triggered because someone noticed
combo-865 looked weak post-signing and went looking for explanations, that's a
sixth read disguised as a bug fix — and every gate re-evaluation inherits that
selection."*

This log exists to answer that concern with primary evidence.

---

## Timeline (all times 2026-04-23 UTC unless noted)

### Pre-bug-discovery context (reading: not triggered by weakness)

- **2026-04-23 ~01:45 UTC** — Probe 4 verdict signed at commit `c419391`
  (SESSION_CONFOUND per §5 row 2). Downstream binding: "B1 session-structure
  sweep authorized."
- **2026-04-23 ~01:50 UTC** — Probe 4 per-trade parquets addendum committed
  (`ec60bf0`).
- **2026-04-23 ~02:18 UTC** — Session wrap-up; session closed with both
  Probe 4 and Scope D (not yet started) as forward-looking items. Combo-865
  Probe 3 PAPER_TRADE verdict (commit `b68fe62` on 2026-04-22) was **not
  under review**; it had been signed and was not being re-examined at the
  time the bug was found.

### Session reopens (2026-04-23 ~02:20 UTC onward — new chat session)

- **~02:20** — User opened a new session. No discussion of weak Probe 3
  results; session began with "What is next?" — a forward-looking question
  about B1 scope, NOT a retrospective question about combo-865 or Probe 3.
- **~02:30** — LLM Council fired for B1 scope. Council's
  **chairman recommended Scope D** as a mechanism check on combos {865,
  1298, 664}. The motivation was the Amendment 1 SES_2 bundling issue
  (16:00–18:00 post-RTH + halt window in the same bucket as pure GLOBEX
  overnight), explicitly flagged in the original Probe 4 verdict as a
  disclosure (§Amendment 1 practical impact). At this point the TZ
  interpretation was assumed correct.
- **~02:35** — Scope D brief authored at `tasks/scope_d_brief.md`,
  `_scope_d_readout.py` written. Scope D executed (pure pandas, <1s);
  output `data/ml/scope_d/readout.json` showed "SES_2a (pure overnight)
  dominates" unanimously across all three combos. Result was broadly
  *consistent* with Probes 3/4 narrative, not anomalous — the `regime_label`
  field in the readout returned unanimous "SES_2a dominates."
- **~02:45** — Scope D committed at `0ad0153`, pushed to origin.

### Bug discovered (~02:50 UTC) — NOT triggered by Probe 3 / combo-865 weakness

- **~02:50** — User invoked the **stats-ml-logic-reviewer** subagent with a
  paper-trade-readiness framing: *"Are we actually ready for paper-trading
  on combos {865, 1298, 664}?"* This was a forward-looking readiness review
  on the Scope D result, not a retrospective audit of Probe 3. The user's
  triggering phrasing was: *"I want you to launch the stat/ml reviewer.
  Are we actually ready for paper-trading?"* — the review was a
  pre-paper-trade gate, not a hunt for explanations of anomalous results.
- **~02:55** — stats-ml-logic-reviewer independently ran empirical checks
  on `data/NQ_1h.parquet` and found three convergent signatures that the
  data is naive Central Time, not UTC:
  1. Hour 16 is structurally absent in the raw bar distribution (CME halt
     is 16:00–17:00 CT = 17:00–18:00 ET).
  2. `session_break == True` fires ONLY at raw hour 17 (= 17:00 CT =
     18:00 ET = GLOBEX re-open).
  3. Volume peaks at raw hours 8–14 (= 08:00–14:00 CT = 09:00–15:00 ET
     RTH).
  Reviewer also noticed the authoritative vendor marker at
  `scripts/data_pipeline/update_bars_yfinance.py:37`: `LOCAL_TZ =
  "America/Chicago"  # CSV timestamps are CT (Barchart export)`.
- **~02:55** — Reviewer returned NOT_READY verdict with the TZ bug as
  CRITICAL-1 finding. Review artifact:
  `tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer-paper-trade-readiness.md`.

### Independent verification (not acceptance of reviewer's claim)

- **~03:00** — Main-agent independently re-ran the three empirical checks
  against `data/NQ_1h.parquet` (hour distribution, session_break pattern,
  volume peak hours). All three signatures reproduced. This verification
  was done *before* applying any fix, to confirm the reviewer's claim was
  not spurious.

### Fix applied (~03:05 UTC)

- **~03:05** — Four scripts patched: `tasks/_probe3_1h_ritual.py:186`,
  `tasks/_probe3_15m_nc.py:207`, `tasks/_probe4_readout.py:129`,
  `tasks/_scope_d_readout.py:133`. `tz_localize("UTC")` →
  `tz_localize("America/Chicago", ambiguous="infer", nonexistent="shift_forward")`.
  Variable renamed `ts_utc → ts_ct` for semantic clarity. Misleading
  comments replaced with explicit CT references.
- **~03:10** — Scripts re-run locally, producing corrected
  `data/ml/scope_d/readout.json` (inverted regime labels),
  `data/ml/probe3/1h_ritual.json` (8/16 → 12/16), `data/ml/probe3/15m_nc.json`
  (0/16 → 8/16), `data/ml/probe4/readout.json` (SESSION_CONFOUND → branch
  COUNCIL_RECONVENE per §5 row 4).
- **~03:15** — Engine-side audit completed: `scripts/param_sweep.py:1567`
  builds `bar_hour` from raw CSV hour (= CT). Engine `session_filter_mode
  1/2/3` operates on CT hours, though comments imply ET intent.
  Decision: **Option B (documented, don't change)** to preserve past-sweep
  cache validity. Added documentation comment to `src/strategy.py:82-103`.

### Retraction commits (~03:20 UTC)

- **~03:20** — All retractions committed at `a38d3c5`:
  - Retraction banners added to `tasks/probe3_verdict.md`,
    `tasks/probe4_verdict.md`, `tasks/scope_d_brief.md`.
  - Detailed Amendment sections appended to each verdict.
  - `lessons.md` entry `2026-04-23 tz_bug_in_session_decomposition`.
  - Memory updates: `feedback_tz_source_ct.md`, `project_tz_bug_cascade.md`,
    retraction banners on `project_probe3_combo865_pass.md` and
    `project_probe4_signed.md`.
  - `CLAUDE.md` combo-865 Probe 3 block annotated with retraction banner.
  - `src/strategy.py` CT-documentation comment.
- **~03:25** — Pushed `ec60bf0..a38d3c5` to origin/master.
- **~03:30** — Housekeeping commit `8569959` for regenerated Probe 3 EX
  parquets + new manifest JSON files from the script re-runs.

### Post-retraction council (Probe 3 COUNCIL_RECONVENE)

- **~03:40** — Probe 3 COUNCIL_RECONVENE fired per signed §5.2 routing
  (F=0 → F=1 triggered by §4.3 flip). 5/5 reviewers picked First
  Principles' reframe strongest; 5/5 picked Expansionist biggest blind
  spot. Chairman issued dissenting verdict: suspend A/B/C/D, audit Probe 1
  upstream first. Artifacts:
  `tasks/council-report-2026-04-23-probe3-reconvene.html` +
  `tasks/council-transcript-2026-04-23-probe3-reconvene.md`.

---

## Answer to Reviewer 1's admissibility test

**Was the TZ audit triggered by someone noticing combo-865 looked weak
post-signing?** No.

The trigger was a forward-looking paper-trade-readiness review on the
Scope D result, explicitly framed as "are we ready to move to paper
trading?" — not a retrospective audit on combo-865. At the moment the bug
was discovered, the Scope D verdict had just been committed (as `0ad0153`
at ~02:45) with a result that was *consistent* with the prior "overnight
concentration" narrative — not anomalous. If the trigger had been "865
looks weak, go find the explanation," the reviewer would not have
reported the bug as a CRITICAL-1 *finding about all three combos* — they
would have zeroed in on 865. They didn't; they named the bug as the root
cause of all three combos' session decomposition being mis-labeled, and
of the Scope D "regime_label" unanimity being an artifact.

**Was the discovery itself a selective read?** The reviewer's verification
was a set of three pre-specifiable empirical checks (hour distribution,
session_break pattern, volume peaks) against a vendor marker that existed
in the repo independently (`update_bars_yfinance.py:37`). The
falsification direction was symmetric: any of the three checks could have
disconfirmed the CT hypothesis, and they instead all converged. The
subsequent main-agent verification reproduced the three signatures
independently before any fix was applied. This is structurally different
from "searched for an explanation until one fit."

**Was the Probe 3 `§4.4 "8/16 exactly at threshold"` result a clue that
pushed someone to audit?** Possibly motivating-in-the-background — that
at-threshold result was disclosed in the Probe 3 verdict (`18a22ee` →
`b68fe62`) on 2026-04-22 as a concern the Phase E1 council was supposed
to weigh. But the bug discovery path did not start from 865's §4.4
readout and work backward; it started from a forward-looking
readiness review that happened to surface the TZ bug as part of a broader
readiness analysis (see stats-ml-reviewer artifact for the full decision
tree — the reviewer checked partition reuse, effective n, friction
sensitivity, and hold-time *before* getting to the TZ finding).

### Epistemic status of the TZ correction

- **Admissible as a bug fix** for the purposes of this provenance log,
  under the standard that bug-fixing a pipeline whose source-TZ
  assumption contradicts its vendor marker is warranted regardless of
  whether the fix confirms or disconfirms prior results.
- **NOT a pre-registered sixth test-partition read.** The correction
  changed the session labels on existing per-trade records; it did not
  run any new sweeps or add any new data queries against the test
  partition. The retraction of Probes 3/4 verdicts that the correction
  triggered is a *re-adjudication* under corrected labels, not a sixth
  confirmation read.
- **The specific concern of "6th read disguised as a bug fix" is
  defused** by the fact that the TZ fix was not selection-pressure on
  combo-865's weak readout; it was a forward-looking readiness
  review's byproduct. A different concern — "5 prior reads on the same
  partition" — stands independently and is what the Probe 3 chairman's
  verdict addressed by routing to "audit Probe 1 upstream first."

---

## Commit chain affected by the TZ correction

| Commit | Date | Scope | Status |
|---|---|---|---|
| `b68fe62` | 2026-04-22 | Probe 3 verdict amendment (audit W1-W4) | **RETRACTED** (Amendment 2) |
| `18a22ee` | 2026-04-22 | Probe 3 verdict F=0 PAPER_TRADE | **RETRACTED** via Amendment 2 |
| `c419391` | 2026-04-23 | Probe 4 verdict SESSION_CONFOUND | **RETRACTED** via Amendment 2 |
| `ec60bf0` | 2026-04-23 | Probe 4 per-trade parquets addendum | Stands (data-only) |
| `0ad0153` | 2026-04-23 | Scope D + B1 scope council | Scope D brief amended |
| `a38d3c5` | 2026-04-23 | TZ bug retraction commit | Superseding |
| `8569959` | 2026-04-23 | Probe 3 re-run artifacts (housekeeping) | Stands |

---

## Related references

- `tasks/probe3_verdict.md` Amendment 2
- `tasks/probe4_verdict.md` Amendment 2
- `tasks/scope_d_brief.md` Amendment
- `tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer-paper-trade-readiness.md` — original discovery
- `tasks/council-transcript-2026-04-23-probe3-reconvene.md` — Probe 3 COUNCIL_RECONVENE that demanded this log
- `memory/feedback_tz_source_ct.md` — durable rule
- `memory/project_tz_bug_cascade.md` — cross-probe cascade summary
- `lessons.md` `2026-04-23 tz_bug_in_session_decomposition`
- `scripts/data_pipeline/update_bars_yfinance.py:37` — authoritative vendor TZ marker
