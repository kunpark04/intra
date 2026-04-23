---
from: code-logic-reviewer
run_id: probe4_2026-04-22
for: user
timestamp: 2026-04-22T00:00:00Z
subject: Pre-sign code-logic audit of revised tasks/probe4_preregistration.md
launch_recommendation: PROCEED WITH FIXES
critical_count: 1
warn_count: 4
info_count: 3
cross_references:
  - tasks/probe4_preregistration.md
  - tasks/probe3_preregistration.md
  - tasks/_probe3_readout.py
---

## Goal Understanding

Probe 4 is a "property test" — not a combo generalization test — that asks whether the cross-TF-coherence selection rule that produced combo-865 reliably identifies held-out edge (H1) or is a Stage-1 selection artifact (H0). Combo-1298 is the treatment (top-1 1h gross Sharpe in v11 sweep); combo-664 is the matched-microstructure but pre-gate-failing combo. The pre-reg locks three gates (1298 abs, 664 abs, Welch-t per-trade primary) plus a non-gate-bound session decomposition, and routes the four observable outcomes through five mechanically-applied branches. My audit scope: implementation correctness, edge cases, off-by-one risks, and internal consistency — NOT design (stats reviewer covers that).

## Scope Reviewed
- `tasks/probe4_preregistration.md` (lines 1–268, full read)
- `tasks/probe3_preregistration.md` (§4 / §5 / §8 conventions check)
- `tasks/_probe3_readout.py` (full read — branch-routing implementation reference)

## Findings

### CRITICAL

**C1. §5.2 narrow-miss disclosure mis-states which branch row fires (internal contradiction with §5).**

- **Location**: `tasks/probe4_preregistration.md` lines 147–149 (§5.2) vs lines 130–137 (§5 table).
- **Issue**: §5.2 states: *"If `gate_1298_abs_pass` == FALSE because `net_sharpe_1298` falls in [1.1, 1.3) AND Welch-t ≥ 2.0, the INCONCLUSIVE routing in row 1 is the correct mechanical outcome..."* But row 1 of the §5 table requires `gate_1298_abs_pass == FALSE AND Welch-t < 2.0` (both conditions, conjunction). With Welch-t ≥ 2.0, row 1 does **not** fire. Walking the table: row 1 NO (Welch-t condition fails), row 2 NO (requires 1298 PASS), row 3 NO (requires 1298 PASS), row 4 NO (requires 1298 PASS), row 5 YES (catch-all — and §5 row 5 explicitly enumerates *"`gate_1298_abs_pass` == FALSE with Welch-t ≥ 2.0 anomaly"* as a row-5 case). So the actual mechanical route is **COUNCIL_RECONVENE**, not INCONCLUSIVE.
- **Why it matters**: This is a load-bearing pre-registration document and the readout script will encode §5 mechanically. If §5.2 says "INCONCLUSIVE" but the table routes the same observation to "COUNCIL_RECONVENE", the post-data verdict writer will be forced to choose between contradictory clauses of the same signed document. That is exactly the post-hoc-reinterpretation hazard §6.1 forbids. It also creates ambiguity in Rule-2 LR accounting (§7.2 lists ~0.05 P(route|H0) for COUNCIL_RECONVENE row 5 vs ~0.94 for INCONCLUSIVE — the routing matters for posterior).
- **Suggested fix**: Either (a) rewrite §5.2 to reflect that the narrow-miss case routes to **COUNCIL_RECONVENE** (row 5) and the council framing must include the narrow-miss flag, OR (b) add a new explicit row to §5 between current rows 1 and 2 of the form *"`gate_1298_abs_pass` == FALSE AND `net_sharpe_1298` ∈ [1.1, 1.3) AND Welch-t ≥ 2.0 → INCONCLUSIVE (with narrow-miss flag)"*. Option (a) is the smaller edit and consistent with §5 row 5's own enumeration of this anomaly. Either way, §5.2's text and §5's table must align bit-exactly before signing.

### WARN

**W1. Ordering interaction between SESSION_CONFOUND (row 2) and COUNCIL_RECONVENE both-pass (row 4) is undocumented.**

- **Location**: `tasks/probe4_preregistration.md` lines 133–135.
- **Issue**: If both 1298 and 664 PASS absolute AND 1298 also exhibits the SES_2-only confound pattern (SES_2 PASS, SES_1 FAIL, Δ > 1.0), row 2 fires first per "first match wins" (line 138), routing to SESSION_CONFOUND and silently suppressing the both-pass adjudication §5.1 row 4 promises. That may be the intended behavior (session structure swamps both-pass interpretation), but it should be made explicit. Otherwise a future verdict reader will be confused why the both-pass council scope from §5.1 row 4 was bypassed.
- **Suggested fix**: Add a one-line clarification to §5 or §5.1 row 2 binding: *"Row 2 fires first even when 664 also passes absolute; in that case the SESSION_CONFOUND council framing must additionally note the both-pass observation as a secondary signal. The both-pass adjudication of row 4 is suppressed by ordering, not by negation."*

**W2. §4.3 "n_trades ≥ 50 for both" failure-mode JSON shape is not specified — risk of NaN/None ambiguity in readout.**

- **Location**: `tasks/probe4_preregistration.md` lines 106–107.
- **Issue**: The pre-reg says "If either has fewer, the primary gate defaults to FAIL" but does not specify what value `welch_t` takes in that path (NaN? null? sentinel string `"insufficient_n"`?). The §5 routing table evaluates `Welch-t < 2.0` and `Welch-t ≥ 2.0` as numeric comparisons. If `_probe4_readout.py` receives a JSON `null` or NaN, a naive `if welch_t >= 2.0:` will be False (NaN compares False) and a naive `if welch_t < 2.0:` is also False — both rows 1 and 3 then evaluate False, dumping the case to row 5 silently. That may be the desired routing, but it should be specified explicitly so the readout script author writes the correct comparison logic.
- **Suggested fix**: Add to §4.3: *"If either combo has n_trades < 50 on the test partition, `welch_t` is recorded as JSON `null`; the readout must treat null as 'gate FAIL and Welch-t comparisons in §5 evaluate False on both sides', routing the case to row 5 (COUNCIL_RECONVENE)."* This also reinforces the §6.1 mechanical-routing commitment.

**W3. Per-session decomposition execution model — one run × tagging vs three runs — is unspecified, with implications for trade-count consistency.**

- **Location**: `tasks/probe4_preregistration.md` §4.4 (lines 112–122) + §8.2 (lines 213–214) + §8.4 (line 225).
- **Issue**: §8.4 estimates "2 frozen combos × 3 session filters × 1 timeframe × 1 partition ≈ 6 backtest runs", which implies three independent runs per combo with `session_filter_mode` overridden per run (mirroring Probe 3 §3.2.4's grid execution). But the readout requirement says "every metric computed in §4.1/§4.2/§4.3 is additionally decomposed by session filter" (line 114) — a decomposition wording that more naturally implies one run with per-trade session tagging. The two execution models will produce **different trade sets**: a single-run model can have an SES_1-eligible signal opening at 09:30 ET and exiting in SES_2 (overnight), classified by entry session; a per-run model with `session_filter_mode` filtering entries will allow that same trade in the SES_1 run but not in the SES_0 = "all" run if the SES_0 mode is identical to none-filter and the engine deduplicates. Probe 3 §3.2.4 used the grid model, but its session/exit cells were independent runs. For Probe 4, if both models exist in the engine, the choice affects per-session Sharpe magnitudes and the §5 row 2 magnitude floor (Δ > 1.0) calibration.
- **Suggested fix**: §8.2 must specify: *"Per-session decomposition is executed as three independent backtest runs per combo (SES_0 = no-filter, SES_1 = RTH-only, SES_2 = overnight-only), mirroring Probe 3 §3.2.4's session-filter override semantics. Per-trade tagging is NOT used."* — OR the alternative if the user prefers tagging. Without this lock, the §7.2 P(route|H0/H1) numbers for SESSION_CONFOUND are not reproducible.

**W4. Per-trade PnL series storage — not stated as a deliverable in §8.3.**

- **Location**: `tasks/probe4_preregistration.md` §8.3 lines 218–222.
- **Issue**: §8.3 lists per-session `*.parquet` trade tables and per-combo `*_gate.json`, but does not call out that the per-trade `net_pnl_dollars` series for both 1298 and 664 must be persisted on the **SES_0 (all-sessions) partition** for the §4.3 Welch-t computation. The parquet trade tables presumably contain it, but the readout script needs an explicit pointer (which parquet, which column) to be implementable in the 10-line function the audit asked about. Right now `_probe4_readout.py` would need to know to read `combo1298_SES_0.parquet` and pull `net_pnl_dollars` — that's not specified.
- **Suggested fix**: §8.3 add a bullet: *"Welch-t in §4.3 is computed over `net_pnl_dollars` from `combo1298_SES_0.parquet` and `combo664_SES_0.parquet` (the unfiltered, all-sessions runs). Per-session Welch-t decompositions in §4.4 readout use the corresponding per-session parquets."* Also clarifies that Welch-t is a single all-sessions test (the natural reading), not three per-session tests.

### INFO

**I1. §7.2 LR table footer — "PROPERTY_VALIDATED LR ≈ 629×" is a derived figure; the gate-level entries are LR_H1a/H0, but the branch-level claim mixes branch-route probabilities.** That's mathematically fine (P(PROPERTY_VALIDATED|H1a) / P(...|H0) = 0.88/0.0014 ≈ 629), but worth a one-line label clarification so future readers don't conflate gate-LR and branch-LR. Cosmetic.

**I2. §1 "Branch map" (lines 31–34) lists branches in this order: PROPERTY_VALIDATED, SESSION_CONFOUND, INCONCLUSIVE, COUNCIL_RECONVENE. §5 table evaluates them in this order: INCONCLUSIVE (row 1), SESSION_CONFOUND (row 2), PROPERTY_VALIDATED (row 3), COUNCIL_RECONVENE (rows 4 & 5).** Different orderings are not wrong, but a one-line note in §1 — "*ordering in this section is exposition order; mechanical evaluation order is in §5*" — would prevent confusion.

**I3. §6.1 forbids "no 'the observed t was 1.95 which is close enough to 2.0' re-interpretation" — strong and matches §4.5 of Probe 3.** Good. No fix needed; recording as a positive observation.

## Checks Passed
- §4.3 Welch-t formula is implementable in ~10 lines (`scipy.stats.ttest_ind(..., equal_var=False)` or hand-rolled with `mean`, `var(ddof=1)`, `len`). Self-contained, no engine internals required.
- §5 branch routing is **exhaustive over the 4 observable variables** (gate_1298_abs_pass, gate_664_abs_pass, welch_t_ge_2, ses_confound_pattern). Walking the 5 reviewer-supplied scenarios: scenario 1 → row 4 (COUNCIL_RECONVENE) ✓; scenario 2 → row 2 (SESSION_CONFOUND) before row 3 ✓; scenario 3 → row 3 (PROPERTY_VALIDATED) ✓; scenario 4 → row 1 (INCONCLUSIVE) ✓; scenario 5 → row 5 (COUNCIL_RECONVENE) ✓. The §5 row 5 catch-all guarantees no observation falls through.
- Row 2 third clause (SES_2 net Sharpe − SES_1 net Sharpe > 1.0) is **operationalizable from the per-session JSON output** assuming W3 is resolved with three independent runs producing per-session `net_sharpe`. No new computation required beyond subtraction.
- Welch t-statistic remains valid for sign-equal-different-magnitude PnL distributions (reviewer's edge case 6b). Confirmed: Welch is mean-difference / SE-difference, not distribution-shape-sensitive at large n.
- §1 Branch map enumerates exactly 4 branches; §5 table produces exactly the same 4 branch labels (PROPERTY_VALIDATED, SESSION_CONFOUND, INCONCLUSIVE, COUNCIL_RECONVENE). PROPERTY_FALSIFIED is fully removed (grep-equivalent pass on all 268 lines for "FALSIFIED" — only legitimate appearances are §1 line 36 "Probe 1 §7.6 remains terminal" and §7.5 family-level reference + a Probe-1 historical mention, none asserting a Probe-4 branch).
- No dangling §4.5 references in §5 or §6 of Probe 4. (Probe 3 had a §4.5 "Tie-breaking"; Probe 4 deletes that section consistent with the revision spec — confirmed by absence of the string in §5/§6 content.)
- §6 irrevocable commitments (lines 154–161) cover: no mid-flight gate edits, no early-stop, mechanical routing, no methodology shift, scope lock, no 1298 deployment, Stage-0 multiplicity acknowledgment. All 7 commitments are well-formed and consistent with the gates above them.
- §9 review workflow is unchanged from Probe 3's pattern; parameter manifest gap (§2.1, §2.2) is bound to a post-authorization coding-agent step (line 235), reviewable on a second bus pass — acceptable deferral.
- Test partition boundary (2024-10-22 00:00 UTC) — §3 line 74 says "2024-10-22 00:00 UTC → 2026-04-08 00:00 UTC". Probe 3 §3.1 says test bars are 2,058,591 – 2,573,238 with first bar 2024-10-22 05:06. The 00:00 vs 05:06 discrepancy is cosmetic (Probe 4's bound is the date, Probe 3 surfaces the actual first-bar timestamp); not an off-by-one risk for Probe 4 because the partition is inherited byte-for-byte from Probes 2/3, not re-defined.
- Combo-1149 exclusion (§2.3) is clearly documented and out-of-scope-locked.
- §7.4 trade count discrepancy is correctly framed as "feature, not confound" with the Welch-Satterthwaite handling acknowledged. No over-claim.
- §7.5 basin-counting clause is present (line 199) and correctly observes that 1298 ∈ Δ=1 microstructure neighborhood from 865 → "closer to one observation than two" for Probe 1 N_1.3 counting. Good.
- §7.6 partition-reuse caveat correctly distinguishes differential gate (Welch-t cancels common-factor window favorability) from absolute gates (do not). Sound.

## Open Questions for User

1. **Engine session-filter semantics**: Does `scripts/param_sweep.py` support per-trade session tagging in a single run, or only entry-time session filtering via three separate runs? W3 hinges on this, and I did not read engine source per the scope cap. If the user can confirm "three-run grid model is what Probe 3 used and Probe 4 will use", W3 collapses to a 1-line documentation fix.
2. **Welch-t scope**: Is the §4.3 primary gate intended to be computed on the **all-sessions (SES_0) PnL series only**, or as the SES_0 result with per-session decompositions reported in §4.4? The current §4.3 text reads as a single all-sessions test (which is the natural reading and what W4 assumes), but the §4.4 phrase "every metric computed in §4.1/§4.2/§4.3 is additionally decomposed by session filter" could be read as mandating three Welch-t computations. Clarify.
3. **Heavy-tail / normality on per-trade PnL**: The reviewer's edge-case 6a asks whether the readout should emit a Shapiro-Wilk normality flag. My read: at n_1298 = 498, n_664 = 2,658 (training-partition values from §2.1/§2.2), Welch-t's Central-Limit robustness easily covers heavy tails, and a normality flag would be a §7-level diagnostic, not a gate input. I'd recommend NOT adding it pre-data — but flagging as a user decision since it's an addition to §8.3 if accepted.

## Launch Recommendation

**PROCEED WITH FIXES.**

C1 must be resolved before signing — it is an internal contradiction in a load-bearing pre-registration document, exactly the class of ambiguity §6.1 is designed to forbid post-data. The fix is a 2–3 sentence edit. W1–W4 are documentation tightenings that prevent post-data interpretive friction; I recommend resolving all four before signing but they are not contradictions, just specificity gaps. The branch-routing logic itself is sound and exhaustive over observable scenarios; the Welch-t formula is implementable; the partition is well-defined; the design is internally consistent once C1 is patched. Once the 1 CRITICAL is fixed (estimated 5-minute edit) and W1–W4 documented, the document is signing-ready.
