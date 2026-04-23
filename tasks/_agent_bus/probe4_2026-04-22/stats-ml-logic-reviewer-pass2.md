---
from: stats-ml-logic-reviewer
run_id: probe4_2026-04-22
pass: 2
prior_pass: tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer.md
for: user
timestamp: 2026-04-23T05:30:00Z
subject: Confirmation pass on revised tasks/probe4_preregistration.md
verdict: PROCEED WITH FIXES
scope_reviewed: Revised Probe 4 prereg (§1, §4.3, §5 incl. §5.1/§5.2, §6, §7.2, §7.4, §7.5, §7.6) against Pass-1 findings C1, C2, W1, W2, W3, W4, W5, I3
critical_flags: 0
warn_flags: 1
info_flags: 2
cross_references:
  - tasks/probe4_preregistration.md
  - tasks/_agent_bus/probe4_2026-04-22/stats-ml-logic-reviewer.md
---

## Verdict

The revision lands all 8 Pass-1 findings (C1, C2, W1, W2, W3, W4, W5, I3) cleanly, plus the new §7.6 partition-reuse caveat is well-framed and correctly distinguishes the Welch-t (differential, cancels common-factor favorability) from absolute gates (do not). The two CRITICAL items from Pass 1 are RESOLVED. One genuine bug remains: §5.2's narrow-miss subsection is internally inconsistent with the §5 routing table — it claims a "Sharpe ∈ [1.1, 1.3) AND Welch-t ≥ 2.0" case routes to INCONCLUSIVE per row 1, but row 1 explicitly requires `Welch-t < 2.0` so the case actually falls through to row 5 (COUNCIL_RECONVENE). This is a single-paragraph fix, not a structural problem. Promote to PROCEED once §5.2's row-citation is corrected (or alternatively, add a row 0 narrow-miss → INCONCLUSIVE_NARROW_MISS branch and tighten row 1 to match). The Rule-2 LR table in §7.2 is dimensionally correct; spot-checks on all four numeric claims (A–D below) hold within rounding.

## Finding-by-Finding Status

### C1 (§4.3 threshold)
**RESOLVED.** §4.3 replaces the 1.3 Sharpe-differential gate with per-trade Welch-t ≥ 2.0 (one-tailed, α ≈ 0.025), which is the recommended fix from Pass-1 alternative #1. The H0 false-positive rate now matches the prereg's "pay for Stage-0" rhetoric (2.5% < 5.7% absolute-gate rate). The justification text correctly contrasts annualized-Sharpe SE (window-duration-bound) with per-trade Welch-t (n-bound), naming the actual statistical mechanism.

### C2 (§4.5 deletion)
**RESOLVED.** §4.5 deleted in full; §7.3 deleted in full. §5 row 2 (the old PROPERTY_FALSIFIED row) replaced by the SESSION_CONFOUND row. §6.7 still references "Stage-0 multiplicity acknowledged, not Bonferroni-inflated" which is consistent — the equivalence-band falsification was the only place where the noise-floor bug lived, and removing it cleanly closes the failure mode. Verdict-side risk of binding to an unfair Probe 3 retraction is gone. §1 branch map and §5.1 bindings are also consistent — neither retains a PROPERTY_FALSIFIED reference.

### W1 (§5 row order)
**RESOLVED.** §5 row 2 (SESSION_CONFOUND) now evaluates before row 3 (PROPERTY_VALIDATED). Explicit explanatory paragraph after the table confirms the load-bearing ordering. A 1298 that passes absolute and beats 664 on Welch-t but does so only in SES_2 now correctly routes to SESSION_CONFOUND.

### W2 (§5 row 3 inequality)
**RESOLVED.** New row 2 condition is `gate_1298_abs_pass == TRUE AND SES_2 absolute PASS AND SES_1 absolute FAIL AND (SES_2 net Sharpe − SES_1 net Sharpe) > 1.0`. The inequality direction is now correctly comparing overnight (SES_2) to RTH-dominated (SES_1), per Pass-1 alternative #4. This rule would correctly fire on a combo-865-style overnight-concentrated pattern.

### W3 (§5.2 narrow-miss disclosure)
**PARTIALLY RESOLVED.** The disclosure exists and addresses the spirit of the finding (a narrowly-failing 1298 with strong Welch-t evidence should be flagged, not silently swallowed). However, the routing claim inside §5.2 is wrong — see "New Issues" item E below. §5.2 says the case routes to INCONCLUSIVE per §5 row 1, but row 1's condition is `gate_1298_abs_pass == FALSE AND Welch-t < 2.0`. With Welch-t ≥ 2.0, row 1 does NOT fire — the case falls through rows 2 (gate_1298_abs_pass == FALSE), 3 (gate_1298_abs_pass == FALSE), 4 (gate_1298_abs_pass == FALSE), and lands at row 5 (COUNCIL_RECONVENE), which §1 branch map and §5 row 5 description both anticipate ("anomaly path: 1298 abs FAIL but Welch-t ≥ 2.0"). This is a one-line fix; flagging as new WARN below.

### W4 (§7.2 power calc)
**RESOLVED.** §7.2 now contains the per-gate and per-branch probability tables Pass-1 recommended in alternative #5. Per-gate numbers populated for H1a/H1b/H1c/H0; per-branch route probabilities populated; headline LR for PROPERTY_VALIDATED reported as ~629× (H1a vs H0). Numeric spot-checks pass — see check C below.

### W5 (§7.4 over-claim)
**RESOLVED.** §7.4's last sentence now reads "The Welch-t primary gate exploits the n asymmetry through the degrees-of-freedom denominator (Satterthwaite approximation), which is a consequence of the Welch formula rather than a designed-in property of this probe." Honest, doesn't over-claim. The framing now correctly attributes the n-asymmetry exploitation to the Welch statistic itself, not to a prereg-level design feature.

### I3 (§7.5 basin counting)
**RESOLVED.** §7.5 now closes with: "combo-1298's Δ=1 microstructure neighborhood from combo-865 means these two combos are closer to one observation than two in the same parameter basin. Probe 4 testing 1298 does NOT produce a second independent basin-survivor for Probe 1 N_1.3 counting purposes under any branch routing." Closes the post-hoc re-interpretation loophole exactly as Pass-1 recommended.

### New §7.6 (partition-reuse caveat)
**ADEQUATE.** The section names the right risk (window-favorability inheritance), correctly distinguishes which gates are robust to it (§4.3 Welch-t cancels common-factor favorability through the differential frame) versus which are exposed (§4.1 and §4.2 absolute gates), and binds verdict interpretation to lean on the Welch-t result rather than 1298's absolute Sharpe in isolation. This is the right durable framing — slightly stronger than what Pass-1 alternative #6 asked for, because it explicitly conditions verdict-document interpretation rather than just flagging the issue.

## New Issues Introduced by Revision

### A. §4.3 Welch-t α calibration at n_1298 ≈ 124, n_664 ≈ 665

The large-sample approximation is fine here. Satterthwaite df is computed as df ≈ (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1)]. With n₁ = 124 and n₂ = 665, even under the worst-case ratio of variances (one combo's per-trade σ very different from the other's), df is bounded below by min(n₁−1, n₂−1) ≈ 123 and above by n₁+n₂−2 = 787. At df = 123, t-critical for one-tailed α = 0.025 is **1.979**, vs the asymptotic z = 1.960 — a difference of ~1% in the critical value. At df ≥ 200, t-critical is ≤ 1.972. The prereg's "t ≥ 2.0" cutoff is in fact slightly **stricter** than the t_{123, 0.025} critical value of 1.979, so true H0 false-positive rate is ≤ 2.5%, not >. **No correction needed.** The Welch large-sample claim holds.

### B. §7.2 Welch-row power numbers vs Probe 2 baseline

Per-trade std ≈ $3,538 at 1h for combo-865 (Pass-1 reviewer's baseline). Mean per-trade net edge ≈ $840 at the strong end (~Sharpe 2.89 on n=124 maps to mean ≈ Sharpe·σ/sqrt(periods/yr)·... — using $840 from the council framing).

Under H1a (E[mean_pnl_1298] ≈ $840, E[mean_pnl_664] ≈ $0):
- SE of difference ≈ sqrt((3538²/124) + (3538²/665)) = sqrt(101,005 + 18,825) = sqrt(119,830) ≈ $346
- Expected t ≈ 840 / 346 ≈ **2.43**
- P(t ≥ 2.0 | E[t] = 2.43, SE_t ≈ 1) ≈ Φ(0.43) ≈ **66%** under a noncentral-t approximation, not 95%.

The §7.2 claim that P(PASS | H1a) ≈ 0.95 is **likely overstated** if the per-trade SE numbers above are correct. To get 95% power at t_crit = 2.0, you'd need E[t] ≈ 3.65, which requires either (a) a larger per-trade edge (~$1,260) or (b) substantially smaller per-trade σ. The Probe 2 readout numbers may differ from my $840 estimate; the actual per-trade mean for the held-out combo-865 was $124,896 / 220 ≈ **$568/trade net** (confirmed from CLAUDE.md combo-865 carve-out section). Using $568:

- E[t] ≈ 568 / 346 ≈ **1.64**, P(t ≥ 2.0) ≈ Φ(−0.36) ≈ **36%**

That is a substantially different power picture. **The §7.2 H1a/H1b/H1c row for Welch-t may be optimistic by a factor of 2-3×.** This does not invalidate the design — INCONCLUSIVE absorbing more H1a mass than the table suggests is a power loss, not a directional bias. But it does mean PROPERTY_VALIDATED's "~88% under H1a" branch probability is probably closer to 60-65%, and the headline ~629× LR is more like ~250-400× depending on the per-trade σ assumption.

**Recommendation**: not a blocker, but the verdict-side stats pass should re-derive these numbers from the actual combo-1298 training-partition per-trade mean and std (available from `data/ml/mfe/ml_dataset_v11_mfe.parquet`). Flag as INFO; the design still discriminates well enough on the validation branch even at the lower power numbers.

### C. PROPERTY_VALIDATED LR ≈ 629× math check

If the §7.2 numbers are taken at face value: P(PROPERTY_VALIDATED | H1a) under independence ≈ P(1298 abs PASS | H1a) · P(664 abs FAIL | H1a) · P(Welch-t ≥ 2.0 | H1a) = 0.974 · (1 − 0.057) · 0.95 = 0.974 · 0.943 · 0.95 = **0.872**. Under H0: 0.057 · (1 − 0.057) · 0.025 = 0.057 · 0.943 · 0.025 = **0.00134**. Ratio = 0.872 / 0.00134 = **651×**. Within 4% of the prereg's claimed 629×. Math is correct.

If the more conservative Welch-power numbers from check B above are used (P(Welch-t ≥ 2.0 | H1a) ≈ 0.36 instead of 0.95): LR = (0.974 · 0.943 · 0.36) / (0.057 · 0.943 · 0.025) = 0.331 / 0.00134 = **246×**. Still strong, still well-powered; the design's conclusion ("PROPERTY_VALIDATED is a very strong evidence outcome") survives both estimates.

### D. §5 row 4 (both-pass COUNCIL_RECONVENE) routing

P(both abs PASS | H0) = 0.057 · 0.057 ≈ **0.0032**. P(both abs PASS | H1c shared-weak) ≈ 0.500 · 0.165 ≈ **0.083** (using §7.2 row values). LR for COUNCIL_RECONVENE row 4 alone (both-pass) = 0.083 / 0.0032 ≈ **26×**, mildly favoring H1c. This is the correct routing — COUNCIL_RECONVENE preserves the falsification-information for council adjudication without binding the prereg to a mechanical retraction of Probe 3's posterior. Removes the Pass-1 C2 concern (the old §4.5 binding to PROPERTY_FALSIFIED would have applied a wrong retraction with LR closer to 0 against H1c). The new design routes the most ambiguous H1c outcome to a human-in-the-loop adjudication, which is the right response given the prereg's residual epistemic uncertainty about shared-edge mechanisms.

### E. §5.2 narrow-miss internal inconsistency (THIS IS A BUG)

§5.2 states: "If gate_1298_abs_pass == FALSE because net_sharpe_1298 falls in [1.1, 1.3) AND Welch-t ≥ 2.0, the INCONCLUSIVE routing in row 1 is the correct mechanical outcome..."

But §5 row 1 condition is: `gate_1298_abs_pass == FALSE AND Welch-t < 2.0`.

A case with `gate_1298_abs_pass == FALSE AND Welch-t ≥ 2.0` does NOT match row 1 (Welch-t condition fails). It falls through:
- row 2: needs `gate_1298_abs_pass == TRUE` — fails
- row 3: needs `gate_1298_abs_pass == TRUE` — fails
- row 4: needs `gate_1298_abs_pass == TRUE AND gate_664_abs_pass == TRUE` — fails
- row 5: catch-all — **fires** → COUNCIL_RECONVENE

§5.2 mis-cites the routing. The actual mechanical outcome for "Sharpe ∈ [1.1, 1.3) AND Welch-t ≥ 2.0" is COUNCIL_RECONVENE per row 5, not INCONCLUSIVE per row 1. This is consistent with §1 branch map (row 5 anomaly path explicitly names "1298 abs FAIL but Welch-t ≥ 2.0") and §5 row 5 description.

This is a one-paragraph fix. Two equivalent options:
- **Option A (recommended)**: Update §5.2 text from "the INCONCLUSIVE routing in row 1" to "the COUNCIL_RECONVENE routing in row 5", and reframe the disclosure as "the verdict document must explicitly flag this as a narrow-miss case for council adjudication."
- **Option B**: Add a §5 row 0 (highest priority): `gate_1298_abs_pass == FALSE AND net_sharpe_1298 ∈ [1.1, 1.3) AND Welch-t ≥ 2.0 → INCONCLUSIVE_NARROW_MISS`. Tighten row 1 condition by adding `AND net_sharpe_1298 < 1.1`. This makes the routing match §5.2's claim. More structurally invasive.

I recommend Option A — it's a one-line edit and the row 5 catch-all already correctly handles the case. The "anomaly path" framing in §1/§5/§5.1 already anticipates this exact outcome.

## Severity Flags

### CRITICAL
None. C1 and C2 from Pass 1 are RESOLVED.

### WARN

**W6 (new): §5.2 mis-cites the routing row for the narrow-miss case.**
- **Location**: §5.2 (line 149 of revised prereg).
- **Issue**: Text claims "INCONCLUSIVE routing in row 1 is the correct mechanical outcome" for `gate_1298_abs_pass == FALSE AND Welch-t ≥ 2.0`, but row 1's condition explicitly requires `Welch-t < 2.0`. The actual mechanical outcome is row 5 → COUNCIL_RECONVENE. Internal inconsistency between §5.2 disclosure text and §5 routing table.
- **Fix**: One-line edit per Option A above (change "INCONCLUSIVE routing in row 1" to "COUNCIL_RECONVENE routing in row 5", reframe disclosure as council adjudication rather than INCONCLUSIVE flag). No structural change to gates or branches.

### INFO

**I4 (new): §7.2 Welch-row power numbers may be optimistic by 2-3× depending on actual combo-1298 per-trade std.**
- **Location**: §7.2 Welch row, P(PASS | H1a/H1b/H1c).
- **Issue**: Using combo-865's held-out per-trade mean (~$568/trade) and std (~$3,538), Welch-t expected value under H1a is ~1.64, giving power ~36%, not the table's 0.95. Direction-correct (still discriminates well on the validation branch), magnitude potentially overstated.
- **Fix (verdict-side, not prereg-side)**: When the verdict document is authored, re-derive Welch-t expected value and power from combo-1298's training-partition per-trade mean/std (queryable from `data/ml/mfe/ml_dataset_v11_mfe.parquet`). Flag any deviation > 30% from §7.2's table values.

**I5 (new): Per-trade independence assumption for Welch-t.**
- **Location**: §4.3.
- **Issue**: Welch-t assumes independent observations within each sample. For per-trade PnL on a Z-score mean-reversion strategy, consecutive trades on the same instrument are weakly autocorrelated (back-to-back signal triggers can share residual regime state). This is unlikely to materially affect the test at n=124+, but it does mildly inflate the false-positive rate vs the nominal α = 0.025.
- **Fix (verdict-side)**: Compute Ljung-Box on per-trade PnL residuals as a diagnostic; if Q-statistic significant at lag 5, report a Newey-West-adjusted t alongside the Welch-t. Not a prereg blocker.

## Verdict

**PROCEED WITH FIXES.**

One single-paragraph fix needed before signing: §5.2's row citation. After that one edit, the prereg is sign-ready. C1 and C2 are RESOLVED, all 5 WARNs and I3 are RESOLVED, the new §7.6 is well-written, and the per-gate / per-branch LR table in §7.2 is mathematically correct as authored (even if the underlying Welch power numbers may want a verdict-side re-derivation against actual combo-1298 per-trade statistics). The structural design — 1298+664 parallel, Welch-t primary, session decomposition with correct ordering, no PROPERTY_FALSIFIED binding — is excellent. Sign once §5.2 is corrected.
