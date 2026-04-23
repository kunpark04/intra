---
from: stats-ml-logic-reviewer
run_id: probe4_2026-04-22
for: user
timestamp: 2026-04-23T03:40:00Z
subject: Pre-sign audit of tasks/probe4_preregistration.md
verdict: PROCEED WITH FIXES
scope_reviewed: Probe 4 unsigned draft preregistration (combo-1298 + combo-664 cross-TF-coherence property test)
critical_flags: 2
warn_flags: 5
info_flags: 3
cross_references:
  - tasks/probe4_preregistration.md
  - tasks/probe3_preregistration.md
  - tasks/probe3_multiplicity_memo.md
  - tasks/probe3_verdict.md
  - tasks/council-transcript-2026-04-22-b2-scope.md
  - tasks/probe1_verdict.md
---

## Verdict

The design correctly implements the council verdict's core reframe (664 as treatment, relative-Sharpe primary, per-session decomposition, drop 1149) and the document is operationally clean. However, a Jobson-Korkie power analysis on the 1.48-year test partition exposes **two CRITICAL threshold calibration issues** that the prereg itself flagged for this review: (1) the 1.3 relative gate is materially **looser** than the absolute gate against H0 because Δ's SE is sqrt(2) larger than an individual Sharpe SE — P(relative PASS | H0) ≈ 13.3%, compared to P(absolute PASS | H0) ≈ 5.7%. The "differential frame pays for Stage-0 selection cost" rhetoric is directionally correct but the chosen magnitude does the opposite of what it advertises. (2) The §4.5 equivalence band of 0.3 Sharpe at these sample sizes is inside the measurement noise floor (σ_diff ≈ 1.17), meaning a substantial mass of BOTH-PASS-under-H1 outcomes will falsely route to PROPERTY_FALSIFIED. Both findings are calibration errors, not design errors — fix the numbers and sign, do not rewrite the structure. The design remains information-rich and is the single strongest epistemic pivot the project has made since Probe 1.

## Methodological Audit

### §4.3 primary relative gate — threshold magnitude is miscalibrated

The gate is (net_sharpe_1298 − net_sharpe_664) ≥ 1.3 on a 1.48-year window.

Jobson-Korkie closed-form σ(SR_ann) ≈ sqrt((q/T)·(1 + SR²/(2q))). For any combo on this partition, q/T ≈ 1/1.48 = 0.676 — so σ(SR) ≈ **0.82** for SR near the 1.3 gate and **0.84** for SR near 2.9, largely independent of n (because q scales with n; the leading term is 1/years_span). Trade-count asymmetry between 1298 (~124 test trades) and 664 (~665 test trades) does **not** shrink either SE — §7.4's framing that "relative gate is robust to n asymmetry" is correct only in the weak sense that the SE is the same; it is not a feature, it is the arithmetic consequence of using an annualized Sharpe on a fixed-duration window.

σ_diff = sqrt(σ_1298² + σ_664²) ≈ **1.17 Sharpe units**.

Under H0 (E[Δ] = 0): P(Δ ≥ 1.3) = Φ(−1.3/1.17) = Φ(−1.111) ≈ **13.3%**.
Under H1 strong (E[Δ] = 2.89): P(Δ ≥ 1.3) = Φ(1.359) ≈ **91.3%**.
Under H1 moderate (E[Δ] = 1.5): P(Δ ≥ 1.3) = Φ(0.171) ≈ **56.8%**.
Under H1 weak (E[Δ] = 0.8): P(Δ ≥ 1.3) ≈ 33.5%.

Compare to the absolute gate for a single combo at net_sharpe ≥ 1.3:
- Under H0: Φ(−1.581) ≈ **5.7%**
- Under H1 strong: Φ((2.89−1.3)/0.82) = Φ(1.94) ≈ **97.4%**

**The relative gate is 2.3× more permissive under H0** than the absolute gate it claims to tighten. §7.2's justification ("mirrors Probe 2's absolute gate magnitude on a differential scale") confuses point-estimate magnitude with discriminating power. The two have different noise characteristics and cannot inherit each other's threshold.

To restore parity with the absolute gate's 5.7% H0-false-positive rate, the relative threshold should be **z·σ_diff = 1.58·1.17 ≈ 1.85 Sharpe**. Alternatively, to keep the 1.3 cutoff, gate on per-trade t-statistic instead of annualized-Sharpe-difference — per-trade t_stat gets power from n asymmetry and Welch's approximation handles the unequal variances cleanly.

### §4.5 falsification clause — equivalence band inside noise floor

|Δ| < 0.3 with σ_diff ≈ 1.17 is a ±0.26σ window.

Unconditionally:
- P(|Δ| < 0.3 | H0, E[Δ]=0) = 2·Φ(0.256) − 1 ≈ **20.2%**
- P(|Δ| < 0.3 | H1 strong, E[Δ]=2.89) ≈ 0.010 − 0.003 ≈ **1.0%**
- P(|Δ| < 0.3 | H1 moderate, E[Δ]=1.0) ≈ **14.2%**

Conditional on both absolute gates passing (the actual PROPERTY_FALSIFIED trigger):
- P(PROPERTY_FALSIFIED | H0) ≈ P(both abs pass | H0) · P(|Δ|<0.3 | both pass, H0) ≈ 0.003 · ~0.5 = **~0.0015**
- P(PROPERTY_FALSIFIED | H1 strong) ≈ 0.974 · 0.057 · ~0.01 ≈ **~0.0006**
- P(PROPERTY_FALSIFIED | H1 moderate where 664 has weak shared edge, E[Δ]=1.0): 664 passes abs with ~15% probability; 1298 passes with ~82%; conditional |Δ|<0.3 ≈ 0.20 → **~0.025**

This is the design's biggest calibration problem. The 0.3 band does **not** have meaningful H0/H1 separation at this sample size. The mode it's most sensitive to — "both combos share a modest ~1.0 Sharpe edge" — is exactly the mode where a property-based explanation is most plausible (both passed a legitimate pre-gate and share microstructure/session structure). The clause routes the most ambiguous H1 realizations to PROPERTY_FALSIFIED. §7.3's anchor to "Δ magnitude Probe 3 observed for EX_3" is category-confused — EX_3's Δ was a deterministic point estimate of a mechanical lock-in effect, not a sampling-distribution quantile of two-combo Sharpe differences.

Correct anchor: the equivalence band should come from a TOST procedure at a preregistered α, with margin m chosen so that |Δ| < m implies a meaningful practical indistinguishability. For σ_diff = 1.17 and α=0.05 TOST, a defensible margin is ~0.5 at 80% power against a reference alternative of 1.3, but this is tight and the prereg should acknowledge that equivalence is statistically hard at n=124.

### §4.3 computability rule — silent INCONCLUSIVE swallow

"If either has fewer than 50 trades, relative gate defaults to FAIL." Combined with §5 row 1 ("gate_1298_abs_pass = FALSE → INCONCLUSIVE"), this creates a specific failure mode: if combo-1298 has Sharpe 1.25 (fails abs by 0.05) but combo-664 has Sharpe −0.5 (massive isolation underperformance), the observed data gives **strong** evidence for H1 (Δ = 1.75), yet §5 row 1 fires first and routes to INCONCLUSIVE. This is the exact concern raised by the review charter.

The routing table evaluates §4.1 (1298 abs) before §4.3 (relative). Under H1-moderate, ~40% of the H1 mass for combo-1298 lands between 1.0 and 1.5 Sharpe; a non-trivial fraction of genuinely-informative H1 outcomes land in the INCONCLUSIVE bucket and lose their relative-gate signal.

### §5 row 3 operationalization — numeric thresholds not justified

"SES_2 PASS AND (SES_0 − SES_2) > 0.5" — the 0.5 is ungrounded. Is this 0.5 annualized Sharpe? Per-trade? What's its P(fire | H_session-only)? Probe 3 observed SES_0 = 2.895 vs SES_2 = 3.322 (Δ = −0.43, SES_2 > SES_0) so under the 865 precedent the §5 row 3 rule would NOT fire. But the *intent* of the rule is "SES_2 alone supplies substantially more edge than the aggregate" — for combo-865 that's true despite (SES_0 − SES_2) being negative, because SES_2 has higher Sharpe than the aggregate. The rule as written will route 865-like edges away from SESSION_CONFOUND. This is an encoding bug in the mechanical rule, not a design flaw in the concept.

### Branch-route ordering is load-bearing

Rows evaluate in order. The sequence §5 row 2 (both PASS + |Δ|<0.3 → PROPERTY_FALSIFIED) precedes §5 row 3 (session confound). If both combos pass absolute with small Δ AND the session decomposition shows SES_2-only concentration, PROPERTY_FALSIFIED fires and the SESSION_CONFOUND signal is suppressed. Given the Probe 3 overnight-GLOBEX finding, this is a non-theoretical risk: if 664 and 1298 share the same overnight concentration, their Sharpes may be close AND the session pattern matters.

## Data Audit

### Test-partition reuse and evidence independence

Probes 2, 3, 4 all use the same 1h chronological held-out window 2024-10-22 → 2026-04-08. The prereg correctly frames this as NOT a multiplicity violation for the property test (1298 and 664 have never been run on this partition by the project). However:

1. **Combo-865's Probe 3 result is informative about 1298 under H1.** 1298 is a Δ=1-microstructure-neighbor of 865; if 865 had an overnight-GLOBEX-concentrated real edge in this window, the strongest prior on 1298's held-out distribution is that it inherits that structure. This is correctly why the council chose 1298+664 (same-basin comparison cancels the shared factor). It does mean the H1 posterior ideal-update pipeline should condition on 865's Probe 3 result. The prereg does not attempt this, which is fine as long as the verdict document does.

2. **The partition itself has been "looked at" three times.** Each probe had the option to fail on this partition; none did. If the test window happened to be favorable to the Z-score mean-reversion family (something Probe 3 §2 from the multiplicity memo explicitly flagged as a residual concern), every probe run on it inherits that favorability. This is partially addressed by the relative-gate frame (1298 − 664 cancels window-favorability common factors), but it is NOT addressed for §4.1/§4.2 absolute secondary gates.

3. **No in-partition independence assumption is made explicitly.** 1298 and 664 share entry-timing windows on the same bars; their per-trade PnL series are correlated through shared market micro-structure. σ_diff = sqrt(σ₁² + σ₂²) assumes zero correlation; if ρ(PnL_1298, PnL_664) > 0 (likely, since both are Z-score mean-reversion entries on NQ), σ_diff is **overstated** — the relative test becomes slightly *more* sensitive than the naive formula predicts, at both the H0 and H1 tails. Minor effect but worth flagging in verdict-side analysis.

### Sample-size floor for combo-1298 deserves disclosure

Training (5.92 years) → 498 trades = 84/yr. On 1.48-year test → expected ~**124 trades**. The 50-trade absolute floor is 40% of expected; at n=124 the Sharpe SE is 0.82, not "narrow." One unlucky regime half of the test window could deliver ~60 trades — still above the 50 gate but with σ(SR) ≈ 1.2. This is not a blocker, but the prereg should name the expected trade count as part of the readiness check, not assume "both easily clear the 50-trade floor."

## Result Interpretation

### Rule 2 per-branch probabilities (quantitative — §7.2 deferred calc)

Using Jobson-Korkie SE at n ≈ 124 (1298) and n ≈ 665 (664), σ_SR ≈ 0.82 for each; σ_diff ≈ 1.17. Independence assumed (see caveat above). Three H1 variants named:

- **H1a (property strong, E[SR_1298]=2.89, E[SR_664]=0)** — 1298 inherits 865's edge, 664 has no edge
- **H1b (property moderate, E[SR_1298]=1.8, E[SR_664]=0)** — 1298 has real but smaller edge, 664 none
- **H1c (property weak/shared, E[SR_1298]=1.3, E[SR_664]=0.5)** — both have some edge, 1298 larger
- **H0 (property overfit, both SR=0 OOS)**

| Gate | P(PASS \| H1a) | P(PASS \| H1b) | P(PASS \| H1c) | P(PASS \| H0) | LR H1a/H0 |
|---|---:|---:|---:|---:|---:|
| §4.1 1298 abs (≥ 1.3) | 0.974 | 0.730 | 0.500 | 0.057 | 17.1× |
| §4.2 664 abs (≥ 1.3) | 0.057 | 0.057 | 0.165 | 0.057 | 1.0× |
| §4.3 relative (≥ 1.3) | 0.913 | 0.676 | 0.246 | 0.133 | 6.9× |
| §4.5 |Δ|<0.3 (cond. both pass) | ~0.01 | ~0.08 | ~0.45 | ~0.50 | ~0.02× |

Branch-route probabilities (from §5 table applied mechanically):

| Branch | P(route \| H1a) | P(route \| H1b) | P(route \| H1c) | P(route \| H0) |
|---|---:|---:|---:|---:|
| PROPERTY_VALIDATED | 0.918·(1−0.057)·0.913 ≈ **0.79** | 0.730·0.943·0.676 ≈ **0.47** | 0.500·0.835·0.246 ≈ **0.103** | ~0.003 |
| PROPERTY_FALSIFIED | ~0.0006 | ~0.005 | ~0.037 | ~0.0015 |
| SESSION_CONFOUND | ~0.10 (if SES_2-only shared) | lower | lower | ~0.01 |
| INCONCLUSIVE | ~0.03 | 0.27 | **0.50** | 0.94 |

Headline per-branch LRs (PROPERTY_VALIDATED):
- H1a/H0 ≈ **263×** — strong discriminating branch if H1 is property-strong
- H1b/H0 ≈ 157× — strong
- H1c/H0 ≈ 34× — meaningful but modest
- **H0 route consolidation: 94% of H0 mass → INCONCLUSIVE, 0.3% → PROPERTY_VALIDATED.** The design is well-powered AGAINST H0 false positives on the validation branch.

**The calibration concern is exclusively in the PROPERTY_FALSIFIED branch**, where H0 (1.5×10⁻³) and H1a (6×10⁻⁴) have similar route probability. Fixing §4.5's equivalence band solves this.

### What each branch actually licenses

- **PROPERTY_VALIDATED**: strong evidence the cross-TF-coherence property identifies edge. Posterior moves from prior 0.167 to ~0.98 against H0 under the central LR, moderated by shared-partition caveats. The prereg's §5.1 binding ("authorizes B1 as follow-on, not 1298 paper trade") is the correct pre-commitment.

- **PROPERTY_FALSIFIED** (as currently specified): under-specified; conflates "shared weak edge" (mild H1c) with "property is artifact" (H0). Posterior update math should NOT collapse the [0.65, 0.85] Probe 3 posterior uniformly — the falsification evidence is mode-dependent. §5.1 binding to "retract project_probe3_combo865_pass.md posterior range" is too strong a commitment for a weakly-discriminating branch.

- **SESSION_CONFOUND**: design-wise correct. Probability of triggering is small (needs SES_2 to carry edge and SES_0 aggregate to be meaningfully lower) — under combo-865's observed pattern the rule as written would NOT fire (SES_2 > SES_0). Verify the rule matches the intent.

- **INCONCLUSIVE**: absorbs ~50% of H1c mass and ~94% of H0 mass. An INCONCLUSIVE outcome is not neutral — it is mild-negative for H1 relative to prior. §5.1 binding ("no retroactive change to Probe 3 verdict") is correct.

## Impact

If the 1.3 relative threshold ships as-is:
- Under H0, the relative gate fires ~13% of the time — the "pay for Stage-0 multiplicity" claim in §6.7 is not quantitatively supported.
- The PROPERTY_VALIDATED route still has LR ≈ 100-250× under H1 (because it also requires 1298 to pass abs AND 664 to fail abs), so the headline conclusion will usually be correct, but the prereg's rhetorical case for why the relative gate matters is wrong.

If the 0.3 equivalence band ships as-is:
- PROPERTY_FALSIFIED will be a rare event in general (<1% under most states), but when it does fire under H1c (weak-shared-edge), the binding to retract 865's posterior is too aggressive given the gate's discrimination.
- This is the narrowest path to an unfair retraction of Probe 3 evidence.

Both are fixable at the numeric level without altering the structural design. Neither blocks the experiment's core logic.

## Recommended Alternatives

Ranked by information value:

### 1. Replace §4.3 threshold with a per-trade t-statistic gate (HIGH IV)

Rather than gating on (SR_1298_ann − SR_664_ann) ≥ 1.3, gate on:

```
t = (mean_pnl_1298 - mean_pnl_664) / sqrt(var_1298/n_1298 + var_664/n_664) ≥ 2.0
```

(Welch two-sample t, one-tailed). Benefits:
- Exploits the full n asymmetry (664 has ~5.4× the trade count of 1298).
- Has a clean null distribution under H0.
- Preregistered α = 0.025 one-tailed gives `t_crit ≈ 2.0` and corresponds to a well-understood 2.5% H0 false-positive rate.
- Power under H1a is essentially 1.0 given Probe 3's observed per-trade mean of ~$960 for 865-class.

**Decision rule**: if you prefer to keep a Sharpe-based metric for narrative consistency with Probe 2, raise the threshold to **1.85 Sharpe** (which is 1.58·σ_diff, matching the abs gate's 5.7% false-positive rate).

### 2. Either drop §4.5 or replace with a TOST-calibrated band (HIGH IV)

Option A (recommended): drop the unconditional falsification clause. Replace §5 row 2 with: "If both PASS absolute AND Δ < 0.3, route to COUNCIL_RECONVENE. Council decides whether to retract Probe 3 posterior in light of the full per-session readout."

Option B: keep the clause but widen the band to match a TOST margin at 1-β = 0.8 against reference alternative 1.3. Approximately m ≈ 0.5 at this sample size, preregistered as: "Δ < 0.5 AND both PASS absolute AND per-session readout shows shared session structure → PROPERTY_FALSIFIED."

Both variants remove the mode where H1c falsely routes to PROPERTY_FALSIFIED.

### 3. Re-order §5 table so SESSION_CONFOUND precedes PROPERTY_FALSIFIED (MEDIUM IV)

Current row 2 (PROPERTY_FALSIFIED on |Δ|<0.3) fires before row 3 (SESSION_CONFOUND). Swap order so a genuinely session-structured signal gets classified correctly even when Δ is small.

### 4. Fix the §5 row 3 operational rule (MEDIUM IV)

Replace "(SES_0 − SES_2) > 0.5" with "(SES_2 Sharpe − SES_1 Sharpe) > 1.0" (or a similar rule that actually captures "overnight concentrates the edge"). Grounded in Probe 3's observed overnight/RTH ratio, not in a guessed threshold. The current rule would not fire on combo-865's own pattern (SES_2 > SES_0 by 0.43).

### 5. Add a Rule 2 table to §7.2 (MEDIUM IV)

Populate §7.2 with the per-gate LR table above. This is the "Rule 2" commitment from `feedback_council_methodology.md`. The table below is reproducible and should live in the prereg, not in a reviewer's bus artifact, to be binding:

| Gate | P(PASS \| H1 central) | P(PASS \| H0) | LR |
|---|---:|---:|---:|
| §4.1 1298 abs ≥ 1.3 | 0.73 (at E[SR]=1.8) | 0.057 | ~13× |
| §4.2 664 abs ≥ 1.3 | 0.057 | 0.057 | 1× |
| §4.3 relative ≥ X (X TBD per rec #1) | populate | populate | target ≥ 10× |
| §4.5 equivalence clause | populate | populate | — |

### 6. Pre-commit partition-sharing caveat (LOW IV)

Add a §7.6 ("Partition reuse transparency") that states: "The 1h 2024-10-22 → 2026-04-08 partition has been consumed by Probes 2 and 3. Combo-1298 and combo-664 have never been run against it by the project. Shared-window favorability is partially cancelled by the relative-Sharpe frame. The verdict writeup must carry a residual-window-favorability flag regardless of outcome."

### 7. Add sample-size readiness note to §3 (LOW IV)

State the expected trade counts: combo-1298 ~124 on test; combo-664 ~665. Both clear 50. σ(SR) ≈ 0.82 at those counts, leading to 95% CI half-width ~1.6 Sharpe — wide enough that an observed SR of 2.0 has CI [0.4, 3.6]. Preregister the expected CI half-width so post-hoc readers don't mis-read a point estimate as precise.

## Severity Flags

### CRITICAL

**C1: §4.3 threshold of 1.3 is materially looser than advertised.**
- **Location**: §4.3, §7.2.
- **Issue**: P(relative PASS | H0) ≈ 13% vs absolute gate's 5.7% under the same "pay for Stage-0" framing. The justification "mirrors Probe 2's absolute gate on a differential scale" confuses point-magnitude with power because σ_diff = sqrt(2)·σ_SR.
- **Fix**: Either (a) raise to 1.85 Sharpe, or (b) replace with per-trade Welch t ≥ 2.0, or (c) explicitly acknowledge in §7.2 that 1.3 is NOT multiplicity-equivalent to the absolute gate and document the trade-off. Recommend (b).

**C2: §4.5 equivalence band of 0.3 Sharpe is inside the noise floor at n ≈ 124 / 665.**
- **Location**: §4.5, §5 row 2, §7.3.
- **Issue**: σ_diff ≈ 1.17 makes 0.3 a ±0.26σ window. Under H1c (shared weak edge, E[Δ]=1.0), ~14% of realizations fall inside the band; conditional on both passing absolute, ~45%. This routes genuinely-ambiguous H1 outcomes to PROPERTY_FALSIFIED with a binding that retracts Probe 3's posterior. The falsification branch is under-powered for what it commits to.
- **Fix**: Drop the unconditional falsification binding and route both-PASS-with-small-Δ to COUNCIL_RECONVENE. If the clause is kept, calibrate the margin via TOST and document.

### WARN

**W1: §5 routing-order can suppress SESSION_CONFOUND.**
- **Location**: §5 table.
- **Issue**: Row 2 (PROPERTY_FALSIFIED on |Δ|<0.3) evaluates before row 3 (SESSION_CONFOUND). If both-PASS-small-Δ co-occurs with SES_2-only concentration, SESSION_CONFOUND is suppressed.
- **Fix**: Re-order so SESSION_CONFOUND is row 2; PROPERTY_FALSIFIED becomes row 3.

**W2: §5 row 3 operational rule is mis-encoded.**
- **Location**: §5 row 3, last paragraph of §5 table.
- **Issue**: "(SES_0 − SES_2) > 0.5" is the wrong inequality direction for capturing "overnight-concentrated edge" — under combo-865's own precedent (SES_2 > SES_0 by 0.43) the rule would not fire. The rule should compare SES_2 to RTH-dominated aggregates (SES_1), not to SES_0.
- **Fix**: Replace with something like "(SES_2 PASS) AND (SES_2 Sharpe − SES_1 Sharpe > 1.0)" or "(SES_2 PASS) AND (SES_1 FAIL)".

**W3: §4.3 computability-default swallows informative small-margin-1298 failures.**
- **Location**: §4.3 + §5 row 1.
- **Issue**: If 1298 narrowly fails absolute (e.g., Sharpe 1.28) while 664 fails catastrophically (e.g., Sharpe −0.5), row 1 fires first and INCONCLUSIVE swallows strong H1 signal from the Δ.
- **Fix**: Either (a) make §5 row 1 condition on "gate_1298_abs_pass = FALSE AND |Δ| < 0.8" (otherwise route to COUNCIL_RECONVENE), or (b) accept the loss and document in §7 that small-margin abs-failures are routed to INCONCLUSIVE by design.

**W4: §7.2 and §7.3 numeric anchors are category-confused.**
- **Location**: §7.2, §7.3.
- **Issue**: §7.2 anchors the 1.3 threshold to Probe 2's absolute gate (different noise scale). §7.3 anchors the 0.3 band to EX_3's deterministic Δ (not a sampling-distribution quantity).
- **Fix**: Replace with a Rule-2 LR table and a TOST-derived margin respectively (see Recommended Alternatives #2 and #5).

**W5: §7.4 trade-count asymmetry framing overclaims robustness.**
- **Location**: §7.4.
- **Issue**: "Relative gate is designed to be robust to per-combo n asymmetry" is true only in the sense that σ_SR_ann is nearly independent of n on a fixed-duration window. The actual n asymmetry can be exploited via a per-trade Welch-t (recommendation #1). Current framing implies more robustness than the statistic provides.
- **Fix**: Soften the claim; add sentence acknowledging the trade-off.

### INFO

**I1: Verdict-writeup should do an ideal-Bayesian posterior update conditional on Probe 3 result.**
Because 1298 is a parameter neighbor of 865 and Probe 3 has already shifted the posterior on the basin, a proper H1 for 1298 should condition on that. Not a prereg issue — flag for the post-readout verdict authoring.

**I2: σ_diff assumes ρ(PnL_1298, PnL_664) = 0.**
Both are Z-score mean-reversion entries on NQ with overlapping signal timing; ρ is almost certainly > 0. True σ_diff is lower than the naive 1.17 estimate, making the relative gate slightly sharper than the power numbers above. Minor; does not change recommendations.

**I3: Probe 1 §7.6 terminality language in §7.5 is tight enough.**
"Two 1h survivors is still not N_1.3 ≥ 10" is the right frame. Add a sentence "and 1298's microstructure Δ=1 from 865 means they share a parameter basin — for the purpose of counting admissible OOS survivors at the family level, this is closer to one observation than two" to close an obvious post-hoc re-interpretation loophole (Reviewer 4 flagged the same point in the council).

## Verdict

**PROCEED WITH FIXES.**

C1 and C2 are numeric calibration errors, not design errors. The prereg's structure — 1298+664 parallel, relative primary, session decomposition, drop 1149, no paper-trade binding under any branch — is the single strongest experimental design this project has produced. The 1.3-and-0.3 pair of numeric thresholds should not be signed as-is; fix them (recommend Welch-t ≥ 2.0 for the primary gate, drop or widen the equivalence band), then sign. The four WARN items should be addressed in the same pass because they are cheap to fix and they tighten the routing table against mode confusion.

If the user chooses to ship as-drafted regardless, the verdict document must carry the following caveat: "Under the preregistered 1.3 relative threshold, P(relative PASS | H0) ≈ 13% per Jobson-Korkie, not the 1-sigma-plus event §7.2 describes. The PROPERTY_VALIDATED branch still discriminates at LR ~250× against H0 in combination with the absolute gates, but the relative gate alone does not pay the Stage-0 multiplicity cost at the magnitude the prereg rhetoric implies." That caveat is non-trivial and preferably avoided by fixing the threshold.
