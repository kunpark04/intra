# Council Transcript — B2 Scope & Sequencing

**Date**: 2026-04-22
**Project**: intra (NQ/MNQ 1-min futures mean-reversion backtest)
**Report**: `tasks/council-report-2026-04-22-b2-scope.html`
**Methodology**: Karpathy-style LLM Council (5 advisors + anonymized peer review + chairman synthesis).
**Framing enforced**: `feedback_council_methodology` Rule 1 (multiplicity naming) + Rule 2 (per-gate P(PASS|H1)/P(PASS|H0)).

---

## Original Question

User committed to Path B (broaden evidence base, no deployment of combo-865). Immediate decision: B2 sub-scope and B2 vs B1 sequencing. Want the council to stress-test main-thread recommendations for candidate selection, sequencing, multiplicity accounting, per-candidate probability estimates, and gate inheritance.

## Framed Question (distributed to all 5 advisors)

See report HTML for full framing. Key elements:
- Probe 1 falsified Z-score family at bar-timeframe axis (N_1.3 = 1/9/4 on 1min/15m/1h; gate ≥ 10).
- Probe 2: combo-865 @ 1h PASS (net Sharpe 2.89, 220 trades, +$124,896/yr on 2024-10-22 → 2026-04-08 holdout); @ 15m FAIL.
- Probe 3: F=0, posterior [0.65, 0.85], §4.4 at floor 8/16. Edge concentrates overnight (per-trade net 3.45× vs RTH).
- ML#2 V3/V4 both revoked 2026-04-21 after combo-agnostic audit exposed combo_id memorization leak.
- Combo-865 is the only combo with top-10-on-both-TFs cross-TF coherence in the 3000-combo Probe 1 sweep.

**Shortlist** (only 3 combos with any cross-TF coherence):
- Combo-1298: 1h gross Sharpe 2.272 (rank 1 of 3000), 498 trades; param-neighbor of 865 on entry_timing_offset + fill_slippage_ticks. Pre-gate PASS.
- Combo-1149: 1h 1.650 (rank 4), 101 trades; distant from 865 on cooldown (Δ=+7). Pre-gate PASS.
- Combo-664: 1h 1.200 (rank 6, FAILS pre-gate <1.3), 2,658 trades; identical microstructure to 865.

**5 questions**: (Q1) candidate selection, (Q2) B2 vs B1 sequencing, (Q3) Stage-0 multiplicity accounting, (Q4) 1298 rank-1 as H1 boost or H0 discount, (Q5) gate inheritance vs tightening.

---

## Advisor Responses

### The Contrarian

The shortlist is a trap. Stage-0 multiplicity is already a lie — screened the entire 3000-combo sweep for cross-TF coherence, found 3, now want robustness on survivors. Cross-TF coherence has no preregistered evidentiary weight; reverse-engineered from 865.

Combo-1298 is most dangerous, not safest. Rank 1 of 3000 is extremum. Microstructure Δ=1 neighborhood of 865 means same basin — basin-wide artifact explains both. P(PASS|H0) on 1298 HIGHER than random rank-500.

1149 is fig leaf (101 trades; regime-halves sub-gate will fail). 664 as negative control is only intellectually honest entry.

**Answer: (D) 664 alone, Probe 2 gate unchanged.** If 664 passes despite rank-6, gate broken and 865 was luck. Don't advance 1298 — can't out-audit rank-1 extremum. B2 before B1.

### The First Principles Thinker

Wrong question. Real question: what would B2 evidence actually prove?

H1 as framed is mush. Cross-TF coherence is Stage-1-selected; passing 1298 corroborates selection procedure, not 865. Two survivors from property-filtered triplet is autocorrelated evidence dressed up.

Correct question: what minimal experiment distinguishes "coherence property identifies genuine edge" from "property is itself overfit"? That experiment is 664 — the pre-gate failure. Under H0, 664 passes OOS as likely as 1298. Under H1, 664 underperforms. **664 is the ONLY candidate whose result carries information.**

Reframe: Run 1298 AND 664 together; 664 is TREATMENT not control. Drop 1149 (distant params = separate hypothesis, contaminates). Gate on RELATIVE performance (1298 minus 664 net Sharpe), not absolute thresholds.

Stop optimizing for PASS. Optimize for falsifiability.

### The Expansionist

Asymmetry nobody is pricing: combo-1298 is not a "second candidate" — it's a potential family-level unfalsification event. Probe 1 said N_1.3(1h) = 4. If 1298 passes held-out, we have two independent 1h survivors, Probe 1 Branch A sunset narrative fractures.

Q1: Run B (1298 + 1149 parallel). Include 664 as negative control.
Q2: B2 first.
Q3: Don't over-penalize Stage-0 — ~3-way on property with prior evidence. Bonferroni α/3, not α/3000.
Q4: 1298's rank-1 is H1 evidence — rank-1 AND coherent is joint signal.
Q5: Inherit Probe 2 gate exactly.

### The Outsider

Coming in cold: 1298 rank 1 of 3000 should scare, not excite. Watched two ML#2 stacks dissolve last week. Why would the literal extremum be the exception? 1298's rank-1 is H0 discount. 1149 at rank 4 / 101 trades is more honest.

Including 664 as negative control is the only option that would convince an outsider you're doing science. **Run D.**

Stage-0 isn't "pick 1 of 3" — it's 3000-sweep multiplicity with a filter. Filter itself needs justification.

Naive question: what stops overnight-GLOBEX concentration from being the ACTUAL signal, with combo-865 just being a parameter realization trading it? If yes, B1 should come first, not B2.

### The Executor

Monday morning: **Option B — 1298 + 1149 parallel.** Both pre-gate PASS, both fit in one remote job. Sequencing 1298 first buys nothing — 1h evaluation is cheap (Probe 2 harness exists). Drop 664 — doubles writeup burden; already have 865 @ 15m from Probe 2.

Q2: B2 before B1.
Q3: Bonferroni — 2 of 3 selection, inflate α by 2, net Sharpe ≥ 1.4.
Q4: Discount H1 for 1298 — rank-1 is construction extremum.
Q5: Reuse gate structure but raise Sharpe to 1.4.

**First step: write B2 preregistration, sign, push to sweep-runner-1.**

---

## Anonymization Mapping (for peer review)

- Response A = Expansionist
- Response B = Outsider
- Response C = Contrarian
- Response D = Executor
- Response E = First Principles

## Peer Reviews (5 reviewers, each saw all 5 anonymized responses)

### Reviewer 1

Strongest: Response E. Only response that interrogates inferential structure rather than optimizing within pre-given frame. Reframing 664 from "negative control" to "treatment," gating on relative (1298 − 664), makes experiment informative regardless of outcome. C reaches similar conclusion but stops at "run 664 alone," leaves no contrast.

Biggest blind spot: Response A. Treats 1298 rank-1 as H1-positive because "coherence is orthogonal to raw Sharpe." Wrong: coherence was reverse-engineered from 865; rank-1 in 3000-sweep is textbook extremum after V3/V4 leak disasters. A ignores project's own base rate.

All 5 missed: overnight-GLOBEX concentration as confound for every candidate. If 1298/1149/664 all trade GLOBEX like 865, B2 isn't testing combo-level generalization — it's re-testing a session artifact four times. Prereg must stratify by session.

### Reviewer 2

Strongest: Response E. Only one that reframes from "which candidate passes" to "what experiment is informative." Gating on relative Sharpe directly tests cross-TF-coherence property. D is operationally cleanest but epistemically lazy (reusing 865@15m as control ignores it's the same basin).

Biggest blind spot: Response A. Frames 1298 passing as "fracturing Probe 1 sunset narrative" — doesn't. Probe 1 was family-level N_1.3 ≥ 10; a second singleton doesn't get to 10. 1298 being Δ=1 param-neighbor of 865 means same basin. A counting basin-mates as independent survivors.

All 5 missed: overnight-GLOBEX concentration from Probe 3 isn't a nuisance variable — it's a confounder for B2. Preregistration should require per-session decomposition or result is uninterpretable.

### Reviewer 3

Strongest: E (First Principles). Only response correctly identifying inferential target. Making 664 the treatment (not control) and gating on relative performance is the cleanest experimental design on offer. Dissolves multiplicity mess Q3/Q4 by making rank irrelevant.

Biggest blind spot: A (Expansionist). Probe 1 §7.6 is terminal on bar-timeframe axis; one more 1h survivor is within-family noise, not re-opening. A downplays multiplicity (α/3) without engaging extremum argument. Ignores V3/V4 audit lesson.

All missed: combo-865's edge is overnight-GLOBEX (3.45× RTH). Whatever B2 runs must decompose 1298/664 by session. B1 (session probe) may be logically prior to B2, reversing recommended sequencing.

### Reviewer 4

Strongest: E. Only response reframing H1/H0 correctly. Relative-gate beats absolute-gate because it's the only design that can actually falsify the coherence property. A/B/D all "which PASS do I want"; C gets multiplicity but stops at nihilism.

Biggest blind spot: A. Treats 1298 passing as "family unfalsification" — but 1298 is Δ=1 microstructure neighbor of 865 in same basin. Two points in one basin is one observation, not two. A proposes Bonferroni α/3 when real multiplicity is 3000-with-reverse-engineered-filter.

All five missed: Probe 3 already showed 865's edge is ~3.45× overnight-GLOBEX concentrated. If all three edges collapse to same overnight window, B2 is re-measuring one phenomenon three times — B1 is strictly prior to B2, inverting sequencing.

### Reviewer 5

Strongest: E. Q1-Q5 collapse into one question: what experimental design can distinguish signal from selection? E's reframe — 664 as treatment, relative-Sharpe gate, optimize for falsifiability — is the only design where H0 and H1 make different predictions.

Biggest blind spot: A. Calls 1298 rank-1 + coherent a "joint signal" — backwards. Coherence filter was applied AFTER observing rank ordering on same sweep; rank-1 conditional on filter surviving is construction extremum, not independent evidence. A's Bonferroni α/3 ignores 3000-combo reverse-engineering.

All missed: Stage-0 is conditional on Probe 3's §4.4 exactly-at-floor 8/16 pass and overnight-GLOBEX concentration finding. None asked whether B2 should stratify by session (SES_2 overnight vs SES_0/1 RTH). If edge lives in GLOBEX, real H1 is session-structure, not combo-identity. B1's session probe may dominate B2 entirely.

---

## Chairman Synthesis

### Where the Council Agrees

Four of five advisors (Contrarian, First Principles, Outsider, Executor) — and every peer reviewer — treat combo-1298's rank-1-of-3000 Sharpe as an H0 discount, not an H1 boost. After two ML#2 stacks dissolved under combo-agnostic audit last week, "literal extremum of a sweep" is textbook failure mode. Only Expansionist read rank-1 as positive evidence, and all 5 peer reviewers flagged this as the single worst call in the council.

Four of five agree B2 before B1 in principle, though reviewers' blind-spot finding destabilizes this.

All 5 reviewers unanimously ranked First Principles as strongest, specifically for its reframe: 664 is not a negative control, it's a treatment.

Everyone — including Expansionist — concedes Stage-0 multiplicity is not 3-way Bonferroni.

### Where the Council Clashes

**Include 664 or drop it?** Contrarian, First Principles, Outsider say include (as treatment). Executor says drop. Expansionist includes as control, not treatment.

**Gate: absolute or relative?** First Principles: relative-Sharpe (1298 − 664) dissolves multiplicity. Executor: absolute with Bonferroni-inflated 1.4. Contrarian: keep 1.3 exactly.

**1149 in or out?** Expansionist and Executor keep. First Principles, Contrarian drop. On evidence, 1149 is dead weight (101 trades cannot survive halves sub-gate).

**B2 vs B1 sequencing — real clash surfaced by reviewers, not advisors.** Every advisor said B2 → B1. All 5 reviewers independently flagged that none of the advisors asked whether 1298/1149/664's 1h edge concentrates overnight the way 865's does. If yes, B1 logically prior to B2 — reversing entire sequencing plan.

### Blind Spots the Council Caught

- **Overnight-GLOBEX confound.** Unanimous among reviewers, invisible to all five advisors. Decision-changing: current B2 design cannot distinguish combo generalization from GLOBEX session inheritance.
- **1298-is-a-Δ=1-param-neighbor of 865 means "same basin, one observation."**
- **Rank-1 conditional on a reverse-engineered filter is a construction extremum.**
- **Probe 1 §7.6 terminality.** A second 1h singleton does not re-open family-level falsification.

### The Recommendation

Run 1298 + 664 in parallel on 1h held-out, with per-session decomposition baked into preregistration. Drop 1149. Gate on relative-Sharpe test (1298 − 664) as primary, absolute gates as secondary. Do not sequence B2 before B1 unconditionally — make B2's design produce the session-structure evidence that would otherwise require B1.

Concretely:

- **Q1 (Candidates)**: Modified (D). 1298 + 664, parallel. 664 as treatment per First Principles reframe. Drop 1149.
- **Q2 (Sequencing)**: B2 first, BUT B2 must include per-session stratification (SES_2 overnight vs SES_0/1 RTH, mirroring Probe 3 §4.4).
- **Q3 (Rule 1 multiplicity)**: Do not pretend this is 3-way Bonferroni. Pre-register Stage-0 filter as selection procedure under test, not free prior. Relative-Sharpe gate is how you pay for it.
- **Q4 (Rule 2)**: 1298's rank-1 is H0 discount. Primary gate: (1298 − 664) net Sharpe ≥ 1.3 on 1h held-out, both on ≥50 trades, decomposed by session.
- **Q5 (Gate inheritance)**: Inherit Probe 2 absolute gate (1.3 / 50 / $5k) as secondary for 1298 individually. Relative gate is primary.

One commitment binding in prereg: if 1298 and 664 both pass absolute gate within 0.3 Sharpe of each other, coherence property is falsified regardless of 1298's isolation performance. No post-hoc rescue.

### The One Thing to Do First

Write B2 preregistration now with per-session decomposition (SES_2 / SES_0 / SES_1) and a relative-Sharpe primary gate (1298 − 664 ≥ 1.3 on ≥50 trades each on 1h held-out). Sign it, commit it, push to sweep-runner-1 before running a single backtest.

The session decomposition is the non-negotiable addition every advisor missed and every reviewer caught — without it, B2 cannot distinguish combo generalization from GLOBEX session inheritance, and you will have spent another probe cycle re-measuring what Probe 3 already found.
