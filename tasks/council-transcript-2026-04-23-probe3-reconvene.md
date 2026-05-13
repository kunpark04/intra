# LLM Council Transcript — Probe 3 COUNCIL_RECONVENE (2026-04-23 UTC)

**Question counciled**: Adjudicate the COUNCIL_RECONVENE triggered on Probe 3 (combo-865 @ 1h) under §5.2 of the signed preregistration, following the 2026-04-23 TZ-bug retraction that moved F-count from 0 to 1.

**Authority**: Probe 3 preregistration `tasks/probe3_preregistration.md` §5.2 ("Exactly 1 / 4 FAIL → council re-convene") — signed at commit `f8447af` + review fixes `8636167`. Verdict retraction: `tasks/probe3_verdict.md` Amendment 2 (commit `a38d3c5`).

**Anonymization mapping revealed:**
- Response A → The Outsider
- Response B → The Executor
- Response C → The Contrarian
- Response D → The Expansionist
- Response E → The First Principles Thinker

---

## Framed Question

Adjudicate the COUNCIL_RECONVENE triggered on Probe 3 (combo-865 @ 1h) under §5.2 of the signed preregistration. Original signed verdict: F=0 → PAPER_TRADE. Under a corrected timezone fix (data is Central Time per `scripts/data_pipeline/update_bars_yfinance.py:37`, not UTC), F becomes 1 because §4.3 15m negative control flips from 0/16 PASS to 8/16 FAIL (rescue fires — 15m edges can be recovered via session/exit filter combinations under correct session labels). §4.1 regime halves and §4.2 param ±5% neighborhood remain PASS (both TZ-agnostic). §4.4 1h session/exit ritual strengthens from 8/16 exactly-at-threshold to 12/16 comfortably clear.

Corrected session decomposition for combo-865 on the 1h test partition (key reversal from 5 weeks of project narrative):

| Bucket | Pre-fix Sharpe | Post-fix Sharpe |
|---|---:|---:|
| SES_0 all trades | 2.895 | 2.895 (unchanged — session-agnostic) |
| SES_1 RTH 09:30–16:00 ET | 0.639 | **2.635** |
| SES_2 ex-RTH | 3.322 | 1.467 |

Combo-1298 (from Probe 4, also TZ-affected) shows RTH Sharpe **4.414** on n=77 RTH trades, +$146,545/yr — the cleanest single readout in the project.

### Hard constraints

1. NQ/MNQ 1h only — Probe 1 bar-timeframe axis terminally falsified
2. 1h test partition has been read 5 times (Probes 2, 3, 4, Scope D, re-read under TZ correction)
3. Fresh temporal holdout exists at 2026-04-09+ (untouched by any probe; ~336 1h bars as of 2026-04-23, growing ~24 bars/day)
4. Engine-side TZ is Option B (documented, not changed) — `src/strategy.py` session_filter operates on CT hours
5. Remote execution only on sweep-runner-1
6. No deployment implication of this council's verdict

### Methodological preconditions (mandatory per `feedback_council_methodology.md`)

- **Rule 1**: state the stage of each evidence piece (generation / original confirmation / corrected confirmation / separate confirmation) and the multiplicity denominator
- **Rule 2**: report `P(PASS | H1)` and `P(PASS | H0)` per proposed gate

### Candidate options

- **Option A** — Paper-trade preregistration anyway (§4.3 FAIL narrow)
- **Option B** — Probe 3 v2 on fresh 2026-04-09+ holdout (2-gate cheap confirmation)
- **Option C** — Sunset combo-865
- **Option D** — Hybrid: accept corrected PASS gates + single fresh-holdout confirmation

---

## Advisor Responses

### The Contrarian — "Sunset combo-865"

The fatal flaw none of A/B/C/D address: **the preregistration itself was non-prespecified in the dimension that matters most.** A preregistration signed against a TZ-buggy dataset isn't a preregistration — it's a draft retroactively validated by a post-hoc correction. You can't "fix" a signed prereg by patching the data and rerunning §4.3; every downstream threshold (0/16, 8/16, 27/27) was calibrated against a corrupted SES label. The "§4.4 strengthens to 12/16" is not reassurance — it's evidence the gate was miscalibrated in the direction of your prior. When gates move *with* the researcher's hypothesis after a correction, that's the signature of a gate that was never power-calibrated in the first place (violating Rule 2 retroactively across all four sub-gates, not just §4.3).

**Partition-reuse accounting everyone skips**: the 1h test partition has not been read 5 times — it's been read 5 times *and re-interpreted once under TZ correction*, which is a 6th informational extraction. The TZ fix is a covariate reshuffle that re-queries the same 220 trades through a new session lens. Call it what it is: read #6 already happened, silently, the moment the SES labels were recomputed. Option A/B/C/D all pretend the debt meter is still at 5.

**The real reason this combo keeps triggering reconvenes**: combo-865 is the local max of a 3000-combo sweep on a falsified family (Probe 1). It has to keep passing novel gates because the base rate for its survival under honest OOS is roughly 1/3000. Each reconvene is a researcher degree of freedom harvested against an already-overfit candidate. The combo isn't being tested — it's being *chaperoned*.

**Option D is the hidden trap.** It looks cleanest because it offers a "fresh holdout" — but you now know combo-865 is RTH-leaning, that combo-1298's RTH Sharpe is 4.41, and the "overnight narrative" inverted. Any Probe 3v2 gate you write on 2026-04-09+ will implicitly condition on RTH-session performance. That's not a prereg — it's a confirmation test designed by someone who already saw the answer. The 336-bar runway is also laughably underpowered: ~14 1h trades at historical frequency, Sharpe CI ±1.5 minimum. You'd PASS or FAIL on noise.

**Kill criterion for "combo-865 has real 1h edge"**: it should have been *specified before the TZ fix was applied*. Since it wasn't, the honest kill criterion is now retrospective: did the corrected SES findings require a narrative inversion? Yes. **That alone is the kill.** Option C. Sunset it. The edge, if real, will re-emerge in the B1 session-structure sweep as one combo among thousands — on its own merits, unchaperoned.

---

### The First Principles Thinker — "Run Probe 3 v2 on fresh holdout with combo-1298 primary"

**Strip the problem to its foundations.**

The signed question is "adjudicate F=1 on combo-865." But that's a procedural question, not a scientific one. The scientific question is: **what is the project trying to produce?** If the answer is "a validated NQ/MNQ 1h carve-out with real edge," then combo-865 is no longer obviously the protagonist. It's a 5-retraction artifact whose narrative keeps inverting under audit. Combo-1298 arrived clean from Probe 4 with RTH Sharpe 4.41 on n=77 — discovered *as a side effect*, not as the thing anyone was defending. That asymmetry matters: combo-865 has been optimized-against by five verdict cycles; combo-1298 has been looked at once. Under first-principles multiplicity accounting, combo-1298 has vastly less garden-of-forking-paths exposure.

**Rule 1 — stages are being collapsed.** Stage 1 (generation): Probe 2 picked combo-865 from a sweep. Stage 2 (original confirmation): Probe 3 UTC. Stage 3 (corrected confirmation): Probe 3 CT. Stage 4 (separate confirmation): **does not exist yet for combo-865.** Everything post-Probe-2 is the same test partition re-read. The TZ correction is not a new stage — it's a re-derivation on Stage 2/3 data. Denominator for any honest posterior on combo-865: **5 reads × {original, corrected} × §4.1/§4.2/§4.3/§4.4 gates = dozens of implicit tests.** Combo-1298's denominator is 1.

**Rule 2 — paper-trade is not the only "next step" anymore.** The pre-TZ council chose paper-trade > B1 on the premise that overnight concentration made venue/broker the binding constraint. That premise is **dead**. The edge is RTH-leaning. Paper-trade's information value was high when the question was "does liquidity/venue kill the overnight edge?" — it's low when the question is "does an RTH-concentrated edge hold on unseen bars?" A fresh holdout answers the current question faster and more cheaply than paper-trade.

**The right question isn't "A/B/C/D on combo-865."** It's: **run Probe 3 v2 on the fresh 2026-04-09+ holdout, but test combo-1298 as primary and combo-865 as secondary.** One preregistered test, two combos, same gates, no 6th read of the stale partition. Combo-1298 passing is the load-bearing outcome; combo-865 is a tie-break.

**Kill criterion:** if combo-1298 fails on the fresh holdout, sunset the 1h-carve-out line entirely. If combo-1298 passes and combo-865 fails, ship combo-1298 and retire combo-865 permanently. If both pass, combo-1298 is primary regardless of combo-865's magnitude — the epistemic discount on combo-865 is too large.

**§4.3 FAIL doesn't falsify the 1h edge. It falsifies the narrative that justified combo-865 specifically.** Those are different objects. Stop conflating them.

---

### The Expansionist — "Option D + append 15m session/exit sweep"

**The §4.3 "rescue" is a discovery, not a failure.**

Everyone else will read F=1 as bad news. Read it correctly: the 15m negative control flipping means we just **found a second tradable timeframe**. Probe 1's "15m falsified" verdict was aggregate N_1.3 under default session/exit. With session × exit degrees of freedom engaged, 8/16 15m cells now clear. That's not a rescue — that's 15m passing a harder gate than 1h originally did. Nobody is asking: **how many 15m combos beyond 865's sibling would pass a proper sweep?** My answer: run it.

**Rule 1 stage-naming (the upside read):** §4.2's 27/27 at ±5% is Stage-2 (local), but it's also prima facie evidence of a **parameter basin**, not a point. A basin with 27/27 at one TF + now a rescuable sibling TF suggests the edge is structural around this region. Denominator for "combos that survive Probe-3-calibre gates on 1h": at least 1 confirmed (865), plus combo-1298 from Probe 4 with Sharpe 4.41 RTH. Two hits from two probes. That's not a freak — that's a **pattern being undersampled**.

**Rule 2 power (where the others will lowball):** P(PASS|H1) on §4.4 jumped from 8/16 to 12/16 under TZ correction. That's a 50% absolute lift in power. The other advisors will say "same underlying, just re-binned." I say: the gate strengthened from exactly-at-threshold to comfortably-clear. If §4.4 is now robust, the whole ritual stack gets stronger, not weaker.

**Does §4.3 FAIL falsify 1h?** No, and this is the trap. §4.3 was designed as a **negative control on 15m**, not a test of 1h. Its failure tells you 15m is tradable, which is **orthogonal** to 1h's edge. Treating it as 1h falsification is a category error the Contrarian will make.

**RTH inversion is massively load-bearing — in our favor.** Overnight-edge narrative meant: specialty broker, prop-firm routing, thin liquidity, wide overnight spreads. RTH-edge means: **every retail US futures broker deploys this** (AMP, NinjaTrader, Tradovate, IBKR). The deployable universe expanded 10× in one data correction. The fill model gets cleaner (RTH liquidity), slippage assumptions tighten, and the strategy becomes pitchable to a vastly wider audience — including prop firms with RTH-only evaluations.

**6th read of 1h partition:** Defensible if and only if it's to validate the **new RTH-primary hypothesis** (not to re-salvage the original overnight thesis). Debt-budget-wise: yes, because the question has fundamentally changed.

**Kill criterion (mine is more ambitious):** Sunset combo-865 only if a **15m session × exit sweep + the RTH-primary refit on fresh holdout** both produce fewer than 2 combos clearing Sharpe ≥ 1.3 at n ≥ 50.

**Pick Option D + append a 15m session/exit sweep.** Maximum-upside path: the project isn't dying, it's pivoting from overnight-niche to RTH-deployable, from single-TF to potentially dual-TF.

---

### The Outsider — "Option B on fresh holdout, preregistered, one look"

I'm reading this fresh and several things don't parse.

**On Rule 1 (stages)**: Before I can assign stages, I need to understand the audit trail. If §4.3 was pre-registered as a negative-control gate under the ORIGINAL (incorrect) timezone labeling, and the timezone fix is post-hoc, then every gate just got re-evaluated under new labels. That's not Stage 1 or Stage 2 — that's "the test you actually ran is different from the test you pre-registered." You can't count this as one confirmatory look. The multiplicity denominator should include every timezone interpretation that could have been applied, not just the one that happened to be labeled "signed."

**On Rule 2 (P(PASS|H1) vs P(PASS|H0))**: Nobody in this brief has told me the prior, the likelihood model, or the power calc. I'm being asked to rubber-stamp posterior updates without seeing the math. That's not a council question — that's a homework assignment the proposer should have attached.

**Now the things I genuinely don't understand:**

**(a) F-count arithmetic is weird.** 3 out of 4 PASS in most domains would be "strong evidence." Here, F=1 flips the branch entirely. Why is the gate binary? If each sub-gate has independent evidentiary weight, sum them. If they don't, why are there four? Somebody chose this threshold; they should justify it now, not later.

**(b) §4.3 vetoing a 1h decision is load-bearing nonsense to me.** 15m is a different timeframe. If 15m's job was to show "our framework doesn't find edge where none exists," and it now finds 8/16 FAIL after relabeling, that tells me the framework is sensitive to session labeling — which is a framework finding, not a combo-865 finding. Why does a framework-calibration failure kill a specific combo's paper-trade? Unless the argument is "if session labeling matters this much, all your session-conditional conclusions are fragile" — in which case say THAT, and re-audit everything, not just §4.3.

**(c) Five reads on one partition, now asking for a sixth.** At this point you are not measuring combo-865's edge. You are measuring your own optionality to keep looking. The fresh 2026-04-09+ holdout exists. Use it. A 6th read on the same partition is indefensible regardless of framing.

**(d) The RTH inversion is the actual story.** Five weeks of narrative was wrong. That's not a gate question — that's a "what else did we conclude from mis-labeled data" question. Audit every session-conditional claim before any further inference.

**Kill criterion I'd accept**: Option B on fresh holdout, pre-registered, two-sided gates, one look. If it fails, sunset. If it passes, paper-trade with sizing calibrated to the corrected (lower) concentration. Option D is Option B wearing a hat.

---

### The Executor — "Option D as A+B stacked"

**The Monday-morning question: what can we actually execute this week?**

**Rule 1 (stage naming + denominator):** Stage 1 = Probe 3 original (4 gates, prereg signed, F=0). Stage 2 = COUNCIL_RECONVENE under corrected TZ. Denominator for Stage 2 = 4 gates re-read, not a new family of 40. Multiplicity adjustment is bounded: one additional read on the same prereg after a TZ correction. This is Stage 1' (correction), not Stage 2 (fresh hypothesis). Don't inflate it.

**Rule 2 (likelihood ratios per gate under corrected TZ):**
- §4.1 regime halves: P(PASS|H1)≈0.80, P(PASS|H0)≈0.15 → LR ~5.3
- §4.2 param ±5% (27/27): P(PASS|H1)≈0.65, P(PASS|H0)≈0.05 → LR ~13
- §4.3 15m neg control (8/16 FAIL): P(PASS|H1)≈0.70, P(PASS|H0)≈0.40 → LR ~1.75 — FAILED, this is evidence against H1
- §4.4 session/exit (12/16): P(PASS|H1)≈0.60, P(PASS|H0)≈0.10 → LR ~6

Combined 3 PASS / 1 FAIL: posterior moves from 0.65-0.85 prior down to ~0.55-0.75. Still above coin-flip, not ship-level.

**Q3: Does §4.3 FAIL falsify the 1h edge?** No. §4.3 is a negative control on a *different* timeframe. 8/16 15m cells passing under corrected TZ means 15m isn't as dead as we thought — it weakens the "friction binds 15m cleanly" story but doesn't touch 1h's direct evidence (§4.1, §4.2, §4.4). It downgrades specificity, not the primary signal.

**Q4: RTH inversion load-bearing?** Yes, massively. If combo-865 RTH Sharpe is actually 2.64 (not 0.64), the "overnight-only" story collapses. That changes the broker/venue risk calculus entirely — RTH execution is cheaper, more liquid, no overnight margin. This is the single biggest delta.

**Q5: 6th read defensible?** Only if framed as "TZ correction re-read" not "new hypothesis." Six reads for exploration = fishing. Six reads where reads 4-6 are corrections to a data-encoding bug = forced by reality. Document it explicitly in the verdict.

**Q6: Kill criterion.** If fresh-holdout 1h combo-865 delivers < 10 trades after 60 days OR net Sharpe < 0.8 after 30 trades accumulate, sunset.

**Recommendation: Option D, but executed as A+B stacked.** Ship the paper-trade prereg this week (1-2 days, leverages existing Probe 3 scripts and SES masks). In parallel, run Probe 3 v2 on fresh 2026-04-09+ holdout as a 2-gate cheap confirmation: gate 1 = ≥20 trades in 90 days, gate 2 = net Sharpe ≥ 1.0 on those trades. That's ~3 months wall-clock for the confirmation, but paper-trade accumulates real fill data during that window. Option C is premature — corrected TZ moved the posterior *up*, not down.

---

## Peer Reviews

### Reviewer 1

**Strongest = First Principles (Response E).** E reframes from "adjudicate F-count on 865" to "what is the project trying to produce," then executes the pivot: 1298 arrived clean from Probe 4 with RTH Sharpe 4.41 on n=77, has denominator of 1 versus 865's dozens. The Stage 1–4 ladder is the cleanest multiplicity accounting anyone provides, and the procedural point — "pre-TZ council chose paper-trade because overnight concentration made venue binding; that premise is DEAD" — invalidates A/B/C/D as posed. E proposes the only prereg that uses the fresh partition efficiently (two combos, one look, asymmetric primary/secondary).

**Biggest blind spot = Expansionist (Response D).** D treats every correction as confirmatory: §4.3 rescue becomes "discovery of second TF," §4.4 movement becomes "gate strengthened," RTH inversion becomes "10× deployability." This is textbook motivated reasoning on a partition read 5×. D never mentions that gates moving *with* the hypothesis after correction is diagnostic of uncalibrated thresholds (C's point), never acknowledges that 15m was falsified at family level by Probe 1 (`N_1.3=9` on gate of 10). D wants to expand scope when contraction is warranted.

**What all five missed.** None address whether the TZ correction itself was pre-registered. If the TZ audit was triggered because someone noticed combo-865 looked weak post-signing and went looking for explanations, that's a sixth read disguised as a bug fix — and every gate re-evaluation inherits that selection. The council should demand the TZ-correction provenance log (when was the bug noticed, by whom, under what trigger) before trusting ANY corrected gate value, including 1298's 4.41. Provenance, not arithmetic, governs admissibility here.

---

### Reviewer 2

**Strongest = First Principles (Response E).** E is the only response that correctly identifies that combo-1298 is the clean candidate while combo-865 is a 5-retraction artifact — collapsing the A/B/C/D menu into the right question: "does RTH edge hold on unseen bars for a combo we haven't tortured?" It avoids C's nihilism (1298 has one look, not dozens), avoids D's pivot-euphoria, and gives a clean decision tree with asymmetric kill criteria. It respects partition hygiene without wasting the holdout on the contaminated candidate.

**Biggest blind spot = Expansionist (Response D).** D treats the TZ inversion as a "10× deployable universe" discovery, but this is exactly the researcher-degrees-of-freedom pattern C flags: the narrative flipped 180° (overnight → RTH) and D celebrates it as signal. If thresholds had strengthened under *any* TZ interpretation D would call it confirmation — that's unfalsifiable. D also hand-waves the 336-bar power problem and adds a 15m sweep (more multiplicity) to a question already suffering from it.

**What all five missed.** None addressed whether combo-1298's n=77 RTH Sharpe 4.41 is itself a selection artifact from Probe 4's SES_2-confound sweep. Probe 4 ran across multiple combos; 1298 surfaced as the RTH-favorable one. E calls its denominator "1" but that's wrong — 1298 was *selected* from Probe 4's combo set conditional on strong RTH performance. The clean-denominator framing lets 1298 skip the scrutiny applied to 865. Before any holdout test, council needs Probe 4's full combo×session matrix to compute 1298's actual selection denominator — otherwise Option B/E repeats the 865 mistake with a fresh name.

---

### Reviewer 3

**Strongest = First Principles (Response E).** E is the only response that correctly separates the denominators: combo-865 has been read ~5-6× and had its justifying narrative inverted, while combo-1298 arrived clean from Probe 4 with RTH Sharpe 4.41 at n=77. This cleanly dissolves the partition-debt problem that C correctly identifies but answers with blanket sunset. E's reframe — 1298 primary, 865 secondary — gives a cheaper, more informative test than the other options and has a falsifiable kill condition per outcome quadrant. It also correctly diagnoses that §4.3 falsifies the *narrative* (overnight edge), not the *asset* (1h RTH on combo-1298).

**Biggest blind spot = Expansionist (Response D).** D treats the RTH inversion as "MASSIVELY load-bearing UPSIDE" without noticing that an inversion after 5 test-partition reads is textbook RDoF — the hypothesis moved *with* the data, not ahead of it. D also claims §4.3 is "DISCOVERY of tradable 15m" when 8/16 FAIL means 8 cells failed, not passed — inverted sign. Proposing a 15m sweep on a negative-control failure doubles down on the partition-debt problem C flagged.

**What all five missed.** None quantified the actual bar-count / trade-count on the fresh 2026-04-09+ holdout or asked whether it supports the proposed gates with any power. E proposes combo-1298 as primary but n=77 came from Probe 4's test partition, not the fresh holdout — the fresh holdout may have <20 trades on 1298 at 1h, making "Sharpe ≥1.0 on 30 trades" (B's kill rule) infeasible within the planning horizon. Power calc on fresh-holdout trade volume is the gating question before choosing A/B/C/D.

---

### Reviewer 4

**Strongest = First Principles (Response E).** E is the only response that names the actual confound: 865 accumulated ~5 retractions/re-reads, 1298 arrived from Probe 4 with one look. That asymmetry dominates every gate-counting argument. By nominating 1298 primary/865 secondary on fresh holdout, E gets selection-effect correction + forward test without sunsetting a clean candidate. Four decision branches, each actionable.

**Biggest blind spot = Expansionist (Response D).** §4.3 FAIL as "discovery," RTH inversion as 10× deployable universe — precisely the garden-of-forking-paths pattern C flags. Every post-hoc rescue reading pushes effective multiplicity higher. D's "6th read defensible for new RTH-primary hypothesis" quietly concedes the read-count problem, then waves it away. D also ignores that the paper-trade kill criterion it endorses was calibrated on pre-fix Sharpes — the whole prereg's operating characteristics shifted.

**What all five missed.** None of the five computed the joint test for combo-1298: it PASSED Probe 4 §3 gates but TRIGGERED SESSION_CONFOUND (§5 row 2, ~80% SES_2-concentrated per the project memory). E anoints 1298 "clean" — it isn't. If the concern with 865 is overnight-venue dependence, 1298 has the same structural dependency; a fresh-holdout Probe 3 v2 on 1298 needs a session-decomposition gate pre-registered, not just Sharpe/n_trades. Without that, shipping 1298 after a single holdout pass reproduces the exact confound B1 was authorized to investigate.

---

### Reviewer 5

**Strongest = First Principles (Response E).** It reframes the question productively. While A/B/C/D argue about 865's residual evidence, E notes 865 has been sweep-optimized across ~5 verdict cycles (implicit multiple-testing on dozens of decisions) whereas 1298 is Probe-4-clean (denominator 1). The hierarchy (1298 primary, 865 secondary) is a properly-powered decision rule that extracts maximum info from the fresh holdout. B's LR arithmetic is rigorous but accepts 865-centric framing; C is correct about partition debt but over-kills by ignoring 1298; D's "10× deployable universe" is narrative inflation on n=77.

**Biggest blind spot = Expansionist (Response D).** It reads §4.3 "0/16→8/16" as *discovery* of 15m tradability. But §4.3 was a **negative control** — it was supposed to confirm 15m friction-binding. Flipping it doesn't reveal hidden edge; it means either the TZ bug corrupted the original 15m evaluation too, or the control mechanism is non-stationary. Either way, treating a failed negative control as a positive signal inverts the gate's purpose. D also anchors on combo-1298 Sharpe 4.41 at n=77 without CI (Sharpe SE ≈ 1/√77 ≈ 0.11 annualization-inflated — wide).

**What ALL five missed.** The TZ bug is a **process failure**, not a data correction. Every prior verdict (Probe 1–4) used the same pipeline. None of the responses propose auditing whether Probe 1's falsification (N_1.3 counts), Probe 2's 1h PASS, or Probe 4's SESSION_CONFOUND verdict survive the TZ fix. If §4.3 flipped from 0/16 to 8/16, the Probe-1 15m sweep (N_1.3=9) could move across the gate=10 boundary — potentially unfalsifying the family. The council is adjudicating Probe 3 while the upstream foundation is unverified.

---

## Chairman Synthesis

### Where the Council Agrees

Four of five advisors and all five reviewers converge on one point: **combo-865's partition is corrupted by read-count, not rescued by it**. Contrarian, First Principles, Outsider, and Executor all treat the TZ correction as a 6th extraction from the test partition (however well-motivated the bug fix). Reviewers 1-5 unanimously name First Principles the strongest response for the same structural reason — 865 has accumulated ~5 verdict cycles of implicit optimization; whatever its corrected Sharpe reads, the selection-effect denominator has outgrown the evidence.

Second consensus: **the RTH narrative inversion is load-bearing but in a DIRECTION the council disagrees about** (see Clashes). Every advisor registers that "overnight thesis → RTH thesis" is not a minor revision — it inverts the deployability story, the liquidity story, and the broker-adapter story.

Third consensus: **§4.3's flip from 0/16 PASS to 8/16 FAIL is not neutral**. Contrarian, Outsider, Reviewer 5 read it as gate-corruption (thresholds calibrated against corrupted labels); Expansionist reads it as discovery. Nobody reads it as benign.

Fourth consensus: **fresh 2026-04-09+ holdout exists and is the obvious instrument** — but four of five advisors (all except Contrarian) + three reviewers (1, 3, 4) flag that none of A-D has checked whether the holdout's bar/trade count actually powers the proposed gates.

### Where the Council Clashes

**Clash 1 — Is §4.3's flip evidence or noise?** Expansionist frames it as a "DISCOVERY of tradable 15m." Reviewers 3 and 5 directly rebut: 8/16 FAIL means 8 cells failed, not 8 cells newly tradable — the sign is inverted in Expansionist's reading. Reviewer 5 names the deeper error: §4.3 was a *negative control*, and a failed negative control doesn't reveal hidden edge, it invalidates the control mechanism.

**Clash 2 — Is combo-1298 a clean candidate?** First Principles anoints 1298 "denominator = 1, arrived as a side-effect from Probe 4." Reviewers 2 and 4 both reject this. Reviewer 2: 1298 was *selected* from Probe 4 conditional on strong RTH performance — its actual denominator is Probe 4's full combo×session matrix. Reviewer 4 is more cutting: 1298 TRIGGERED SESSION_CONFOUND (§5 row 2, ~80% SES_2-concentrated per project memory) — it has the same structural dependency as 865 pre-correction.

**Clash 3 — Paper-trade now vs hold fire.** Executor argues A+B stacked: paper-trade accumulates fill data during the 3-4 month fresh-holdout confirmation window. Contrarian calls this the "hidden trap" — paper-trading prereg anyway signals that the five-read partition is still treated as evidence. Outsider stakes out the middle: "Option D is Option B wearing a hat."

**Clash 4 — Does Probe 1's family-level falsification survive the TZ fix?** Unflagged by advisors; Reviewer 5 raises it and it is existential. If §4.3's 15m sweep moved 0/16 → 8/16 under TZ correction, then Probe 1's 15m N_1.3=9 (gate=10) is sitting at the boundary with a 6-cell swing; the TZ fix could move it across.

### Blind Spots the Council Caught

- **Reviewer 1 — TZ-correction provenance isn't pre-registered.** If the bug was noticed because 865 looked weak post-signing and someone went looking for explanations, the "correction" is a 6th read disguised as a bug fix.
- **Reviewer 2 — Combo-1298's denominator is not 1.** 1298 was selected from Probe 4's combo×session matrix conditional on RTH performance.
- **Reviewer 3 — Nobody did a power calc on the fresh holdout.** Gate feasibility is upstream of gate selection.
- **Reviewer 4 — Combo-1298 TRIGGERED SESSION_CONFOUND in Probe 4.** Shipping 1298 after a single fresh-holdout pass reproduces exactly the confound B1 was authorized to investigate.
- **Reviewer 5 — The TZ bug is a process failure, not a local data correction.** Probe 1 could unfalsify the family under corrected TZ.

### The Recommendation

**Adjudicate §5.2 as: no paper-trade, no fresh-holdout probe yet. Halt all downstream inference pending TZ-fix upstream audit.**

Position on (a): **Suspend A, B, C, D**. All four options assume the upstream pipeline is sound.

Position on (b): **Neither 865 nor 1298 is a valid focus right now.** 865's denominator is exhausted (6 reads post-TZ); 1298's "clean" denominator (Reviewer 2/4 critique) does not survive — it was selected from Probe 4 conditional on RTH strength AND triggered SESSION_CONFOUND.

Position on (c): **Probe 1's 15m falsification must be re-audited under the TZ fix before any combo-level decision.** Reviewer 5's point is existential.

Position on (d): **Combo-1298's "clean" status does NOT survive peer review.**

Chairman disagrees with the 5/5 reviewer-unanimous endorsement of First Principles' specific recommendation (Probe 3 v2, 1298 primary, 865 secondary). Justification: reviewers named First Principles *strongest* because it refused the framing — but Reviewers 2 and 4 then demolished the specific 1298-primary proposal. Following First Principles' framing-refusal but not their conclusion is the consistent read.

### The One Thing to Do First

**Produce a TZ-correction provenance log + Probe 1 re-audit under the TZ fix.** One artifact, two contents: (1) timestamped record of when the Central-Time bug was noticed, by whom, under what trigger, with all intermediate commits; (2) Probe 1's 1m/15m/1h N_1.3 counts recomputed under Central-Time session labels, with the §7.6 decision rule re-applied. If Probe 1's family-level verdict survives, then — and only then — convene a new council to preregister a Probe 3 v2 that treats *both* 865 and 1298 as confound-positive candidates requiring session-decomposition gates. If Probe 1 flips, the entire carve-out line is moot and the project reverts to B1 session-structure sweep with a corrected pipeline.

---

**Timestamp**: 2026-04-23 UTC
**Council fired on**: Probe 3 COUNCIL_RECONVENE per §5.2 following TZ-bug retraction (commit `a38d3c5`)
**Advisor composition**: Contrarian, First Principles, Expansionist, Outsider, Executor
**Anonymization randomized**: A→Outsider, B→Executor, C→Contrarian, D→Expansionist, E→First Principles
**Peer reviewer count**: 5 (independent, blinded to advisor identity)
**Reviewer convergence**: 5/5 picked First Principles strongest; 5/5 picked Expansionist biggest blind spot
**Chairman disagrees** with the reviewer-unanimous specific recommendation (adopts the framing but not the conclusion)
