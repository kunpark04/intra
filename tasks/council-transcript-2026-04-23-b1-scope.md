# LLM Council Transcript — B1 Scope (2026-04-23 UTC)

**Question counciled**: Scope the B1 preregistration for the session-structure sweep authorized by the Probe 4 SESSION_CONFOUND verdict (commit `c419391`, 2026-04-23 UTC).

**Anonymization mapping revealed:**
- Response A → The Executor
- Response B → The Contrarian
- Response C → The Outsider
- Response D → The First Principles Thinker
- Response E → The Expansionist

---

## Framed Question

Scope the B1 preregistration for the session-structure sweep authorized by the Probe 4 SESSION_CONFOUND verdict. Probe 4 showed combos 1298 and 664 both ~80% trade-concentrated in SES_2 (ex-RTH) on the 1h held-out test partition with SES_1 RTH failing absolute for both. Primary question: what is the right scope for B1, a session-structure sweep across multiple combos on 1h NQ, subject to Rule 1 (stage-naming & multiplicity denominator) and Rule 2 (P(PASS|H1) and P(PASS|H0) per gate)? Four candidate scopes were provided: A (top-k confirmation on test), B (train-select / test-confirm), C (portfolio-level test), D (sub-session partitioning on {865, 1298, 664}).

Hard constraints: NQ/MNQ 1h only (bar-timeframe axis falsified by Probe 1); 1h test partition already consumed 3 times (Probes 2, 3, 4); net-of-$5/contract-RT only; remote execution only on sweep-runner-1; B1 passing does NOT authorize deployment — it's research-stage family-level characterization only.

### Methodological preconditions (mandatory per `feedback_council_methodology.md`)

- **Rule 1 — Stage naming**: identify whether data under evaluation came from generation (in-sample sweep / selection / filtering) or confirmation (held-out pre-registered test). If a multiplicity correction is proposed, state the denominator it applies to and why it is correct for this stage. For a pre-registered single test, the denominator is 1, not the sweep count.
- **Rule 2 — Power calibration**: for each gate, report `P(gate PASS | H1)` and `P(gate PASS | H0)` with estimation method. Single-threshold gates without power/null calibration are inadmissible. Prefer ≥ 3 independent gates with heterogeneous failure modes whose joint Bayes factor can be computed.

---

## Advisor Responses

### The Contrarian

B1 Preregistration Attack. Before anything: Rule 1 stage-naming sleight-of-hand is already embedded in every scope. Scopes A, C, and D all propose running on the 1h test partition. That partition has been consumed three times. Call it what it is: **the "test partition" is now a training-adjacent surface**. Pricing this as a "Bonferroni-1" single test is the exact 2026-04-21 Probe 3 error dressed in new clothes. The honest denominator for any 1h-test-partition test is not 1 and not the sweep count — it's **an unknown researcher-DOF-inflated number bounded below by 3 prior probes + whatever selection pressure produced the combo set**. Every advisor who writes "denominator = 1 under Rule 1" for Scope C is running the Probe 3 council's mistake in reverse.

**Fatal flaws per scope:**

**Scope A** — "Selection bias; only characterizes the top of the family" — the stated con understates it. Top-k by training-Sharpe on a family Probe 1 *falsified* (N_1.3=4 vs gate 10) is selection on noise. You'd be confirming that noise-selected combos cluster overnight because *all* 1h noise clusters overnight. P(PASS|H0) is not the nominal α — it's inflated by the training-gross-Sharpe floor conditioning on session structure implicitly.

**Scope B** — the "cleanest" one, and therefore the most dangerous. "Selection criterion is the thing being confirmed" sounds rigorous, but **the selection criterion is computed on the training partition and the mechanism being confirmed is session-structural**. Session structure leaks across the train/test split because the *clock* doesn't know about the split. Overnight is overnight on both sides. You are not Stage-1/Stage-2 separating anything — you're selecting on session, confirming on session, and calling it partition discipline.

**Scope C** — portfolio-level, "multiplicity denominator = 1." No. Portfolio construction (equal-weight vs Sharpe-weight vs correlation-pruned) is **three gardens-of-forking-paths** before you run a single trade. The deliverable asks for "researcher degrees of freedom" scoring — Scope C's RDoF is the highest, not the lowest.

**Scope D** — honest about N=3 being too small, but there's a deeper problem: **the 17:00-18:00 CME settlement halt means Probe 4's SES_2 includes a window where no trades should execute anyway**. Sub-partitioning might just recover the Probe-3 narrow-wraparound result with extra steps. If D's "kill criterion" is "16:00-18:00 doesn't contribute," that's nearly pre-determined by market microstructure, not by your hypothesis.

**Rule 2 attack:** Every scope proposes Welch-t or Sharpe-difference gates. `P(PASS|H1)` requires an assumed effect size. Probe 3 and Probe 4 both observed ~3× overnight/RTH ratios — but those *are the training data for your H1*. Using observed ratios to power-calibrate a confirmation test is circular.

**Kill criterion that's actually falsifying:** None of A/B/C/D proposes one that Probe 3 + Probe 4 haven't already pre-satisfied. A real kill criterion: pre-register that **SES_1 RTH must remain negative or zero on an independent NQ-1h data vendor's partition** (Polygon, Databento) — if RTH is net-profitable on clean vendor data, the session-structure thesis dies. None of the candidates propose this.

**Recommendation: do not launch B1 until partition exhaustion is priced.**

---

### The First Principles Thinker

Stop. Before answering A/B/C/D, ask what B1 actually buys you that you don't already have.

**What are you trying to learn?** Three candidate questions are getting conflated:

(Q1) *Does 1h NQ Z-score mean-reversion have a real edge?* Probe 1 already answered this at the family level: no. N_1.3 = 4 of 1500. Sunset fired. That question is closed.

(Q2) *Is there a tradable pocket inside the falsified family?* Probes 2-4 answered this: yes, for combos {865, 1298, 664}, concentrated overnight. You have three carve-outs. You have the mechanism narrative (overnight/RTH ~3.45× ratio, SES_2 ~80% trade concentration, SES_1 RTH fails absolute).

(Q3) *Does the overnight-structural story generalize to a basin large enough to justify deployment confidence?* This is what B1 ostensibly tests. But re-read constraint #5: **B1 passing does not authorize deployment.** Deployment still requires a fresh downstream probe.

So B1 is research-stage family characterization that by your own hard constraints cannot close the deployment loop. What does it actually produce? A published claim: "the inefficiency is session-structural across a 1h combo basin." That's a narrative asset, not a decision input.

**Rule 1 applied honestly to the real question.** The test partition has been consumed 3 times on this exact signal family. Partition-reuse priors are now meaningfully informative — researcher degrees of freedom across Probes 2, 3, 4 have already biased which patterns you're chasing. Every scope A-D further consumes it. Scope B "train-select / test-confirm" is the only one that re-imposes partition discipline, but it pays full Stage 1 multiplicity on a sweep you already know the answer to at the family level.

**Rule 2 applied honestly.** Under H0 = "no overnight structural edge," you already have three combos refuting H0. Under H1 = "overnight edge generalizes," your effective n is ~3 post-Probes 2-4. The likelihood ratio from another 20-50 combo basin test is bounded — you're mostly re-measuring what you already measured.

**Reframe.** The productive question isn't "does the basin confirm session-structure." It's: **given that combo-865 passed Probe 3 PAPER_TRADE and 1298/664 just passed Probe 4 absolute gates but failed §5 session-purity, what is the minimum-ceremony preregistration that authorizes a paper-trade decision on the combos you already have?**

That probe would specify: (i) the overnight sub-window definition binding (Scope D's pure-GLOBEX vs 16:00-18:00 split is load-bearing because it picks your execution clock), (ii) a forward-walking paper-trade protocol with stopping rules, (iii) an ex-ante sizing floor tied to per-trade net $, not Sharpe. That answers a decision, not a narrative.

**My recommendation**: don't run B1 as scoped. Run Scope D as a *characterization footnote* (cheap, one combo set), then write the paper-trade preregistration. You're three probes deep into a falsified family — the remaining value is in deploying the carve-outs or stopping, not in re-characterizing them.

---

### The Expansionist

**Stage & denominator (Rule 1).** B1 is CONFIRMATION on the 1h test partition for the *session-structural* hypothesis. The sweep ran in Probe 1 (generation); the property "overnight > RTH" emerged from Probes 3 and 4 (two realizations). B1's primary gate denominator is **1 — a single prespecified family-level test**. What I add: partition the test window chronologically into a **discovery half (2024-10-22 → 2025-09-14) and a confirmation half (2025-09-15 → 2026-04-08)**. Use the discovery half for ancillary characterization (free), reserve the confirmation half for the one prespecified gate. This buys us a hold-out *inside* test without reusing it against the primary gate.

**The upside everyone's missing.** A binary PASS/FAIL on "is it overnight-structural" wastes 90% of the compute. If SES_2 is real, the economically load-bearing questions are: **which sub-window, under which conditions, with what decay**. Piggyback all of it onto the same remote run — marginal cost is near-zero.

**Recommended Scope — E (Expanded B).** Train-select on training partition (Scope B's discipline), but run the selected set on test with a **tensor decomposition**, not just SES_0/1/2:
- **Session × sub-window**: GLOBEX-Asia [18:00–02:00 ET], GLOBEX-Europe [02:00–09:30 ET], post-RTH [16:00–18:00 ET including settlement halt]
- **Day-of-week** (Sun-open and Fri-close are structurally different liquidity regimes)
- **Volatility regime** (top/bottom tercile of trailing 20-bar realized vol — is the edge vol-conditional?)
- **Macro event windows** (FOMC, NFP, CPI days vs clean days — does overnight edge survive or amplify on event-Asia follow-through?)
- **Post-gap vs no-gap Sundays** — overnight mean-reversion should be strongest after weekend gap opens

**Primary gate (1 test, denom=1):** On confirmation-half test partition, pooled per-trade net PnL in GLOBEX-overnight ∪ Europe > pooled per-trade net PnL in RTH, Welch-t ≥ 2.5. `P(PASS|H1≈Probe4 effect size 3.45x)` ≈ 0.88 via per-trade bootstrap; `P(PASS|H0)` ≈ 0.006 one-sided. Bayes factor ≈ 145×.

**Sizing:** 30 combos (top-30 by training-partition session-decomposition score, not by Sharpe — selects on the property being tested, not a confound). Ancillary decompositions reported as *characterization*, not gates.

**Kill criterion (falsifiability):** SES_2 edge concentrates in **post-RTH [16:00–18:00]** rather than pure-overnight, OR disappears in Europe session — either outcome falsifies "overnight inefficiency" and points instead at a settlement-mechanics or US-session-spillover artifact. That's a *different* hypothesis worth knowing about, but it kills the current one.

**Session binding:** Do NOT endorse Amendment 1's coarse SES_2 as-is. Sub-partition mandatory — the post-RTH window is mechanically different (settlement halt, thin book) and lumping it with Asia hides the locus of edge.

**Max learning per dollar:** With one remote run we map the full opportunity surface — sub-session, vol-conditional, event-conditional, day-of-week. If B1 passes, we know *exactly* where to aim B2.

---

### The Outsider

**Wait. Before anything else, I need to flag things this document treats as settled that a fresh reader cannot parse.**

**(1) What is "session-structure inefficiency" claiming to be?** The document glides from "80% of trades are in SES_2" directly to "the edge is overnight-structural." But these are not the same claim. NQ overnight has thinner volume, which mechanically produces larger price deviations relative to a rolling std. A Z-score mean-reversion signal will therefore *fire more often* overnight — not because there's more edge per opportunity, but because there are more opportunities crossing threshold. The per-trade numbers (+$1,844 vs -$194 for combo 1298) *do* suggest it's more than just count. But has anyone shown me the *gross* per-trade figures net of the fact that overnight stretches take longer to revert, tying up the book? I don't see a holding-time or opportunity-cost axis anywhere.

**(2) Rule 1 stage analysis.** Scopes A, C, and D all propose to *confirm* on a partition that's been touched 3 times already. That's not confirmation, that's the fourth dip. Scope B is the only one with clean partition hygiene — generate on train, confirm on test, *once*. Denominator for a single pre-registered contrast under Scope B = 1. For A/C/D, the denominator has to include a partition-reuse discount, and I haven't seen anyone in this project actually compute one rigorously (the Probe 3 council failed at this).

**(3) Rule 2 — I genuinely cannot estimate P(PASS|H1) vs P(PASS|H0) for the proposed primary gates as written, and I notice nobody has defined H1 precisely.** What is H1 here? "Overnight Z-score mean-reversion has economic edge"? Or "the session-ratio pattern generalizes across parameter realizations"? These require different simulations. The document assumes we know.

**(4) The thing that's been stopped being questioned: why does SES_2 span 16:00-18:00 ET including the settlement halt?** That's a 2-hour window where liquidity collapses AND a formal trading halt occurs. Bundling it with GLOBEX overnight is a category error. Scope D's partitioning is the only one addressing this, and it's scoped to three combos.

**Scoring:** B > D > A > C. **B** has the cleanest stage discipline but scores a compound selection→confirmation pipeline, which is exactly the kind of thing Probes 2/3/4 keep failing. **D** is small-N but surgical and asks the right mechanical question. **A** reuses the test partition without pricing it. **C** portfolio-averages away the exact heterogeneity we're trying to characterize.

**Kill criterion I want stated:** if the pure-overnight 18:00-09:30 window *alone* doesn't dominate, the "session-structure" framing is wrong and the real finding is "thin-liquidity Z-score noise harvesting," which has a completely different risk profile and is not an edge anyone should size on.

**My recommendation:** Run D first as a cheap mechanism-check on the partition we've already consumed. Only if D shows 18:00-09:30 dominance, then run B with that as a prespecified screening criterion. Do not run A or C.

---

### The Executor

**Methodological preconditions (upfront, because the council demands it):**

**Rule 1 (stages):** A and C operate on generation (Probe 1 training sweep already selected these combos) → confirmation needs the 1h test partition, which has been touched 3×. Denominator for multiplicity = number of gates × combos evaluated on test. B separates cleanly: sweep on training = generation, test = confirmation. D is pure confirmation on an already-identified trio, denominator = 3 combos × 2 sub-windows.

**Rule 2 (power):** I can't give exact P(PASS|H1)/P(PASS|H0) without running the simulation — but scope D (n=3, 2 sub-windows, binomial on sign) is the only one where I can compute it in <30 min on paper. A/B/C need Monte Carlo null calibration before sign-off.

**Monday-morning version of each:**

**Scope A** — Pull top-20 combos from Probe 1's 1h `audit_full_net_sharpe` ranking, run `_probe4_readout.py` in a loop across the test partition with SES_1/SES_2 decomposition. Reuses: Probe 4 session-decomp code verbatim, swap the combo list. Rebuild: none. **Runtime: ~45 min** (Probe 4 did 2 combos in ~4 min across full test partition with session decomp; linear in combo count, ~2 min/combo × 20 = 40 min + overhead).

**Scope B** — Fire `param_sweep.py --range-mode v11` on the **training** partition only for 1500 combos on 1h, then select by `(SES_2 − SES_1)` score, then run selected set on test. **Runtime estimate: Probe 1's 1500-combo 1h sweep took ~6-8h on sweep-runner-1 (train+test, 5y data).** Training-only halves that → ~3-4h for the sweep + ~30 min for the confirmation run. **Total ~4-5h.**

**Scope C** — Same data pull as A, aggregate trades across top-50 into a single PnL series, compute Sharpe on SES_2 vs SES_1 pools. **Runtime: ~75 min** (50 combos × 2 min + aggregation notebook).

**Scope D** — 3 combos × 2 sub-windows (18:00-09:30 pure overnight vs 16:00-18:00 post-RTH+halt). Modify `_probe4_readout.py`'s session mask function to add a third bucket. **Runtime: ~10 min.** This is today, not Monday.

**MVP for end-of-week:** Scope D first (10 min, answers "is the edge pure overnight or is post-RTH/halt carrying load?" — mechanistic win or mechanistic kill). Then Scope A (45 min, answers "does the SES_2 effect generalize beyond the three combos we've now looked at?"). Combined: ~1 hour of compute, both can be prereg'd in a single afternoon.

**End-of-month:** Add Scope B as the family-level confirmation.

**Tightest loop:** D. Prereg is 1 page (combos known, gates = binomial sign test on 3×2 = 6 cells, kill = mixed signs). Result in hand same day. Recommend: **D this week, A next week, B only if D+A both pass.** Skip C — portfolio aggregation confounds the session-decomposition question with covariance structure.

---

## Peer Reviews

### Reviewer 1

**1. Strongest: Response D (First Principles).**

D is the only one that refuses the prompt's framing. It correctly identifies that B1 is narrative characterization inside a family Probe 1 already falsified, and that constraint #5 prevents B1 from closing the deployment loop regardless of outcome. D reframes toward the decision-relevant probe (paper-trade preregistration on combos already in hand) rather than burning a fourth test-partition consumption on a question whose answer is bounded by n=3 prior realizations. B's partition-exhaustion attack is sharp, but B stops at critique; D proposes the productive pivot.

**2. Biggest blind spot: Response E (Expansionist).**

E proposes a five-axis tensor decomposition (session × sub-window × DoW × vol regime × macro events × gap/no-gap) and calls the extras "characterization, not gates" — this is textbook garden-of-forking-paths laundering. Claiming denom=1 because only one contrast is "primary" while reporting 30+ ancillary cells is exactly the Probe 3 council error B flags. E's P(PASS|H1)≈0.88 / P(PASS|H0)≈0.006 / BF≈145× numbers are presented without showing the per-trade bootstrap calibration, and the H1 effect size (3.45×) is lifted from the data generating the hypothesis — C and B both flagged this circularity; E ignores it.

**3. What ALL five missed.**

None priced the opportunity cost of B1 compute against the paper-trade runway. Combo-865 has a signed Probe 3 PAPER_TRADE verdict; 1298/664 passed Probe 4 absolutes. Every week spent on B1 is a week of forward-walking data not being collected on a live-but-paper decision. Forward paper-trade data is the only genuinely fresh partition available — and it's generated by the clock, not by researcher choice. D gestured at this but didn't quantify it.

---

### Reviewer 2

**Strongest: D (First Principles).** It reframes correctly — B1 is not a decision input under the existing constraint that passing doesn't authorize deployment. With the family falsified and three combos already characterized, another 20-50 combo basin re-measures what's known. D correctly routes to paper-trade prereg on {865, 1298, 664} as the actual value-producing next step. B and C diagnose partition exhaustion well but stop at "don't run"; D proposes the productive alternative.

**Biggest blind spot: E (Expansionist).** Claims denominator=1 "because prespecified family-level test" while simultaneously adding tensor decomposition across 5+ axes (session × sub-window × DoW × vol regime × macro × gap). The ancillary-vs-gate split is the exact garden-of-forking-paths Rule 1 exists to prevent — whichever sub-partition pops becomes the "real" finding and the gate gets rewritten post-hoc. Also uncritically uses Probe 4's 3.45× as H1 calibration (circular, as B flags). The P(PASS|H1)≈0.88 and BF≈145× numbers are theater built on that circularity.

**What all five missed:** None priced the **sample-size ceiling** on the confirmation partition. 1h test is 2024-10-22 → 2026-04-08 ≈ 1.5 years. After restricting to overnight-only and requiring a session × sub-window split, per-cell trade counts across a 20-50 combo basin will fall below the n≥50 floor Probes 2-4 already established as load-bearing. B1's statistical power is bounded by wall-clock, not method — and none of A/B/C/D/E checked whether the test partition physically contains enough overnight bars to resolve the sub-session hypothesis at the gate thresholds they're proposing. Partition exhaustion is temporal, not just epistemic.

---

### Reviewer 3

**1. Strongest: Response B (Contrarian).**

B is the only response that correctly prices partition exhaustion as a first-class constraint rather than an accounting detail. A/C/D and E all treat the 1h test partition as if it retains inferential purity after three consumption passes (Probes 2, 3, 4). B names this as the Probe 3 council error reprised and proposes the only mechanically honest kill — an independent vendor partition (Polygon/Databento). D gets closest by questioning whether B1 produces decision-relevant information at all, but still defaults to "D as footnote." E's BF=145× claim is false precision: the power calculation is conditioned on effects estimated from the same partition being tested.

**2. Biggest blind spot: Response E (Expansionist).**

E delivers the most operational-looking prereg (tensor decomposition, Welch-t ≥ 2.5, BF=145×) but commits the worst Rule 1 violation: treating B1 as a single-family confirmation with denom=1. The Probe 4 ~3.45× overnight ratio IS the training signal for E's H1 — computing P(PASS|H1) against an effect size estimated from the partition you're about to test is circular. The tensor axes (session × sub-window × DOW × vol × macro × gap-Sunday) multiply researcher DOF by ~100× while E claims denom=1.

**3. What all five missed:**

None addressed whether combo-865/1298/664 share a common *entry-parameter structure* making them one realization rather than three. Probe 4 found both new combos ~80% SES_2-concentrated — this is likely a selection artifact of the original parameter-sweep geometry (stop-distance, z-threshold, EMA window jointly favoring overnight fills), not three independent confirmations. Effective n may be 1, not 3, collapsing every scope's power calculation. Also missed: per-contract friction is session-invariant, but bid-ask widens overnight — the net-$5 RT assumption may itself fail in SES_2.

---

### Reviewer 4

**1. Strongest: Response B (Contrarian).**

B is the only response that names the actual statistical crisis: the 1h test partition has now been touched in Probes 2, 3, and 4, and each subsequent "confirmation" inherits selection from the previous. A, C, D, and E all propose *more* uses of that partition while calling it rigor — B correctly identifies this as the Probe 3 council's multiplicity error being reprised. Its kill criterion (independent vendor replication) is the only proposal that materially breaks the researcher-DOF loop rather than laundering it through cleaner-sounding partitions.

**2. Biggest blind spot: Response E (Expansionist).**

E's headline claim — "denom = 1 single prespecified family-level test" — is false. The tensor decomposition (session × sub-window × DoW × vol × macro × post-gap Sunday) is exactly the garden of forking paths that inflates the denominator. Declaring a single primary gate after enumerating a 6-way tensor doesn't make it one test; it makes it one *selected* test. The BF ≈ 145× figure is therefore ceremonial. E has also imported the ~3.45× ratio from Probe 3 as H1, which is circular (B catches this; E doesn't).

**3. What all five missed.**

None addressed that Probe 4's SESSION_CONFOUND verdict itself may be the ceiling, not a springboard. The confound finding means the *family-level* falsification (Probe 1 Branch A) and the *carve-out* (Probes 2-4) are now plausibly the same phenomenon — overnight-thin-liquidity noise harvesting concentrated in two combos. B1 in any form risks characterizing an artifact. The council should have asked whether paper-trade (D's direction) or independent-data replication (B's direction) precedes any further partition spend — not which B1 variant.

---

### Reviewer 5

**1. Strongest: Response D (First Principles).**

D is the only response that refuses to answer the framed question and instead asks whether B1 is the right next step at all. Its argument — that constraint #5 means B1 produces a narrative asset, not a decision input, and the remaining value is deploy-or-stop — is load-bearing. Three probes deep into a single-carve-out pocket, further re-characterization has diminishing marginal information. D's reframe (minimum-ceremony paper-trade prereg with overnight sub-window binding, forward-walking stopping rules, per-trade net-$ sizing floor) is the only response that advances the project toward an actionable decision.

**2. Biggest blind spot: Response E (Expansionist).**

E claims P(PASS|H1) ≈ 0.88, P(PASS|H0) ≈ 0.006, BF ≈ 145× — but the H1 effect size (3.45×) is estimated from Probe 3/4 data on the same test partition. This is Response B's circularity point applied rigorously: the Bayes factor is inflated because H1 is calibrated to the data it's being tested against. E also ignores that discovery/confirmation split on the already-3×-consumed test partition doesn't restore partition discipline — it subdivides an already-exhausted resource.

**3. What all five missed.**

None addressed the **economic stopping rule**. Combo-865 Probe 3 delivered +$124k/yr on 220 trades; combos 1298/664 added Probe 4 evidence. At what point does additional characterization cost exceed expected information value for a paper-trade decision? No response computed VOI, specified a capital threshold, or asked whether the user's $50k account even needs a 3-combo basin versus single-combo paper-trade on 865. D gestured at this; none operationalized it. The council converged on methodology without pricing whether B1 is worth running at all.

---

## Chairman Synthesis

### Where the Council Agrees

Four of five advisors converge on one point: **the Expansionist's prereg is the most dangerous submission despite looking the most operational**. All five peer reviewers independently flagged it as the biggest blind spot. The convergent critique: claiming "denominator = 1, single prespecified family-level test" while enumerating a 5-6 axis tensor decomposition is exactly the Rule 1 garden-of-forking-paths violation that sank Probe 3's original council. The BF ≈ 145× is computed against an H1 effect size (3.45×) estimated from the Probe 3/4 data on the same 1h test partition the gate is about to run on — circular. Contrarian, Outsider, and three of five reviewers named this explicitly.

Second convergence: **Scope C is dead**. Executor, Outsider, and the implied votes of Contrarian/First Principles all rank it lowest. Portfolio aggregation conflates session-decomposition with covariance structure and adds researcher DOF without adding inferential signal.

Third convergence: **the 1h test partition is running out of inferential room**. Contrarian names it first-class; First Principles routes around it; Outsider calls A/C/D "fourth dip"; Reviewers 1-4 all endorse partition-exhaustion as the central constraint the framed question elides.

Fourth convergence: **SES_2's 16:00-18:00 tail is a category error**. Outsider flagged the settlement halt bundling; Executor's D proposal and Expansionist's sub-window axis both try to unbundle it. Any binding that keeps SES_2 monolithic is wrong.

### Where the Council Clashes

**Launch vs don't launch.** First Principles (backed by Reviewers 1, 2, 5) says B1 under any scope is re-characterization of a falsified family that cannot close the deployment loop under constraint #5 — route to paper-trade prereg on {865, 1298, 664} instead. Contrarian (backed by Reviewers 3, 4) says even paper-trade is downstream; independent-vendor replication (Polygon/Databento 1h) is the only move that breaks researcher-DOF. Expansionist and Executor take the question at face value and propose scopes. This is the real fork.

**Mechanism vs basin.** Outsider and Executor both endorse Scope D first, but for different reasons: Outsider wants to know whether the edge is pure overnight or a settlement-halt artifact (kill criterion); Executor wants D as a cheap 10-minute mechanistic screen before committing to the 4-5h Scope B. First Principles would accept D as a "footnote" but not a gate.

**Effective n.** Reviewer 3's catch — {865, 1298, 664} may be one parameter realization, not three — directly undercuts Expansionist's 30-combo basin assumption. If effective n = 1, every power calculation in the prereg is overstated.

### Blind Spots the Council Caught

- **Reviewer 1:** Opportunity cost of B1 compute vs forward paper-trade runway. Forward data is the only genuinely fresh partition — generated by the clock, not by researchers. Every week on B1 is a week of uncollected paper-trade data on a decision already half-made.
- **Reviewer 2:** Wall-clock sample-size ceiling. 1h test partition is ~1.5 years; after restricting to overnight-only and splitting by sub-window × combo, per-cell trade counts fall below the n ≥ 50 floor Probes 2-4 established. B1's power is physics-bounded, not method-bounded.
- **Reviewer 3:** {865, 1298, 664} may share common entry-parameter structure (stop-distance × z-threshold × EMA window jointly favoring overnight fills). Effective n may be 1. Also: $5 RT friction is session-invariant by assumption but overnight bid-ask widens — the cost model itself may fail in SES_2.
- **Reviewer 4:** Probe 4's SESSION_CONFOUND may be the ceiling, not a springboard. Family-level Probe 1 falsification and the carve-outs are plausibly the same phenomenon (thin-liquidity noise harvesting). B1 risks characterizing an artifact.
- **Reviewer 5:** No advisor computed VOI or specified a capital threshold. User's $50k account may not need a 3-combo basin — single-combo paper-trade on 865 may be the economically rational next step.

### The Recommendation

**Do not launch B1 in any of the four proposed scopes.** The combined weight of constraint #5 (passing doesn't authorize deployment), partition exhaustion (3× consumption), and Reviewers 2/3's independent ceiling findings (wall-clock power + shared-structure n=1) means B1 is narrative characterization whose information value is dominated by the two alternatives flagged by the council's strongest voices.

**What replaces B1:** a two-track sequence, in order.

1. **Scope D as a 10-minute mechanistic check only** (Executor's specs: modify `_probe4_readout.py` to add a 3rd session bucket splitting SES_2 into 16:00-18:00 and 18:00-09:30 ET on {865, 1298, 664}, test partition). This is cheap, resolves the SES_2 category error Outsider flagged, and its output binds the overnight sub-window for track 2. Call it what it is — a mechanism footnote, not a family-level confirmation. Stage = characterization. Denominator = 3 (combos) × 2 (sub-windows).
2. **Paper-trade preregistration on the combo(s) that survive D** (First Principles' direction, Reviewer 1's fresh-partition argument). Specs: forward-walking protocol with ex-ante stopping rules, per-trade net-$ sizing floor (not Sharpe — Reviewer 5's economic anchor), overnight sub-window bound by D's output, clock-generated data (the only genuinely unconsumed partition available).

**On Probe 4's 3.45× ratio:** it cannot be used to power-calibrate any gate run on the same 1h test partition — that is circular and all four reviewers caught it. It can be used as an a priori effect-size estimate for sizing the paper-trade forward protocol, because forward data is a different partition.

**On Contrarian's independent-vendor replication:** note it as a future option if paper-trade equivocates, but it's a $ and latency cost that paper-trade forward data addresses at zero marginal data-acquisition cost.

### The One Thing to Do First

Run Scope D as a 10-minute patch to `_probe4_readout.py` — split SES_2 into 16:00-18:00 (post-RTH/settlement halt) and 18:00-09:30 (GLOBEX overnight) on combos {865, 1298, 664} against the 1h test partition, and read out per-trade net $ and n_trades per cell. The answer to "is this overnight edge or settlement-halt artifact?" decides whether paper-trade prereg goes forward at all.

---

**Timestamp**: 2026-04-23 UTC
**Council fired on**: Probe 4 verdict (`c419391`) SESSION_CONFOUND branch → B1 preregistration scoping
**Advisor composition**: Contrarian, First Principles, Expansionist, Outsider, Executor
**Anonymization randomized**: A→Executor, B→Contrarian, C→Outsider, D→First Principles, E→Expansionist
**Peer reviewer count**: 5 (independent, blinded to advisor identity)
