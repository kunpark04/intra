# LLM Council Transcript — Phase 5 Redesign Fork

**Convened**: 2026-04-21 UTC
**Project**: intra (MNQ 1-minute mean-reversion backtest bot)
**Trigger**: Both V3 and V4 ML#2 stacks failed combo-agnostic ship-blocker audits; `global_combo_id` feature was per-combo memorization channel. Paper-trading halted. No ML#2 stack passes ship criteria.

---

## 1. Original Question (from user)

User invoked `/llm-council` after the Phase 3 V3 audit FAIL verdict was committed (5135706). Most recent open question to user before compaction:

> "Want me to extract the s3 unfiltered MC verdict from the already-executed notebook, or jump straight to Option A K-fold scaffolding?"

The council was convened on the broader Phase 5 decision fork rather than the narrow next-step question.

---

## 2. Framed Question (sent to all 5 advisors)

**Context — Phase 5 redesign fork after double audit FAIL**

Both V3 and V4 LightGBM ML#2 stacks in the 'intra' MNQ 1-minute mean-reversion backtest bot failed combo-agnostic ship-blocker audits with identical root cause: the `global_combo_id` LightGBM categorical feature was a per-combo memorization channel.

**Audit results** (s6_net Monte Carlo, identical Pool B top-50 combos, identical kernel, $5/contract RT friction, fixed_dollars_500 sizing):

| Metric | Shipped V4 | V4 no-ID | Shipped V3 | V3 no-ID |
|---|---|---|---|---|
| sharpe_p50 | +2.13 | −0.42 | +1.78 | +0.31 |
| ruin_prob | 0.02% | 56% | 6.9% | 54% |
| pos_prob | 99.6% | 30% | 98.7% | 64% |
| trades delta | — | +126% (rejector) | — | +18% (re-weighter) |

- OOF AUC preserved across refit (0.8293 → 0.8293) — AUC is trade-level ranking; leak lives in combo-level calibration, not ranking
- 100% combo overlap between training (588,235 combos) and Pool B top-50 — held-out-combo validation structurally infeasible

**Current state**: Both V3 and V4 production status REVOKED. Paper-trading halted. No ML#2 stack passes ship criteria.

**The 4 redesign options**:
- **Option A** — Combo-level K-fold retrain on V3 (~10h compute)
- **Option B** — Accept V3-no-ID Sharpe 0.31 as honest ceiling
- **Option C** — Abandon ML#2 entirely; 5-minute diagnostic on unfiltered Pool B already executed
- **Option D** — Learning-to-rank redesign (multi-day rebuild)

Ship criterion (s6_net): CI > 0, ruin < 10%, pos_prob > 90%. Currently 0 candidates pass.

---

## 3. Anonymization Mapping

| Letter | Advisor |
|---|---|
| A | The Expansionist |
| B | The Outsider |
| C | The Contrarian |
| D | The Executor |
| E | The First Principles Thinker |

(Mapping was hidden from the 5 peer reviewers; revealed to chairman.)

---

## 4. Advisor Responses (full, de-anonymized)

### 4.1 The Contrarian

Run Option C in the next 5 minutes. Non-negotiable.

Here's what you're avoiding: if unfiltered Pool B on V3-no-ID training data hits Sharpe ≥ 0.31, **the ML#2 layer has been ornamental the entire time**. Every week you spent on V1/V2/V3/V4 boosters, isotonic calibration, per-combo calibrators, Kelly variants, the whole Phase 5D postmortem — weeks of work whose apparent Sharpe uplift came from combo memorization. Selection from the top-50 raw-Sharpe basket (the real source of edge per the v12 ablation) may be doing 100% of the work already. You cannot rationally commit 10h of compute to Option A without this number.

Hidden failure mode in Option A: you're assuming the feature is salvageable. But combo-level K-fold with 588,235 combos and 100% overlap with Pool B means every fold's OOS combos are still statistically neighbors of training combos (same parameter neighborhoods, same sweep distribution). You'll likely get OOS Sharpe somewhere between 0.31 and 2.13 — an ambiguous result you'll have to interpret, and you'll be tempted to ship. AUC parity already fooled you once; don't let "K-fold matches unfolded" fool you again. Define the kill-criterion before you run it: what K-fold Sharpe number makes you abandon `global_combo_id` permanently? Write it down now.

Option D is a cope. You don't have a ranking problem — v12 raw-Sharpe top-50 already ranks fine (Sharpe 2.13 baseline). You have a **trade filtering** problem, and trade filtering may simply not be tractable on this data.

Order: C (tonight) → B (accept) OR A (with pre-committed kill-criterion). Never D.

### 4.2 The First Principles Thinker

You're asking which redesign path to pick. That's the wrong question.

The actual problem: **you have no evidence that a predictable edge exists at the trade-grain on this strategy family.** V3 and V4 both "worked" because they memorized combo identity. When you remove the leak, trade-level AUC stays at 0.83 but the Sharpe collapses from +1.78 to +0.31. That's not a calibration bug. That's the signal telling you the win/loss distribution conditional on features-minus-combo-ID is nearly indistinguishable from the unconditional base rate *at the per-trade level*.

AUC 0.83 with Sharpe 0.31 means the ranking is real but the economic magnitude per ranked unit is trivial once you can't cheat with per-combo priors. A filter that admits +18% more trades and drops Sharpe 5.7x is a filter doing almost nothing useful.

So reframe. The real question is: **where does the edge actually live?** The Pool B top-50 raw-Sharpe portfolio (no ML#2) already delivers Sharpe 2.13. The edge is in **combo selection**, not trade selection. Every week you've spent on ML#2 has been trying to squeeze juice from a layer that, stripped of leakage, contributes ~nothing.

Prioritization:
1. **Option C first, today, 5 minutes.** This is the only cheap falsifiable experiment. If unfiltered Pool B ≥ filtered, ML#2 is dead and you've saved yourself Option A's 10 hours.
2. Only if C shows a real gap: run A. K-fold isn't a redesign, it's a validation of whether the leak *was* the edge or merely rode alongside one.
3. B and D are premature. Don't accept a ceiling or re-architect before you've measured whether the layer earns its existence.

Stop defending the pipeline. Measure it.

### 4.3 The Expansionist

The leak isn't a catastrophe. It's a **free labeled dataset on what combo-memorization is worth** — and you're about to throw it away.

Shipped V4 Sharpe +2.13 minus combo-agnostic −0.42 = **a clean 2.55 Sharpe delta that is pure combo-identity value**. That number is gold. It tells you the market for "knowing which combo you're trading" is massive. The question isn't "how do I remove the leak" — it's **"how do I legitimately productize combo identity?"**

**Option E the council is missing: Combo-specific meta-models.** Don't predict P(win | features) across all 13,814 combos with one global model. Train a **lightweight per-combo calibrator** on each top-50 combo's own historical trades. This is not leakage — it's the entire premise of production trading. Every prop desk runs strategy-specific models. You have 500+ trades per combo post-gate; that's enough for a logistic regression or shallow GBM per combo. The "combo overlap is 100%" problem becomes a **feature**, not a bug — you're not trying to generalize to unseen combos, you only ever trade the 50.

**Option F — learn-to-rank is the adjacent upside.** You're currently raw-Sharpe-ranking 13,814 combos and it beats ML#1 v12. That's a scream that the ranking layer is where value lives, not the trade-level filter. A proper LambdaRank on combo features predicting robust walk-forward Sharpe could **lift the entire top-50 pool quality**, which compounds more than any ML#2 filter ever will.

**Biggest hidden win**: You've been optimizing the wrong layer. ML#2 filters 20% of trades; ML#1/ranking chooses the 50 combos you trade at all. A 0.1 Sharpe improvement at the ranking layer beats a 0.3 Sharpe improvement at the filter layer because it multiplies across every trade. **Go up the stack.**

### 4.4 The Outsider

From the outside, this looks less like a modeling problem and more like a story about a person who built something that felt like it was working, discovered it wasn't, and is now asking "which tool should I use to fix the thing?" The tool question is downstream of a much bigger one.

Here's what jumps out: the gap between "Sharpe 2.13" and "Sharpe −0.42" isn't a bug. It's a measurement of how much you were fooling yourself. That's not a small delta. That's the model telling you it had no predictive power on the thing you actually care about. A domain expert sees "we found the leak, now we fix it." An outsider sees "you spent weeks not knowing your model didn't work, and your validation process didn't catch it until now."

The question you're not asking: **why did it take weeks to find this?** What property of your workflow let a memorization channel survive all the way to "ship"? Whatever answer you give for Option A, B, C, or D — if you don't fix that, the next stack will have a different leak, and you'll find it in another several weeks.

Second thing: "user has invested weeks" shows up in your framing. That's sunk cost talking. The four options are ordered in a way that protects the investment (A = keep going, B = accept a tiny win, C = admit defeat, D = start over). An outsider reads C as the cheapest and most honest: spend 5 minutes checking whether the filter adds anything over no filter. If it doesn't — and the fact that you're asking suggests you already suspect — then A and D are elaborate ways of avoiding that answer.

Run C first. Today. Before choosing anything.

### 4.5 The Executor

**Monday morning, literal first command:**

```bash
cd C:\Users\kunpa\Downloads\Projects\intra && scp runner:evaluation/v12_topk_net_v3_no_gcid/s3_mc_combined.ipynb ./evaluation/v12_topk_net_v3_no_gcid/ && jupyter nbconvert --to script s3_mc_combined.ipynb | grep -A2 "sharpe_p50\|ruin_pct"
```

That's **Option C's diagnostic pull**. 5 minutes. Before any redesign, you need the unfiltered Pool B number. If s3_net (no ML#2 at all) already beats V3-no-ID's 0.31 Sharpe, ML#2 is net-negative and Options A/D are dead on arrival.

**Sequencing:**
1. **Mon AM (5 min)**: Pull s3. Decide if ML#2 is even additive.
2. **Mon PM (if s3 < 0.31)**: Launch Option A. Literal command:
   ```bash
   ssh runner "cd intra && git pull && screen -dmS kfold_v3 bash -c 'systemd-run --scope -p MemoryMax=9G python scripts/models/adaptive_rr_model_v3.py --kfold-by-combo --n-folds 5 --seed 42 2>&1 | tee logs/kfold_v3.log'"
   ```
   Then attach a 10-min Monitor poll.
3. **Tue EOD**: Read OOF Sharpe on held-out fold. Ship-or-kill.

**Hidden ambiguity — Option B is not an action.** "Accept honest ceiling and pivot to combo-scouring" has no Monday command. What script? What sweep? What ranker? It's a mood, not a task. If the user picks B, force them to name the literal next `.py` file they'll write before EOD Monday, or B collapses into procrastination.

**Option D is also ambiguous**: "learning-to-rank" — LambdaMART? XGBoost rank:pairwise? Which library, which target, which eval metric? 2 days of scoping before a single line of code.

A and C have real commands. B and D don't. Run C today, A tomorrow.

---

## 5. Peer Reviews (anonymized; each reviewer saw all 5 responses as A-E with mapping hidden)

### Reviewer 1

1. **Strongest**: E (First Principles). Nails the diagnosis that AUC 0.83 with Sharpe 0.31 means the ranking is real but the economic magnitude per trade is trivial without combo priors. Reframes from "which redesign" to "does the layer earn its existence" and gives a concrete falsifiable sequence.
2. **Biggest blind spot**: A (Expansionist). "Per-combo meta-models" is exactly what the V3/V4 audit just disproved. With 100% combo overlap and `global_combo_id` memorization as the failure mode, training per-combo calibrators on the same 50 combos is the same leak wearing a costume. 500 trades/combo is not "plenty" given walk-forward needs. Smart-sounding, operationally dangerous.
3. **All missed**: Whether s3_net (the unfiltered diagnostic in C) might itself be contaminated. The 13,814-combo pool was ranked by `audit_full_net_sharpe` computed on the *full* history including Pool B's test window. If raw-Sharpe top-50 selection is leaky, Option C's "unfiltered ≥ 0.31" result still isn't clean — could mean selection leaks, not that ML#2 is overhead. Council skipped auditing the ranker sitting above ML#2.

### Reviewer 2

1. **Strongest**: E (First Principles). Correctly reframes: AUC 0.83 + Sharpe 0.31 is a structural diagnosis, not a bug report. E's sequencing (C first as falsification, A only conditional on C showing a gap) is the only framing that treats ML#2 as a hypothesis rather than an asset to rescue.
2. **Biggest blind spot**: A (Expansionist). Per-combo calibrators were already deprecated in Phase 5D (91% DD in portfolio sim, failed B16 gate per CLAUDE.md). A proposes reopening a stack that's already been killed by the same workflow. A also ignores that 100% combo overlap makes per-combo meta-models indistinguishable from the `global_combo_id` leak just revoked.
3. **All missed**: The ship-gate failure mechanism itself. Two consecutive stacks shipped on AUC parity while Sharpe was pure memorization. The process question — what audit becomes mandatory *before* any future ML#2 ships (combo-agnostic refit as a pre-ship gate, not post-hoc) — matters more than path A/B/C/D. Without it, Phase 6 will leak differently and take weeks to catch again.

### Reviewer 3

1. **Strongest**: C (Contrarian). The only one that names a kill-criterion problem explicitly ("AUC parity already fooled you once; don't let K-fold-matches-unfolded fool you again"). Sequences C → {B or A}, rejects D as cope, and anticipates Option A's failure mode (neighbor-combo contamination producing ambiguous Sharpe between 0.31 and 2.13). D is tactically sharp but C reasons about decision theory, not just Monday commands.
2. **Biggest blind spot**: A (Expansionist). "Combo-specific meta-models" sounds clever but collides with the 100%-overlap finding. Per-combo calibrators on 500+ trades is exactly the per-combo memorization channel that just got revoked — dressed up as "prop-desk practice." A also waves at learning-to-rank as upside without acknowledging v12 UCB already failed ablation (`project_ml1_v12_ablation_failure`). Confidence without reading the room.
3. **All missed**: Whether **Pool B itself is overfit to the V3/V4 training window.** 100% combo overlap means the top-50 raw-Sharpe pool was selected *on the same data V3/V4 trained on*. If Option C shows unfiltered Sharpe ≥ 0.31, the honest next question isn't "ML#2 is ornamental" — it's "is Pool B's 2.13 also per-combo memorization at the ranker layer?" That's the real ship-blocker audit nobody proposed.

### Reviewer 4

1. **Strongest**: C (Contrarian). Names the hidden failure mode no one else catches — that combo-level K-fold with 100% overlap still leaks via neighbor-similarity in parameter space. Demands a pre-registered kill-criterion (the exact discipline that let AUC parity fool the team for weeks), and correctly sequences C before A. E reaches the same conclusion but stops at diagnosis; C gives an operational guardrail.
2. **Biggest blind spot**: A (Expansionist). "100% overlap becomes a feature" directly contradicts the measured evidence: shipped V4's per-combo signal *was* per-combo memorization, and it collapsed to ruin 56% under a clean audit. Proposing per-combo meta-models is the same leak wearing a different hat — training on each combo's own history is combo-ID memorization formalized.
3. **All missed**: None questioned whether the **0.31 Sharpe itself survives its own combo-agnostic audit**. V3-no-ID was produced on Pool B combos that 100% overlap training. If C shows unfiltered ≥ 0.31, the honest baseline may itself be inflated — the ship floor is unverified. Also missed: paper trading is halted; there's no "do nothing" option priced in.

### Reviewer 5

1. **Strongest**: C (Contrarian). Names the actual failure mode lurking in Option A — combo-level K-fold with 100% overlap leaves OOS combos as neighbors of training combos, reproducing an AUC-parity-style false pass. Demands a pre-declared kill criterion. The only response that closes the loop that already burned the project twice.
2. **Biggest blind spot**: A (Expansionist). "Per-combo meta-models on 500 trades" is exactly the memorization channel that just got revoked — dressed in prop-desk language. 500 trades per combo after a top-50 gate selected on the same universe is not independent signal; it's the same leak reshaped. Also missing: no falsification criterion, no acknowledgment that the top-50 itself came from a leaky ranker.
3. **All missed**: None challenged the **s6_net Sharpe target** as the ship gate. The V3/V4 revocations show Sharpe point-estimates with tight CIs still mis-called ship twice. Before re-running C/A, the council should have asked what *audit design* would have caught the `global_combo_id` leak ex-ante — e.g. mandatory combo-held-out slice, feature-ablation deltas, or permutation tests on combo identity — and whether the 100% overlap finding makes *any* in-sample ship decision on this strategy family structurally unsafe without a true temporal/universe holdout.

### Peer review tally

| Advisor | Strongest votes | Biggest blind-spot votes |
|---|---|---|
| The Contrarian | 3 | 0 |
| The First Principles Thinker | 2 | 0 |
| The Expansionist | 0 | **5** (unanimous) |
| The Outsider | 0 | 0 |
| The Executor | 0 | 0 |

### Universal "what all missed"

4 of 5 reviewers independently raised the **Pool B ranker leak** concern: if `audit_full_net_sharpe` top-50 selection itself is leaked, Option C's result is uninterpretable. Additional missed points: the ship-gate failure mechanism (no pre-ship combo-agnostic audit mandated), the 0.31 baseline itself is unaudited, "do nothing" not priced.

---

## 6. Chairman's Synthesis

### Where the Council Agrees

- **Option C first, today, 5 minutes.** All five advisors converge.
- **Option D is disqualified.** 4 of 5 explicitly reject.
- **The ML#2 layer is on trial, not under repair.** AUC 0.83 + Sharpe 0.31 = ranking real, economic magnitude trivial once combo priors gone.
- **Per-combo meta-models are the revoked leak in a new costume.** Unanimous peer review rejection.

### Where the Council Clashes

- "If C passes, what next?" — Contrarian/Outsider/First-Principles: accept. Expansionist: go up the stack. Executor: launch A Monday PM if C < 0.31.
- Does Option A survive the 100%-overlap critique? Contrarian + 4/5 reviewers say A produces structurally ambiguous results. Executor still wants A as fallback. Real unresolved disagreement.
- Expansionist vs everyone else on "combo identity is a feature." Measured evidence: combo-identity signal → 56% ruin when forced to generalize.

### Blind Spots the Council Caught

- **Pool B ranker layer itself may be leaky** (4 of 5 reviewers independently).
- **Ship gate failed twice and no one priced it** — pre-ship combo-agnostic audit as mandatory gate is missing.
- **0.31 baseline is unaudited** — Pool B 100% overlap means even the ceiling is suspect.
- **"Do nothing" not priced** — capital idle, conviction decay, halted paper-trading.
- **Outsider alone**: what workflow property let the leak survive to ship?

### The Recommendation

**Run Option C today. Pre-register a kill-criterion before you read the result. Then, regardless of outcome, run a ranker-layer combo-agnostic audit before any further ML#2 work.**

1. **Today (5 min):** Pre-commit kill-criterion in writing. Run C.
2. **Tomorrow (independent of C's result):** Audit the Pool B ranker. Does `audit_full_net_sharpe` top-50 survive recomputation on a pre-2024 / walk-forward slice excluding V3/V4 training window?
3. **Only after both audits:** Decide A vs B.
4. **Reject Option D and per-combo meta-models.** Both are sophisticated-sounding versions of the revoked leak.

Disagreeing with Executor: Option A should *not* auto-launch Monday PM if C < 0.31. The ranker audit is upstream of A's interpretability.

Disagreeing with Expansionist entirely: Per-combo meta-models are the revoked leak in a new costume.

### The One Thing to Do First

Before opening a terminal, open `tasks/phase5_kill_criterion.md` and write this line, dated and timestamped:

> *"If unfiltered Pool B s6_net Sharpe p50 ≥ 0.31 with ruin ≤ V3-no-ID's 54%, I will deprecate ML#2 filtering from the production pipeline and will not train another trade-grain filter without first auditing the ranker layer for per-combo memorization. Signed, 2026-04-21."*

Then, and only then, run Executor's literal command:

```bash
scp sweep-runner-1:evaluation/v12_topk_top50_raw_sharpe_net_v3_no_gcid/s3_mc_combined_net.ipynb . \
  && jupyter nbconvert --to script s3_mc_combined_net.ipynb \
  && grep -A2 "sharpe_p50\|ruin_pct" s3_mc_combined_net.py
```

The criterion has to be written down before the number is read. That is the one discipline that would have prevented both revocations, and it costs nothing.

---

## 7. Artifacts

- HTML report: `tasks/council-report-2026-04-21-phase5.html`
- This transcript: `tasks/council-transcript-2026-04-21-phase5.md`
- Related verdict docs: `tasks/v4_no_gcid_audit_verdict.md`, `tasks/v3_no_gcid_audit_verdict.md`, `tasks/ship_decision.md`

## 8. Follow-up tasks (queued)

- #15 Pre-register Phase 5 kill-criterion before reading C's result
- #16 Run Option C diagnostic — unfiltered Pool B s3 MC extraction
- #17 Audit Pool B ranker layer for combo-agnostic soundness
