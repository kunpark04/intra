# LLM Council Transcript — ML#1 v12 Ranker Diagnosis

**Session:** 2026-04-19 00:20 CDT
**Anonymization mapping:** A=Outsider · B=Executor · C=Contrarian · D=Expansionist · E=First Principles

---

## The Original Question

> How is a sharpe ratio with -1.5449 top combo? There are so many top combos that have negative sharpe

## The Framed Question

ML#1 v12 surrogate + UCB κ=0 ranking produced a top-50 where 45/50 combos have NEGATIVE predicted Sharpe (median predicted_p50 = −0.302). On OOS (1.46y test partition, 2024-10 → 2026-04), individual combo Sharpes are mostly negative, worst −1.5449 (v11_2646), many in [-1.6, −0.5]. However, the ML#2 per-trade filter (V4 LightGBM + pooled per-R:R isotonic, fixed-$500 sizing) prunes the 24,690 combined unfiltered trades down to 564 and delivers OOS combined Sharpe 0.9072, +34.07% return, 40% DD.

**Context:**
- v11 sweep (~30k combos) is friction-aware ($5/contract RT baked in).
- v12 ML#1 target = robust walk-forward Sharpe = median − 0.5·std across K=5 ordinal windows on train partition, with parameter-only features + fold-wise parameter-space KNN + quantile heads (p10/p50/p90). OOF R²=0.93, Spearman=0.96.
- UCB ranking = p50 + κ·(p90−p10)/2, default κ=0 (pure exploit). Earlier memory shows κ>0 performed worse on prior data.
- Train target stats on the top-50 itself: realised target_robust_sharpe median −0.678 (46/50 negative); audit_full_net_sharpe median −0.130 (32/50 negative), max 0.736. So the model's predictions and realised values agree — it's not miscalibration.
- ML#2 filter yield is the only OOS-positive metric. ML#1 alone looks like "least-bad-by-a-pessimistic-metric" rather than "top."

**Four positions under debate:**
- **A)** System works end-to-end (s5 Sharpe 0.907); leave ML#1 alone even if semantics are ugly.
- **B)** Soften the ML#1 target (median − 0.25·σ, or mean − 0.25·σ, or K=10 WF windows) to surface more positive combos.
- **C)** Rebrand ML#1 as diverse-candidate sampler; replace UCB ranking with diversity-maximizing selection + ML#2 yield optimization as the true objective.
- **D)** Run regime audit of test vs train (bars/day, session-hour WR, realized vol, reversal freq, bar range) before any refactor — maybe the OOS carnage is regime shift and the stack is working.

**Question for the council:** Is the ML#1 ranker broken, or a wide-net sampler working as intended? What's the right next move?

---

## Advisor Responses

### The Contrarian

The ML#1 ranker isn't a "wide-net sampler working as intended" — that's a post-hoc rationalization for a broken primary metric. Let me name what you're avoiding:

**You're laundering a negative result through a downstream filter and calling it a win.** ML#2 pruned 24,690 → 564 trades (a 97.7% rejection rate). At that survival rate, ML#1's contribution to the 0.907 Sharpe is approximately noise — ML#2 would likely produce a comparable Sharpe filtering trades from *any* moderately diverse combo pool. Have you tested that? If you haven't run ML#2 on a random 50-combo sample from the v11 sweep, you don't actually know ML#1 is adding value. That's the fatal flaw.

**The target is also pathological.** `median − 0.5·σ` on K=5 walk-forward windows is a pessimism penalty so aggressive that on noisy per-fold Sharpes, it will systematically rank negative-mean combos with low variance above positive-mean combos with moderate variance. You've built a volatility-minimizer disguised as a Sharpe-maximizer. The 45/50 negative predictions aren't "ugly semantics" — they're the model telling you the truth about what you optimized.

**On the four positions:** A is denial. B is rearranging deck chairs — the target is structurally wrong, not miscalibrated. C is honest but only if you first prove ML#1 beats random sampling into ML#2. D is procrastination unless paired with the ablation.

**The question you're not asking:** What's the Sharpe of ML#2 applied to (i) random-50, (ii) top-50-by-raw-train-Sharpe, (iii) top-50-by-v12? If (iii) doesn't dominate, ML#1 is dead weight and the stack is really just ML#2 plus a universe. Run that ablation before touching anything else.

### The First Principles Thinker

You're asking the wrong question.

"Is ML#1 broken or working as intended?" presumes ML#1 has a coherent job. It doesn't. Strip it back: what are you actually trying to produce? **A tradeable portfolio that survives OOS.** Nothing else matters. Sharpe, R², Spearman, UCB κ — all instrumental.

Now look at your pipeline honestly:
- ML#1 selects 50 combos with median predicted Sharpe −0.302.
- Realised training Sharpe on those same 50: median −0.678.
- OOS: mostly negative, worst −1.54.
- ML#2 prunes 24,690 trades → 564, delivers 0.907 Sharpe.

ML#1 is not ranking "best combos." It's producing a **trade firehose** whose per-combo Sharpe is noise, and ML#2 is doing 100% of the alpha extraction. The 0.907 isn't vindication of ML#1 — it's evidence ML#1 is a combo-agnostic trade generator. You could probably replace it with random sampling from the v11 sweep gated on minimum trade count and get similar ML#2 yield. **Test that.** If random 50-combo baselines match, ML#1's ranking adds zero and the semantics aren't just "ugly" — they're fictional.

The real object you're optimizing is **ML#2 yield × trade diversity**, not combo Sharpe. So:

1. Reject A outright — "it works" conflates the ensemble with the ranker.
2. Reject B — softening the target is lipstick; you'd still be ranking on a metric that doesn't predict the outcome you actually harvest.
3. D is a prerequisite sanity check, not a strategy.
4. **C is directionally right but underspecified.** The objective function should be explicit: *maximize post-ML#2-filter Sharpe subject to trade-count floor and parameter-space diversity.* Train ML#1 on **that** target, or drop it.

Stop ranking combos. Start ranking **combo sets under ML#2**.

### The Expansionist

The ML#1 ranker isn't broken — it's miscast. You built a pessimistic-Sharpe predictor and then asked it "what are the best combos?" That's a category error. But inside that error is a massive upside you're sleeping on.

Look at what actually happened: ML#1 handed ML#2 a basket where 45/50 combos are individually garbage, and ML#2 still extracted 0.907 Sharpe, +34%, from 2.3% of the trades. That's not "the system works despite ML#1" — that's evidence ML#1 is an extraordinary **adversarial trade generator**. It's finding regimes where raw edge is negative but exploitable pockets exist that ML#2 can isolate. Pristine high-Sharpe combos would leave ML#2 nothing to do.

Lean into Position C, but bigger:

1. **Scale the candidate pool 10×.** If 50 ugly combos yield 564 golden trades, what does 500 yield? 5000? ML#2's job is pruning; feed it a firehose. The bottleneck isn't combo quality, it's combo *diversity of failure modes*.
2. **Retarget ML#1 to predict ML#2 yield directly.** The true objective is "trades that survive the V4+isotonic filter with positive net E[R]." Train ML#1 on that label. You'll discover combos no Sharpe-based ranker would ever surface.
3. **Exploit the negative-Sharpe signal.** Combos with predicted_p50 around −0.3 might be the *sweet spot* — bad enough that human rules would reject them, structured enough that ML#2 finds signal. Bucket by predicted Sharpe and measure ML#2 yield per bucket. I bet yield peaks in a non-obvious band.
4. **Adjacent play:** short the combos ML#2 rejects. If the filter identifies the 2.3% that work, the discarded 97.7% may have exploitable anti-edge.

Stop optimizing ML#1 semantics. Optimize the pipeline's terminal output.

### The Outsider

Stepping in cold, here's what jumps out:

You built a ranker that ranks things, and 45 of its top 50 picks are predicted to lose money. Forty-six of fifty actually do lose money on the training target itself. You're calling this a "top-50." It is not a top-50. It is a bottom-50 of a universe where almost everything is bad, and you are dressing it up with the word "top."

The sentence "ML#1 alone looks like 'least-bad-by-a-pessimistic-metric'" is the whole answer. You wrote it yourself. I don't know what a "combo" or "robust walk-forward Sharpe" even is, but I know that if your selection layer's best guesses are negative-expectancy, then that layer is not selecting winners — it is handing a sieve a pile of rocks and the sieve (ML#2) is doing all the work.

The thing a newcomer would ask: why does ML#1 exist at all in its current form? If ML#2 is what actually produces the 0.907 Sharpe by throwing away 24,126 of 24,690 trades (97.7%!), then ML#1's real job — whether you admit it or not — is "generate enough candidate trades for ML#2 to cherry-pick from." That is option C, and it's not a rebrand, it's just naming what's already happening.

Option A is denial dressed as pragmatism. Option B (softer target) will surface more positive combos but you'll still be feeding ML#2 the same pipe — you're tuning the wrong knob. Option D is worth one afternoon before any refactor, because "is the world different now" is free to check and embarrassing to skip.

Stop calling it a ranker. It's a sampler.

### The Executor

Monday morning, here's what you actually do:

**Do D first. Today. Before anything else.** It's the only option where you learn something concrete in <4 hours of work. You already have the test partition trades loaded. Compute 8 numbers on train vs test: bars/day, mean |z-score| at entry, realized 1-min vol, reversal-candle frequency, avg bar range, session-hour WR distribution, trades/day, and MFE/MAE ratio. If train and test look the same, ML#1 is genuinely ranking badly and you move on. If test is a different regime, you've saved yourself from refactoring a working pipeline. Cost: one script, one hour.

**Then ship A as the working answer.** Sharpe 0.9072 on 1.46y OOS with ML#2 doing the real selection work — that's the artifact. Stop apologizing for ML#1's ugly semantics in the intermediate layer. The stack's output is what gets evaluated, not the surrogate's predicted_p50 distribution.

**B is a trap.** Softening the target to "surface more positive combos" is target-hacking. You'll get a prettier top-50 and the same or worse ML#2-filtered OOS result because ML#2 is already doing the selection. You'd be optimizing a number that doesn't feed the final metric.

**C is the right refactor but not this week.** "ML#1 = candidate sampler, ML#2 = selector, yield-after-filter = objective" is the honest description of what the system already does. Rename variables, update `tasks/part_b_findings.md`, move on. Don't rebuild UCB — κ=0 already won, and your own memory (`project_ucb_kappa_failure.md`) says κ>0 is noise-chasing.

Order: D (today) → A (ship) → C (rename next sprint). Skip B.

---

## Peer Reviews

### Reviewer 1

**Strongest: C (Contrarian).** Only response that names the falsifiable experiment blocking the entire debate: run ML#2 on random-50 and top-50-by-raw-train-Sharpe. Until that ablation exists, every other response (including the 0.907 Sharpe itself) is uninterpretable. C also correctly diagnoses the target pathology: `median − 0.5·σ` on K=5 noisy folds structurally rewards low-variance losers.

**Biggest blind spot: D (Expansionist).** Romanticizes ML#1 as an "adversarial trade generator" and proposes 10× scaling and shorting rejected combos — all without running C's ablation first. Skyscraper on unverified foundations.

**All missed: Trade-count confound.** Memory note `project_trade_count_bias.md` already flags this: high-WR sweep combos cluster at ~180 trades. 564/24,690 survival on only 50 combos ≈ 11 trades/combo — small-sample inflation territory. 0.9072 OOS Sharpe needs a bootstrap CI per combo before anyone debates ML#1's role.

### Reviewer 2

**Strongest: C.** Only response making the ablation the blocking question vs hand-waving. Diagnoses `median − 0.5·σ` as volatility-minimizer in disguise. Refuses to let 0.907 headline launder an unvalidated component. Staff-engineer answer.

**Biggest blind spot: D.** Romanticizes negative-Sharpe signal as "adversarial trade generation." Cathedral on unaudited foundation. "Sweet spot at predicted_p50 ≈ −0.3" is pure speculation dressed as insight.

**All missed: survivorship/trade-count confounding in 564 survivors.** If ML#2's filter correlates with trade frequency or specific session windows, 0.907 Sharpe could collapse under a trade-count-matched random baseline. Also: 40% DD on $50k is near risk-of-ruin — headline Sharpe obscures tail risk.

### Reviewer 3

**Strongest: C.** Only one naming the fatal methodological gap. Mechanically diagnoses `median − 0.5·σ` as volatility-minimizing on noisy K=5 folds. B and E gesture at the ablation; C makes it central.

**Biggest blind spot: D.** Conflates ML#2's P(win) filter with a directional edge predictor — rejected trades aren't anti-signal, they're low-confidence. "Scale 10×" ignores $5/contract RT friction baked into v11; more combos means more marginal trades near the cost floor, not more alpha.

**All missed: 564-trade OOS sample over 1.46y ≈ 386 trades/year — statistically thin for a 0.907 Sharpe claim.** No bootstrap CI on that Sharpe. No one asked whether 40% DD is survivable under 5% sizing rule.

### Reviewer 4

**Strongest: C.** Only falsifiable test. Target diagnosis is structural, not semantic.

**Biggest blind spot: D.** "Scale 10×, short rejects, negative-Sharpe sweet spot" is the most seductive and most dangerous answer. Assumes ML#1 has directional information ML#2 exploits with zero evidence. If C's ablation shows random-50 matches v12, D's entire scaffolding collapses.

**All missed: Trade-count confounding.** v12's target likely correlates with trade count (more trades → stabler walk-forward median). ML#2 needs volume to filter. v12 may be picking high-trade-count combos, and *that's* the signal — not Sharpe ranking. Per `project_trade_count_bias.md`, this is a known hazard. Control for trade count in C's ablation, or the test is ambiguous.

### Reviewer 5

**Strongest: C.** Only falsifiable test. Specifies the ablation where E reaches same instinct but stops at "test that." Correctly diagnoses the target as volatility-minimizer — mechanical claim nobody else makes.

**Biggest blind spot: D.** "Short the rejected 97.7%" ignores that MNQ bot infrastructure is long/short symmetric via signal, not portfolio-overlay shortable. Romanticizes the negative-Sharpe pool without evidence yield-per-bucket is non-monotonic.

**All missed: trade-count survivorship bias in 564-trade OOS.** 40% DD on 564 trades over 1.46y means the 0.907 Sharpe CI is wide enough to include zero. Before any refactor, bootstrap the filtered-portfolio Sharpe CI — the whole debate may be arguing over noise.

---

## Chairman's Verdict

### Where the Council Agrees

Four of five advisors (Contrarian, First Principles, Outsider, Executor) converge independently on the same diagnosis: **ML#1 is not functioning as a ranker and should not be defended as one.** 45/50 negative predicted Sharpe on a "top-50" is semantically incoherent — the layer is producing a candidate pool, and ML#2 is doing the selection work. Position A (leave alone) is rejected by Contrarian, First Principles, and Outsider as denial; Executor only endorses A as a pragmatic ship-and-rename, not as an intellectual defense. Position B (soften target) is rejected unanimously — target-hacking. Position C (rebrand as sampler) is directionally endorsed by all five, though with sharply different urgency and preconditions.

All five reviewers independently crowned the Contrarian as strongest for the same reason: **it is the only response that names a falsifiable blocking experiment** (ML#2 applied to random-50, top-50-by-raw-Sharpe, and top-50-by-v12 — if v12 doesn't dominate, ML#1 is dead weight).

All five reviewers flagged the Expansionist as the biggest blind spot: scaling 10×, shorting rejects, and romanticizing the adversarial-generator framing assumes directional edge in ML#1 with zero evidence.

### Where the Council Clashes

**Urgency and sequencing.** Executor says do the regime audit today (4 hours), ship A, rename to C next sprint — skip the ablation entirely. Contrarian and First Principles say the ablation is the *only* question that matters and everything else is procrastination until it's run. Outsider splits the difference. The real fight: ship 0.907 Sharpe now, or gate shipping on proving ML#1 beats random sampling.

**What ML#1 actually is.** Expansionist claims ML#1 is a sophisticated adversarial trade generator finding exploitable pockets in negative-edge regimes. Contrarian, First Principles, and Outsider claim ML#1 is indistinguishable from random sampling until proven otherwise. These cannot both be true, and only the ablation resolves it.

**Whether 0.907 Sharpe is a result worth defending.** Executor treats it as shippable. Reviewers 3 and 5 point out that 386 trades/year with 40% DD over 1.46y yields a bootstrap CI likely wide enough to include zero — meaning the headline number may not survive its own confidence interval.

### Blind Spots the Council Caught

Peer review surfaced three issues that no single advisor raised:

1. **Trade-count confounding.** The v12 target (median − 0.5·σ across 5 walk-forward windows) likely correlates with trade count — more trades stabilize the walk-forward median. v12 may be selecting high-trade-count combos, and trade count is the actual signal, not Sharpe ranking. Per `project_trade_count_bias.md`, 564 survivors across 50 combos ≈ 11 trades/combo — small-sample inflation territory.
2. **No bootstrap CI on the 0.907 filtered Sharpe.** 564 trades, 40% DD, 1.46y — the CI is wide enough that the result may not be statistically distinguishable from zero. You've run MC bootstrap on every iteration per `CLAUDE.md`; skipping it on the artifact you're debating whether to ship is the miss.
3. **"Short the rejects" is infrastructurally wrong.** MNQ is long/short symmetric via signal direction, not portfolio overlay. The Expansionist's most exciting suggestion is mechanically incoherent with the codebase.

### The Recommendation

**C, gated on the Contrarian's ablation and a bootstrap CI.** Not A — shipping a 0.907 Sharpe whose provenance you can't explain and whose CI you haven't computed is exactly the "temporary fix / no root cause" failure mode `CLAUDE.md` forbids. Not B — unanimously rejected. Not D alone — necessary but not sufficient.

Concretely: ML#1 is a **candidate sampler**, not a ranker. ML#2 is the selector. The objective function is post-filter Sharpe, not combo-level predicted Sharpe. Rename, update `tasks/part_b_findings.md`, and in the next v12 retrain, target ML#2 yield directly rather than robust walk-forward Sharpe. But none of this refactor is justified until you've proven ML#1 adds value over random sampling and that the 0.907 Sharpe has a CI that excludes zero. If the ablation shows random-50 matches v12-top-50 under ML#2, you haven't built a pipeline — you've built ML#2 with a fancy preprocessor, and the honest move is deleting ML#1 entirely.

### The One Thing to Do First

**Run the Contrarian's ablation with bootstrap CIs.** Apply ML#2 to three candidate pools on the OOS window:

1. Random 50 combos from the v11 post-gate 13,814
2. Top-50-by-raw-training-Sharpe
3. Top-50-by-v12-UCB (current `top_strategies_v12.json`)

Report for each: filtered trade count, OOS Sharpe with 10,000-sim bootstrap CI, max DD, and trades/combo distribution. Budget: one afternoon. This single experiment answers three questions simultaneously — is ML#1 adding value, is the 0.907 Sharpe statistically real, and is trade-count the hidden variable. Everything else (regime audit, rename, retarget, scale) is premature until these three numbers are on the page.
