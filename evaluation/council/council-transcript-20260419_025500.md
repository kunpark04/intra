# LLM Council Transcript — 2026-04-19 02:55

## Original question

> What is the interpretation of the Pool B (raw-Sharpe top-50) eval results?

## Framed question

Interpret the results of a 3-arm ablation + reproduction of a trading ML pipeline.

**Context:**
- Solo researcher, backtest-first MNQ 1-min mean-reversion. $50k equity, 5% risk/trade, $5/contract RT friction.
- Two-stage ML stack: ML#1 (v12 UCB LightGBM ranker, target = robust walk-forward Sharpe) picks top-K combos; ML#2 (V4 booster) filters trades via P(win).
- 2026-04-19 3-arm ablation comparing top-50 selection methods:
  - Pool A = random sampling
  - Pool B = `df.sort_values("audit_full_net_sharpe").head(50)` (simple in-sample Sharpe sort)
  - Pool C = v12 UCB ranker
- All three through identical ML#2 V4 + fixed-$500 + $5/contract + 10k-sim MC bootstrap on OOS 2024-10-22 → 2026-04-08 (1.46 years).

**Ablation results:**

| Pool | Sharpe p50 | 95% CI | DD worst | Ruin prob |
|---|---|---|---|---|
| A Random | 0.78 | [-0.88, +2.39] | 283% | 46.0% |
| B Raw-Sharpe | 2.13 | [+0.56, +3.64] | 55% | 0.02% |
| C v12-UCB | 0.88 | [-0.72, +2.45] | 91% | 1.37% |

**Reproduction (full eval notebook suite on Pool B, production config):**
Sharpe p50 2.1331, CI (0.5562, 3.6428), DD worst 55.1%, DD p99 33.67%, ruin 0.02%, sharpe_pos_prob 99.6%, win_rate 33.4%, n_trades 641.
Matches ablation Pool B to 2 decimals on independent code path.

**Decision:** Ship Pool B as new production top-K? How to communicate confidence? Next research step?

---

## Advisor responses (de-anonymized)

### The Contrarian (anonymized as Response C)

Don't ship. You're about to enshrine a sorting operation as a "production ranker" based on one OOS window. Pool B is literally `df.sort_values("audit_full_net_sharpe").head(50)` — in-sample training Sharpe. You have zero evidence this generalizes. A lower-bound Sharpe of 0.56 after $5/contract friction on 641 trades is not a moat — it's a coin flip wearing a suit.

The reproduction "gate" is theater: you reran the same selection rule on the same OOS window with the same seed family. Reproducibility ≠ robustness. DD worst 55% with ruin threshold 50% means you're already past ruin in the worst MC path. MC bootstrap assumes IID trades — your 5-trades-per-combo structure violates that; real drawdowns will be worse than bootstrap implies. You're also bundling two changes (V3→V4 ML#2, UCB→RawSharpe) into one ship.

**Next step:** walk-forward the Pool B rule itself. Pick top-50 by raw Sharpe on rolling training windows, evaluate on the *next* window. If that degrades to Pool C territory, you've confirmed raw-Sharpe is overfit to the full-history audit. Do it before you touch production.

### The First Principles Thinker (anonymized as Response A)

You're asking the wrong question. The right one: what did you actually learn about the problem? ML#1 was supposed to select combos that survive OOS. The ablation proved it performs indistinguishably from random. That's a statement that your 13,814-combo sweep does not contain learnable structure over your chosen features for your chosen target.

Why did raw-Sharpe win? Not because it's clever — it's the dumbest possible baseline. It won because in-sample Sharpe on 5 trades/combo is mostly a lottery, and the winners correlate with "combos that trade rarely and cleanly," which ML#2 filters successfully. You didn't discover a selection method — you discovered that restrictive combos × a decent per-trade filter beats permissive combos × the same filter. The selector is almost irrelevant.

641 trades / 1.46 years is your actual signal budget. CI [0.56, 3.64] is physics, not a flaw to engineer around. The two-stage architecture may be redundant. **Test: ML#2 applied to the full 13,814 pool, no pre-filter.** If that matches Pool B, kill ML#1 entirely. Ship nothing yet.

### The Expansionist (anonymized as Response E)

The ablation exposed a more valuable truth than "better top-K picker": **ML#2 is a variance-compression engine**, and its power scales with how little the upstream layer interferes. All three pools converge at ~33% WR, so ML#2 is doing the real alpha extraction. The differentiator is pool variance — Raw-Sharpe feeds it cleaner inputs by accident.

Upside moves:
1. **Expand the pool, don't shrink it.** Feed ML#2 top-200, top-500, or the entire 13,814-combo sweep at inference. Pool-selection becomes a computational convenience, not a statistical necessity. Sharpe 2.77 on ML#2-only unfiltered gross suggests massive unused capacity.
2. **Retarget ML#1 as an ML#2-collaborator.** Train v13 on *post-ML#2-filter net Sharpe* — the real objective function. UCB failed because it solved the wrong problem.
3. **Multi-instrument (MES, MYM, MGC)** with the same ML#2 architecture could 4-5× trade count without retraining. Cross-asset ensemble tightens the CI dramatically.
4. **Ship Pool B now** with explicit confidence band. Then immediately start v13 retarget + pool expansion. The frontier is much bigger than picking 50 combos better.

### The Outsider (anonymized as Response B)

The elephant: you have a column literally named `audit_full_net_sharpe`, you acknowledge "full partition is referenced," and you're about to ship a ranker built on it. If "full" means full dataset (train + OOS), then ranking by it and evaluating on OOS is not an out-of-sample test — it's "I picked the 50 combos that did well during the period I'm about to grade them on." The fact that this beats a months-of-work ML model that presumably doesn't cheat is exactly what you'd expect if there's leakage. **One line of code confirming the column's date range vs. the OOS window settles it.**

Second: 1.46 years is not a lot. Bootstrap that resamples the same trades measures sampling noise within your one historical path, not regime robustness. Worst DD 55% is brutal for something being called ship-ready.

Third: the gross-with-friction line (Sharpe −0.19, DD 977%) is doing enormous work. Your entire edge lives inside the filter + selection stack — small changes to friction, fill model, or slippage could flip sign.

Fourth: "stat-indistinguishable from random" on ML#1 is a finding, not a failure to hide. Worth a post-mortem before discarding months of work.

**Prove `audit_full_net_sharpe` uses only pre-OOS bars. If it doesn't, the 2.13 is a mirage and nothing ships.**

### The Executor (anonymized as Response D)

Monday morning. You have one clean result, two bundled changes, and a 1.46-year sample. Sequence:

- **Day 1 (~4h):** Unbundle. Run V3-ML#2 + Pool B as a fourth arm. Extractor + generator already support this — just swap the filter. Tells you whether Sharpe 2.13 came from Pool B, V4, or both.
- **Day 2 (~2h):** `tasks/ship_decision.md` with a 4-row comparison (V3+UCB, V4+UCB, V4+Pool B, V3+Pool B). Pick highest p50 AND smallest single-change delta from prod.
- **Day 3 (~1h):** Ship. Update CLAUDE.md. Tag commit `prod-2026-04-20`. Freeze `evaluation/top_strategies_v12_prod.json`.

**Do not this week:** retrain ML#1 (motion, not progress — you don't yet understand *why* it ties random), collect more OOS (you have no live pipe, this is a stall tactic), drop ML#1 entirely (too permanent).

**CI communication:** one sentence in metadata — `"sharpe_ci_95": [0.56, 3.64], "note": "1.46y OOS, lower bound positive but wide"`. You're not pitching an LP.

Ship Pool B. Don't retrain. Move on.

---

## Peer reviews (anonymized mapping revealed above)

### Reviewer 1

**Strongest: B.** B is the only response that interrogates the load-bearing assumption: `audit_full_net_sharpe` likely includes the OOS window. Every other response argues what to ship without first verifying the column isn't contaminated. A single `min/max(date)` check gates everything downstream. C gestures at this ("in-sample training Sharpe") but doesn't name the mechanism.

**Biggest blind spot: E.** E treats Sharpe 2.13 as real signal and proposes scaling (multi-instrument, pool expansion, v13 retarget) on top of a result that hasn't survived a leakage check. E is building a roadmap on an unaudited foundation.

**All missed:** Whether Pools A and C were constructed with the *same* audit column or a leakage-free one. If only B uses `audit_full`, the ablation is apples-to-oranges by construction.

### Reviewer 2

**Strongest: B.** The single falsifiable check that could invalidate the entire result. One grep beats a week of experiments.

**Biggest blind spot: D.** D writes a 3-day ship plan without ever questioning whether the 2.13 is real. Shipping a leaked-target ranker is the worst outcome; D's framework actively accelerates it.

**All missed:** Whether 33% WR across all three arms means the *sweep's parameter diversity itself* is the ceiling, not the ranker. Real lever is the sweep's combo generator (v11 ranges) or instrument expansion — not how to pick 50 of 13,814.

### Reviewer 3

**Strongest: B.** Identifies a potentially fatal, cheaply-testable flaw. Highest information-per-word.

**Biggest blind spot: D.** Treats this as a scheduling problem while ignoring that B's leakage concern, A's selector-irrelevance test, and C's walk-forward check are all cheap and could invalidate the premise.

**All missed:** The **641-trade denominator economics**. At ~440 trades/year on MNQ with 55% DD, the Sharpe CI width is dominated by trade count, not selector choice. Ablation may be underpowered to distinguish B from C regardless of leakage — the "decisive" win is itself suspect.

### Reviewer 4

**Strongest: B.** Identifies the single question that invalidates everything else.

**Biggest blind spot: D.** Prescribes a 3-day ship plan without addressing leakage risk, pool-size confound, or single-window fragility. Ships a potential mirage on schedule.

**All missed:** Whether 55% DD + 0.02% ruin is tolerable given the user's actual capital path. $50k × 55% = $27.5k drawdown. Psychological/margin-call ruin hits earlier. Also: Pool B's win margin (+0.54 Sharpe) vs CI width (±1.5) — the selector difference is inside noise regardless of leakage.

### Reviewer 5

**Strongest: B.** Only response that challenges the core evidence rather than arguing downstream. One cheap check could invalidate C/D/E and reframe A.

**Biggest blind spot: D.** Decisiveness mis-applied. "Ship Monday" without addressing leakage or single-path fragility is the "motion = progress" trap D warns against.

**All missed:** **Selection mechanism stability.** Pool B is top-K by raw in-sample Sharpe on 5-trades/combo samples — the rank itself is a lottery. Rerun the training window by a month and top-50 likely reshuffles dramatically. Missing experiment: rank-stability under training-window perturbation. If <30% overlap across windows, the "edge" is selection noise, not signal.

---

## Chairman Synthesis

### Where the Council Agrees

Four of five advisors (Contrarian, First Principles, Outsider, Executor's own "you don't yet understand *why* it ties random") converge that **the ablation proved ML#1 UCB is broken, but did not prove raw-Sharpe is good**. Only the Expansionist treats Pool B's Sharpe 2.13 as fully actionable signal; the other four flag it as a single-window result with structural fragility.

All five agree (explicitly or implicitly) that **the two bundled changes (V3→V4 ML#2, UCB→Pool B) must be unbundled** before attributing the win. The Executor bakes this into Day 1; the others imply it by refusing to ship.

All five agree the **CI width [0.56, 3.64] is uncomfortable** — the disagreement is only about whether you ship with a caveat (Executor, Expansionist) or don't ship at all (Contrarian, First Principles, Outsider).

### Where the Council Clashes

**Ship now vs. hold.** The Executor says Monday-morning ship with a 3-day unbundling plan; the Expansionist says ship with a confidence band and immediately start pool expansion. The Contrarian, First Principles, and Outsider say hold — the evidence base has load-bearing flaws that are cheap to test but have not been tested.

**What the ablation actually means.** The Expansionist reads it as "ML#2 is the engine, scale it." The First Principles Thinker reads it as "the whole sweep may lack learnable structure." The Outsider reads it as "you may have leakage and aren't comparing like with like." These are three fundamentally different interpretations of the same numbers.

### Blind Spots the Council Caught

Peer review was **unanimous**: Response B (Outsider) was strongest, and Response D (Executor) had the biggest blind spot.

The unified critique of D across all five reviewers: shipping on a 3-day schedule before addressing the leakage check is "motion = progress" — exactly what D itself warned against. The Executor's plan correctly unbundles V3/V4 and UCB/Pool B but does not unbundle "data clean" from "data leaked."

Reviewers also surfaced three things **none** of the five advisors raised:

1. **Selection-column parity** — did Pools A and C use a leakage-free column while Pool B used `audit_full_net_sharpe`? If so, the ablation is apples-to-oranges by construction (Reviewer 1).
2. **Sample-size power** — at 641 trades over 1.46 years, the ablation may be statistically underpowered to distinguish selectors regardless of leakage (Reviewer 3).
3. **Rank stability** — even if the column is leakage-free, the *membership* of Pool B may shuffle dramatically under a training-window perturbation. This is a separate check from walk-forward Sharpe (Reviewer 5).
4. **Psychological ruin** — MC bootstrap flags ruin at 50% equity, but 55% DD on a $50k account is $27.5k drawn down; real-trader / margin-call ruin hits earlier (Reviewer 4).
5. **Sweep-as-ceiling** — the 33% WR convergence across all three pools may say more about v11 sweep parameter ranges than about any ranker (Reviewer 2).

### The Recommendation

**Hold. Do the leakage check first. Then unbundle, then ship or kill.**

This sides with the majority (Contrarian + First Principles + Outsider), with the Outsider's specific falsification as the load-bearing move. The Executor's plan is good structure but premature — it ships before the cheapest test runs.

The correct interpretation of the results is:
- **ML#1 v12 UCB is retired** regardless of what else is true. The ablation established that clearly and nothing in peer review challenged it.
- **Pool B's Sharpe 2.13 is unvalidated**, not confirmed. The reproduction only proves determinism, not robustness.
- The **two-stage architecture question is genuine** (First Principles' "kill ML#1 entirely" test via ML#2-on-full-pool is strictly better information than retraining ML#1 to v13).
- The **CI width and DD profile are not ship-ready** for a solo researcher's $50k account without at least one more OOS slice or a leakage disproof.

Do not ship Pool B to production yet. Do not retrain ML#1 to v13 yet. Do not expand to multi-instrument yet.

### The One Thing to Do First

**Run a single command: print `min(date), max(date)` of the bars used to compute `audit_full_net_sharpe` in `combo_features_v12.parquet`, and compare against the OOS window start `2024-10-22`.**

If the audit column's max date ≥ 2024-10-22, Pool B's "win" is wholly or partly leakage and the ablation invalidates. If the max date is strictly before 2024-10-22, proceed to the Executor's Day-1 unbundling (V3+Pool B as a 4th arm) — but with full awareness that the CI width and rank-stability concerns still constrain how confidently you can ship.

This one check determines whether the next work is "validate the win" or "explain the artifact."
