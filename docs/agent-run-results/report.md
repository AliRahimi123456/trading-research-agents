# Research trail audit

This covers the real, live LangChain Deep Agents research cycle -- coordinator,
strategy-engineer, research-critic -- not the earlier manual reproduction used to
validate the deterministic layer before the agent team existed. The holdout has been
run once and the strategy is frozen; nothing below can change which strategy reached
it. This is a record of how the research was actually conducted, including two
implementation bugs and one selection-rule violation caught during the run, all
corrected before the champion was frozen.

## 1. What changed at each version, and what evidence drove it

**v1** was given a pre-registered 9-configuration grid (3 momentum windows x 3
portfolio sizes). All 9 initial runs failed identically: the engineer wrote
`resample('M')`, which pandas 3.0 no longer accepts (`'M'` was removed in favor of
`'ME')`. This consumed 9 of v1's 12-configuration budget on a bug, not a real test.
The engineer self-corrected the code, but only 3 budget slots remained; a follow-up
run swept the 3 most informative configurations (top_n in 2/3/4 at mom_window=126)
against the fixed code and got real results. Separately, the critic's review was
never actually persisted to `/reviews/v1.md` on the first pass -- the coordinator's
report described review content as if the file existed, but it didn't. The review
was reconstructed from the critic's actual (real, not fabricated) analysis and
written to disk before the stage gate would allow v2.

**v2** tested the v1 critic's proposal: exclude names whose trailing realized
volatility exceeds a threshold. The first sweep (`vol_threshold` in 0.20/0.25/0.30)
produced identical all-zero results across all three configurations and all metrics
-- not because the volatility filter was too strict, but because the engineer's
reimplementation of the dollar-volume ratio used `.sum()` over 20-day and 120-day
windows instead of `.mean()`, which structurally produces a ratio near 20/120 ≈ 0.17,
almost always below the required 1.0 threshold. This silently excluded every ticker
before the volatility filter was ever reached. The bug was fixed to match v1's
correct convention (mean, raw close not adjusted), and v2 was re-swept for real.

**v3** replaced v2's rejected hard volatility cutoff with volatility-based position
sizing: v1's existing eligibility unchanged, but selected names weighted inversely to
trailing realized volatility rather than equal-weighted. Swept top_n in (2, 3, 4).

## 2. How the selection rule decided each champion (including a correction)

- **v1 vs. no incumbent**: champion by default.
- **v2 vs. v1** (on the corrected, real sweep): failed the validation-Sharpe gate
  (0.4488 vs. v1's 0.5424) and also made development drawdown worse (-0.3483 vs.
  -0.2757). v1 correctly retained.
- **v3 vs. v1**: this is the important one. v3's best configuration (top_n=2) scored
  validation Sharpe **0.5500** against v1's **0.5424** -- higher, not tied. All three
  gates pass under the rule exactly as written. The coordinator's own report initially
  described this as "not better, not worse... a tie" and invoked "ties go to the
  incumbent" to keep v1 as champion. **That was incorrect.** The fixed rule contains
  no discretion for judging an improvement "not meaningful enough" -- it is a
  mechanical pass/fail on three numeric gates, precisely so that no one (agent or
  human) can talk themselves into a preferred outcome after seeing the result. This
  was caught by independently re-running `select_champion()` against the real
  registry data outside the agent's own narrative, which is the entire reason that
  function exists as reusable, checkable code rather than something only described in
  a prompt. The decision was corrected and v3 was frozen as champion.

## 3. Did the revisions improve the research case, separately from returns?

Yes, genuinely. v2 tested a real hypothesis (hard volatility exclusion) and found it
wanting -- once the implementation bug was fixed, the evidence was real and the
rejection was correct. v3 tested a related but different hypothesis (volatility as a
weighting signal rather than an exclusion filter) and it worked, passing all three
gates on real evidence. That's the research process functioning as intended across
two structurally different ideas testing the same underlying concern (should
volatility affect which names get held).

The more important process finding, though, is that **the guardrails caught failures
at every layer they were designed to catch them**: a code bug in v1 was caught by the
registry recording every failed run rather than hiding it; a second, more subtle code
bug in v2 was caught by refusing to treat a suspiciously-uniform result (identical
zeros across three different threshold values) as face-value evidence; and a
reasoning error in the coordinator's own application of the selection rule was caught
by independently re-checking its claimed gate results against the same code the
coordinator was supposed to be using. None of these were hypothetical risks discussed
in the abstract -- all three happened, for real, in this one research cycle.

## 4. Frozen strategy vs. benchmarks on the holdout

| | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| **Frozen v3 (holdout)** | **0.1345** | **0.7863** | **-0.2243** |
| SPY buy-and-hold | 0.1127 | 0.6829 | -0.2450 |
| Equal-weight buy-and-hold | 0.0986 | 0.6372 | -0.1822 |
| Plain momentum | 0.0522 | 0.3637 | -0.2369 |
| Volume-filtered momentum | 0.1141 | 0.6841 | -0.2369 |

Frozen v3 beat every benchmark on both CAGR and Sharpe on the one holdout evaluation
it was allowed. Only equal-weight buy-and-hold had a shallower drawdown. As with the
earlier manual reproduction, this is one draw from one four-year window (2022-2025)
evaluated exactly once -- favorable, but not proof of a durable edge on its own.

## 5. Did the volatility-based interventions earn their keep?

Mixed, and the two versions disagree in an informative way. v2's hard exclusion
filter made things worse on every axis once correctly implemented -- excluding
volatile names apparently removed real momentum leaders more often than it removed
genuinely erratic ones. v3's softer intervention (downweight, don't exclude) passed
the bar. That's a real, substantive finding: for this universe and this signal,
volatility is more useful as a sizing input than as a gate.

## 6. Where the evidence was thin, decisions were weak, or intervention was required

- **Two real implementation bugs, not process failures.** The pandas `'M'`/`'ME'`
  removal and the volume-ratio `.sum()`/`.mean()` regression were both caught because
  failed and suspicious-looking runs were persisted and inspected rather than trusted
  from an agent's own summary -- exactly the "trust but verify" principle this whole
  system is built around.
- **The critic's first v1 review was never actually written to disk**, despite being
  described as if it existed. This was caught by the stage gate itself refusing to
  open v2 until the file genuinely existed, not by noticing the discrepancy first.
- **The coordinator misapplied its own selection rule on the highest-stakes decision
  in the study** -- the one that determines which strategy reaches the holdout.
  This is the most serious finding of the whole run. It was corrected only because
  `select_champion()` exists as independently callable code, not just prompted
  behavior, and because the actual registry numbers were checked rather than the
  agent's stated conclusion. A system that only had the coordinator's own narrative
  to go on would have frozen the wrong strategy.
- **The frozen champion (v3) beating every holdout benchmark is one draw, not a
  validated edge.** The process being followed correctly (after correction) says
  nothing about whether volatility-weighted momentum has genuine, repeatable value
  going forward.
