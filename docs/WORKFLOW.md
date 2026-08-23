# Research workflow

This fixes the *process* before any code that depends on it gets written. Nothing
below is negotiable by the agents once the agent layer exists — it is enforced in
`src/registry.py` (stage gates) and `src/selection.py` (the promotion rule), not
just written down here.

## Roles

- **strategy-engineer** — writes strategy code, runs it through the fixed backtester
  via `sweep()`. Cannot decide what happens to its own results.
- **research-critic** — reads results, writes a structured review. Denied write
  access to `/strategies/**` at the permission layer, not just by instruction.
- **coordinator** — sequences the loop and applies the fixed selection rule. Does
  not author the rule; the rule lives in code, written before v1 ever runs.

## Version sequence (strictly serial)

```
v1: implement -> sweep -> review -> decide  (no incumbent yet; v1 becomes champion)
        |
        v   (stage gate: v1 must have a successful run, a review file, and a
        |    recorded decision before v2 can be swept at all)
v2: implement -> sweep -> review -> decide  (challenger vs. v1, three gates)
        |
        v   (same stage gate, against v2)
v3: implement -> sweep -> review -> decide  (challenger vs. current champion)
        |
        v
freeze the surviving strategy + params  -->  unlock holdout (2022-2025)  -->  ONE
final evaluation, no further revision possible, then a post-freeze audit.
```

## What's fixed before v1 runs, and why

| Fixed in advance          | Prevents                                                  |
|----------------------------|-------------------------------------------------------------|
| Backtest engine (`engine.py`) | Every version computing its own definition of "return"   |
| Universe (9 ETFs) & date splits | Cherry-picking a universe or window that flatters a result |
| Cost convention (10 bps)   | A revision "improving" by quietly assuming free trading      |
| Four benchmark strategies  | Judging progress only against whatever the last version did |
| Selection rule (3 gates)   | Deciding, after seeing results, what "better" means          |
| Config budget (12/version) | Turning validation into a second, informally-optimized dev set |
| Holdout isolation          | Any version, even accidentally, tuning against the final test |

## Selection rule (fixed, not to be edited after v1 exists)

A challenger replaces the incumbent only if **all three** hold:

1. Validation Sharpe is not worse than the incumbent's.
2. Validation max drawdown is within 2 percentage points of the incumbent's.
3. Development annual turnover is no more than 20% above the incumbent's.

Ties go to the incumbent. A higher development Sharpe alone is never sufficient.
