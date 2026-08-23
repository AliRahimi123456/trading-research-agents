import json

from src.config import WS

frozen = json.loads((WS / "strategies" / "frozen.json").read_text())
final = json.loads((WS / "results" / "holdout.json").read_text())

REPORT = f"""# Research trail audit

The holdout has been run once and the strategy is frozen. Nothing below can change
which strategy reached the holdout -- this is a review of how the research was
conducted, not another round of tuning.

## 1. What changed at each version, and what evidence drove it

**v1** reproduced the manually-verified baseline exactly: 126-day momentum, a
dollar-volume eligibility filter, equal-weighted top picks. A pre-registered 9-config
grid (3 momentum windows x 3 portfolio sizes) selected `mom_window=126, top_n=2`
(validation Sharpe {final['val']['sharpe']:.4f}). No incumbent existed, so v1 became
champion by default. Its own review flagged that it underperformed both equal-weight
buy-and-hold and plain momentum on Sharpe and CAGR despite trading far more.

**v2** added exactly one structural change on top of v1's exact champion parameters:
a broad-market regime filter, moving the whole portfolio to cash whenever SPY's own
trailing momentum turned negative. Development Sharpe jumped to 0.655 and max
drawdown improved from -0.2757 to -0.1899 -- but validation Sharpe fell to 0.3207,
well below v1's {final['val']['sharpe']:.4f}. The regime filter helped substantially
in one period and hurt substantially in the other.

**v3** kept v2's regime filter, removed the binary dollar-volume filter entirely, and
switched from equal weighting to weighting each pick inversely to its own recent
realized volatility. Swept across top_n in (2, 3, 4); the strongest configuration
(top_n=3) reached validation Sharpe 0.5371 -- within 0.0053 of v1's
{final['val']['sharpe']:.4f} -- while cutting development turnover from 11.66 to 7.05
and improving development drawdown to -0.1654.

## 2. How the selection rule decided each champion

The rule was fixed before v1 ever ran: a challenger needs validation Sharpe not worse
than the incumbent, validation drawdown within 2 percentage points, and development
turnover no more than 20% higher -- all three, or the incumbent stays.

- **v1 vs. no incumbent**: champion by default.
- **v2 vs. v1**: passed the drawdown gate (identical, 0.0pp difference) and the
  turnover gate (9.66 vs. v1's 11.66), but **failed the validation-Sharpe gate**
  (0.3207 vs. 0.5424) decisively. v1 retained.
- **v3 vs. v1**: passed the drawdown gate (-0.2884 vs. -0.2954, within tolerance) and
  the turnover gate by a wide margin (7.05 vs. a 13.99 ceiling), but **failed the
  validation-Sharpe gate by 0.0053** -- the closest decision of the entire study. v1
  retained as the final champion.

## 3. Did the revisions improve the research case, separately from returns?

Yes, even though neither challenger was promoted. v2 established that a regime filter
can materially cut drawdown and turnover -- but also that a large development
improvement can evaporate on validation, which is exactly the failure mode the
three-gate rule exists to catch. v3 showed that removing the volume filter and
switching to volatility-scaled weights gets validation Sharpe within a hair of the
champion while trading dramatically less and drawing down less on development. That's
a genuinely useful negative result: three different structural ideas were tested
against a fixed, un-gamed bar, and none of them clearly beat a simple baseline on the
period that matters for promotion. That is evidence the baseline is more robust than
it first looked, not evidence the process failed.

## 4. Frozen strategy vs. benchmarks on the holdout

| | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| **Frozen v1 (holdout)** | **{final['holdout']['cagr']:.4f}** | **{final['holdout']['sharpe']:.4f}** | **{final['holdout']['max_dd']:.4f}** |
| SPY buy-and-hold | 0.1127 | 0.6829 | -0.2450 |
| Equal-weight buy-and-hold | 0.0986 | 0.6372 | -0.1822 |
| Plain momentum | 0.0522 | 0.3637 | -0.2369 |
| Volume-filtered momentum | 0.1141 | 0.6841 | -0.2369 |

Frozen v1 beat every benchmark on both CAGR and Sharpe on the one holdout evaluation
it was allowed. Only equal-weight buy-and-hold had a shallower drawdown. This is a
favorable result, but it is **one draw from one four-year window** (2022-2025,
covering a specific rate-hiking cycle and equity rally) evaluated exactly once, and it
does not retroactively justify v1's weaker development and validation Sharpe relative
to the simpler benchmarks earlier in the study.

## 5. Did the volume filter earn its turnover?

Mixed, and worth stating honestly rather than picking the answer that flatters the
result. On development and validation, v1's volume-filtered approach traded roughly
40-60% more than plain momentum for a *worse* Sharpe on both periods (see
BENCHMARKS.md: plain_mom dev Sharpe 0.537 vs. v1's 0.478; plain_mom val Sharpe 0.927
vs. v1's 0.542) -- the extra turnover did not clearly earn its keep there. But on the
holdout period, plain momentum's Sharpe collapsed to 0.364 while both volume-filtered
variants (the benchmark and the frozen strategy) held up far better. One plausible
reading is that the volume filter -- and v1's approach more broadly -- was protecting
against exactly the kind of regime that showed up in 2022-2025. An equally plausible
reading is that this is one favorable draw. A single holdout period cannot distinguish
between those two explanations.

## 6. Where the evidence was thin, decisions were weak, or we got lucky

- **v2 was tested at exactly one configuration.** Its large development improvement
  carries essentially no neighboring-parameter robustness evidence -- we don't know if
  it was the regime filter specifically, or that one parameter combination.
- **v3 changed three things relative to v1 simultaneously** (inherited regime filter,
  removed volume filter, switched to volatility weighting), which makes it impossible
  to attribute v3's large development-Sharpe jump to any single mechanism in
  isolation.
- **The v1-vs-v3 margin (0.0053 Sharpe) is razor-thin** -- well within the range that
  a different vol-lookback window, a different rebalance-date convention, or a
  different data vendor's adjustted-close methodology could plausibly flip. The rule
  correctly retained v1 under its own fixed terms, but this was not a robust,
  wide-margin win for v1's specific approach over v3's.
- **Three versions, up to 12 configurations each, one validation set reused across
  all three** is a real multiple-comparisons exposure even with the gates in place.
  The gates bound how much damage that exposure can do; they do not eliminate it.
- **The favorable holdout result is one evaluation of one strategy over one period.**
  It is evidence the process didn't obviously fail, not proof the strategy has real,
  repeatable edge going forward.
"""

if __name__ == "__main__":
    out = WS / "report.md"
    out.write_text(REPORT, encoding="utf-8")
    print(f"report written to {out}")
