from src.agents import run

V3_BRIEF = """Implement the v2 critic's approved proposal from /reviews/v2.md as the
final version, v3: replace v2's binary volatility eligibility cutoff with
volatility-based position sizing.

Delegate to strategy-engineer: write /strategies/v3.py. Reuse v1's exact eligibility
logic UNCHANGED and copied exactly, not reimplemented from memory:
- momentum = adj_close.pct_change(mom_window)
- dollar volume ratio = (close * volume).rolling(vol_short).mean() divided by
  (close * volume).rolling(vol_long).mean() -- use raw close, not adjusted close, and
  .mean(), not .sum(). This exact convention has caused bugs in both v1 and v2 when
  reimplemented from scratch; copy it verbatim rather than rewriting it.
- eligible if momentum > 0 and volume ratio > vol_ratio_min

Do not modify v1.py or v2.py.

The only change: instead of equal-weighting the top_n eligible names by momentum,
weight each selected name inversely to its trailing realized volatility (126-day
rolling standard deviation of daily returns, annualized), normalized so the selected
weights sum to 1.0. Use a fixed 126-day volatility lookback -- do not add it as a new
sweepable parameter, per the v2 review's overfit-risk note.

Sweep exactly these 3 configurations in one call: top_n in (2, 3, 4), with mom_window,
vol_short, vol_long, vol_ratio_min all fixed at v1's champion values (126, 20, 120, 1.0).

Delegate to research-critic for a review of v3. The critic MUST call write_file to
persist its review to /reviews/v3.md -- describing the review only in response text
does not satisfy the stage gate.

Then apply the selection rule between v3 and the current champion (v1), state exactly
which of the three gates passed and which failed with the real numbers, and call
record_decision for v3. This is the final version -- after this decision, write
/strategies/frozen.json containing exactly {"version": "<champion>", "params": {...},
"rationale": "..."}, where version is whichever the selection rule says is champion
(may be v1 or v3)."""

if __name__ == "__main__":
    run(V3_BRIEF)
