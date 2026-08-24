from src.agents import run

V2_BRIEF = """Implement the v1 critic's approved proposal from /reviews/v1.md as version 2:
a volatility filter added on top of v1's existing momentum + dollar-volume eligibility.

Delegate to strategy-engineer: write /strategies/v2.py. Keep v1's exact logic (126-day
momentum from adjusted close, 20-day/120-day dollar volume ratio filter, rank eligible
names by momentum, top_n equal weight, monthly rebalance) and add ONE new eligibility
condition: exclude a ticker if its trailing realized volatility (mom_window-day daily
return rolling std, annualized by sqrt(252)) exceeds a new vol_threshold parameter.
Do not modify v1.py.

Use v1's exact champion parameters as the base (mom_window=126, vol_short=20,
vol_long=120, vol_ratio_min=1.0, top_n=2) and sweep only vol_threshold in
(0.20, 0.25, 0.30) -- 3 configurations, in ONE sweep call -- so the comparison
to v1 isolates the effect of the volatility filter specifically.

Delegate to research-critic for a review of v2. The critic MUST call write_file to
save its review to /reviews/v2.md as an actual file on disk -- describing the review
in your response text only, without writing the file, does not satisfy the stage gate
and the next version will be blocked.

Then apply the selection rule between v2 and the current champion (v1), state which
of the three gates passed and which failed, and call record_decision for v2."""

if __name__ == "__main__":
    run(V2_BRIEF)
