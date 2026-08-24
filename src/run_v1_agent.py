from src.agents import run

V1_BRIEF = """Build Version 1, the baseline.

Delegate to strategy-engineer. /strategies/v1.py: 126-day momentum from adjusted close;
20-day over 120-day average dollar volume from raw close x raw volume; eligible if
momentum > 0 and volume ratio > 1.0; rank eligible by momentum, hold top 3 equal weight,
rebalance monthly, cash otherwise. Parameters: mom_window, vol_short, vol_long,
vol_ratio_min, top_n.

Sweep exactly these nine in one call: mom_window in (63, 126, 189) crossed with
top_n in (2, 3, 4), everything else at default.

Pass the table to research-critic for a review of v1, telling it to compare against the
fixed benchmarks. Then apply the selection rule (v1 has no incumbent, so it becomes the
champion by default) and call record_decision for v1.

Finally report: the chosen configuration, how it compares to equal-weight buy-and-hold
and plain momentum, and the critic's proposal with your decision."""

if __name__ == "__main__":
    run(V1_BRIEF)
