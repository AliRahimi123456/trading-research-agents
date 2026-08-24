from src.agents import run

V1_FOLLOWUP = """Version 1 already exists at /strategies/v1.py and has already been swept
nine times -- all nine failed with the same bug (pandas resample('M') is invalid in the
installed pandas version; the code has since been corrected to resample('ME')). All 9
of those budget slots are already spent; only 3 configurations remain in v1's budget.

Delegate to strategy-engineer: verify /strategies/v1.py now uses 'ME', then sweep
exactly these 3 remaining configurations in one call: top_n in (2, 3, 4), with
mom_window fixed at 126 and all other parameters at default.

Pass the resulting table to research-critic for a review of v1 (overwrite
/reviews/v1.md with this real review), comparing against the fixed benchmarks. Then
apply the selection rule (v1 has no incumbent, so it becomes champion by default) and
call record_decision for v1 -- no decision has been recorded yet for v1.

Finally report: the chosen configuration, how it compares to equal-weight buy-and-hold
and plain momentum, and the critic's proposal for v2."""

if __name__ == "__main__":
    run(V1_FOLLOWUP)
