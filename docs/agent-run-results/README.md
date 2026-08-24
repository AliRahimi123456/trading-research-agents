# Real agent-run results snapshot

A static copy of the actual output from one real, live run of the LangChain Deep
Agents research team (coordinator + strategy-engineer + research-critic) — not the
regenerable `project/workspace/` runtime data, which is gitignored and rebuilt by
running the pipeline yourself.

- `report.md` — the full audit, including two implementation bugs the team hit and
  fixed live, and a coordinator selection-rule error that was caught and corrected
  before the champion was frozen. Start here.
- `registry.csv` — every configuration the engineer swept, successes and failures
  alike.
- `decisions.jsonl` — the champion decision recorded after each version.
- `reviews/v1.md`, `v2.md`, `v3.md` — the critic's reviews for each version.
- `frozen.json` — the final champion strategy and parameters, after correction.

This run used `gpt-4.1` (coordinator, critic) and `gpt-4.1-mini` (engineer) via the
OpenAI API. Re-running `src/run_v1_agent.py` → `src/run_v2_agent.py` →
`src/run_v3_agent.py` yourself will very likely produce different results — LLM
output isn't deterministic, and the whole point of this project is that the
*process* is what's guaranteed to be sound, not any specific run's outcome.
