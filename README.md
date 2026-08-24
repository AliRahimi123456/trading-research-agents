# Trading Research Agents

A multi-agent research system for developing and evaluating trading strategies, built
around one core idea: **agents can propose and challenge strategy ideas, but they
never control the evidence used to judge them.**

Inspired by Nikhil Adithyan's *"How to Build a Multi-Agent Trading Research System
with LangChain Deep Agents"* handbook, reproduced independently with a from-scratch
implementation and a few deliberate deviations (see below).

## The idea

An LLM agent that writes a strategy, runs its own backtest, and reports its own
results has an enormous, mostly-invisible action space for making its numbers look
better without the strategy actually improving. This project removes that action
space in code, before any agent gets involved:

- A **fixed, hand-verified backtest engine** computes every strategy's returns,
  turnover, and metrics the same way, every time.
- Strategy code runs in an **isolated subprocess** with no network access, no
  credentials, and no path back to the holdout dataset.
- An **experiment registry** logs every run — including failures — and caps how many
  configurations can be tested per strategy version.
- A **fixed 3-gate selection rule**, written before any strategy existed, decides
  whether a challenger replaces the current champion. A higher development Sharpe
  ratio is explicitly not one of the gates.
- The final holdout period stays physically inaccessible until the champion is frozen.

## Status

Complete, end to end. The deterministic layer (engine, benchmarks, isolation,
registry, selection rule) is built, tested, and verified. The real LangChain Deep
Agents team (coordinator / strategy-engineer / research-critic) is wired up and has
run a genuine, live v1 → v2 → v3 → freeze → holdout research cycle.

**The standout result isn't the returns — it's what the guardrails caught.** During
that one real run: the engineer's code hit a pandas version bug that silently failed
9 of v1's 12 allotted configurations; a second implementation bug in v2 produced
identical zero results across three configurations that looked like valid (but
uninteresting) evidence until inspected closely; and — most importantly — **the
coordinator misapplied its own fixed selection rule on the final promotion decision**,
describing a genuine 0.0076 Sharpe improvement as "a tie" and keeping the wrong
strategy as champion. That error was only caught because the selection rule lives as
independently-callable code (`select_champion()`), not just agent-narrated behavior —
re-running it against the real recorded numbers outside the agent's own summary
proved the coordinator was wrong, and the decision was corrected before freezing.
Full writeup: [`docs/agent-run-results/report.md`](docs/agent-run-results/report.md).

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

No API key is required for the deterministic layer — market data comes from
`yfinance` (free, no key). Run the pipeline in order:

```bash
python -m src.data_prep          # fetch + validate + split data
python -m src.engine              # verify the backtest engine
python -m src.benchmarks          # compute fixed benchmarks
python -m src.check_isolation     # verify sandboxed execution matches in-process
python -m src.check_probe         # verify no credential/data leakage
python -m src.check_v1            # sweep the v1 baseline
python -m src.check_v2            # sweep v2 (market-regime filter)
python -m src.check_v3            # sweep v3 (volatility-scaled weights)
python -m src.freeze_and_holdout  # freeze the champion, unlock and evaluate holdout
python -m src.write_report        # write the self-audit report
python -m src.build_dashboard     # generate the local results dashboard
```

Then view the dashboard:

```bash
python -m http.server 8000 --directory project/workspace
```

and open `http://localhost:8000/dashboard.html`.

### Running the real agent layer

Requires an OpenAI API key in `.env` as `OPENAI_API_KEY=sk-...`, plus the agent-layer
packages (already in `requirements.txt`: `langchain`, `langchain-openai`, `langgraph`,
`deepagents`, `langsmith`). This makes real, billed API calls.

```bash
python -m src.run_v1_agent   # coordinator delegates v1, engineer sweeps, critic reviews
python -m src.run_v2_agent   # v2, tested against v1
python -m src.run_v3_agent   # v3, the final version -- freezes the champion
```

Verify each version's decision, review, and successful runs actually landed on disk
before trusting the coordinator's own summary of what happened — see
[`docs/agent-run-results/`](docs/agent-run-results/) for what a real run produced,
including a coordinator error that was caught this way.

## Project layout

```
src/            deterministic research code
  config.py       fixed constants: tickers, date splits, paths
  data_prep.py    EODHD-free data pipeline (yfinance), workspace/private split
  engine.py       backtest engine + metrics, hand-verified
  benchmarks.py   four fixed reference strategies
  runner.py       standalone script executed inside the sandbox
  isolation.py    builds the sandbox, scrubs the environment
  registry.py     experiment log, stage gates, sweep/read_registry/record_decision
  selection.py    the fixed 3-gate promotion rule
  agent_tools.py  @tool-wrapped sweep/read_registry/record_decision for the agents
  agents.py       the real coordinator/strategy-engineer/research-critic team
  run_v1_agent.py, run_v2_agent.py, run_v3_agent.py   run each version through the real team
  write_report.py     self-audit report of the real agent run
  build_dashboard.py  local HTML results dashboard

docs/
  WORKFLOW.md              the research process, fixed before any strategy was written
  source-handbook.md       the original article, saved for reference
  agent-run-results/       static snapshot of a real agent run's actual output

project/        runtime data (gitignored) — raw cache, workspace, private/holdout
```

## Deviations from the original article

- **`yfinance` instead of EODHD** — the free EODHD tier only provides one year of
  history and ignores date-range parameters, incompatible with the 20-year
  dev/val/holdout split this project needs. `yfinance` is free with no key and
  provides full multi-decade history.
- **A local dashboard** (`build_dashboard.py`) — not in the original article.

## Credit

Based on the handbook by Nikhil Adithyan. This repository is an independent, from-
scratch reproduction built for learning purposes, not a copy of the original code.
