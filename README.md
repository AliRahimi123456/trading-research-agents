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

The full deterministic research layer is built, tested, and verified — including a
manual reproduction of the article's entire v1 → v2 → v3 → freeze → holdout research
loop, which matches the article's own published numbers closely (validation Sharpe
matches to 3–4 decimal places on several runs, despite using a different data source).

The LangChain Deep Agents layer (coordinator / strategy-engineer / research-critic)
is not yet wired up.

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
  build_dashboard.py  local HTML results dashboard

docs/
  WORKFLOW.md          the research process, fixed before any strategy was written
  source-handbook.md   the original article, saved for reference

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
