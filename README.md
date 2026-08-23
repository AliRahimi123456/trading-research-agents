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
