"""The Deep Agents research team: coordinator, strategy-engineer, research-critic.

Model note: the article uses "openai:gpt-5.6-terra" with a reasoning={"effort": ...}
kwarg -- neither is real. gpt-5.6-terra doesn't exist, and the reasoning-effort
parameter only applies to OpenAI's reasoning models (o-series / gpt-5 reasoning
variants), not to gpt-4.1. Using gpt-4.1 / gpt-4.1-mini here instead, with no
reasoning kwarg. Swap these for whatever's current on your account if you like.
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import FilesystemBackend

from src.config import WS
from src.selection import SELECTION_RULE
from src.agent_tools import sweep, read_registry, record_decision

load_dotenv(override=True)

# max_retries: automatic exponential-backoff retry on transient errors (rate limits,
# 5xx). Free-tier/new accounts have low tokens-per-minute limits that a multi-agent
# run's accumulated context can exceed mid-run; without this, one 429 kills the whole
# run and wastes every token already spent on it (InMemorySaver can't resume a failed
# run across separate process invocations).
MANAGER = init_chat_model("openai:gpt-4.1", max_retries=8)        # coordinator + critic: judgment, evidence review
WORKER = init_chat_model("openai:gpt-4.1-mini", max_retries=8)    # engineer: mostly mechanical implementation

BENCH_TEXT = (WS / "BENCHMARKS.md").read_text() if (WS / "BENCHMARKS.md").exists() else "not yet computed"

CONTRACT = """
Every strategy file defines exactly one function:

    def target_weights(data, **params) -> pd.DataFrame

    index   : rebalance dates, all of which must exist in data["adj_close"].index
    columns : the nine tickers
    values  : target weights, each row summing to <= 1.0 (remainder is cash)

data keys: adj_close, close, volume, returns (DataFrames, dates x tickers)
Use adj_close for momentum and returns. Use close * volume for dollar volume.
A row dated t is a decision made on t's close; the engine applies it on t+1.
Guard against empty selections: if nothing qualifies, leave the row at zero.

Your code runs in an isolated subprocess with no network, no credentials and no
holdout data. Import only pandas and numpy.

Working skeleton:

import pandas as pd
def target_weights(data, mom_window=126, top_n=3):
    adj = data["adj_close"]
    mom = adj.pct_change(mom_window)
    dates = pd.DatetimeIndex(adj.index.to_series().resample("ME").last().dropna())
    w = pd.DataFrame(0.0, index=dates, columns=adj.columns)
    for d in dates:
        picks = mom.loc[d].dropna().nlargest(top_n).index
        if len(picks):
            w.loc[d, picks] = 1.0 / len(picks)
    return w
"""

RULES = f"""
Layout: /strategies/vN.py, /results/, /reviews/, /registry.csv, /decisions.jsonl

Stage gates, enforced by the sweep tool:
vN cannot be swept until v(N-1) has successful runs, a review at /reviews/v(N-1).md,
and a decision recorded via record_decision. There is no way around this.

Hard limits: three versions; at most 12 configurations per version; one major
structural change per revision. Engine, universe, splits, benchmark and cost
convention are fixed. The holdout does not exist for you; never ask for it.

{SELECTION_RULE}

Fixed benchmarks, computed before any version was written:
{BENCH_TEXT}

Do not call ls, glob, grep or read_file unless told a specific file exists and you
need its contents.
"""

engineer = {
    "name": "strategy-engineer",
    "description": "Writes strategy files and sweeps them through the fixed backtester in one batched call. Use for anything that creates code or produces metrics.",
    "system_prompt": f"""You implement strategies. You do not decide what to implement.
{RULES}{CONTRACT}
Procedure:
1. Write the strategy file with write_file.
2. Call sweep ONCE with the entire parameter grid as a JSON list. Never per configuration.
3. If a run errors, read the message, fix the file, call sweep again. Errors count
   against the budget.
4. Report back in under 200 words: filename, the returned table verbatim, and the one
   configuration you recommend with a one-line reason. Never paste code back.""",
    "tools": [sweep],
    "model": WORKER,
}

critic = {
    "name": "research-critic",
    "description": "Reads a results table and returns exactly one evidence-backed weakness with one proposed structural change. Use after every version is swept.",
    "system_prompt": f"""You review results. You never write or edit strategy code.
{RULES}
The results table is given to you in the task description. Do not go looking for it.
Call read_registry only to compare against an earlier version.

Write your review to /reviews/vN.md under exactly these five headings:

Weakness     one sentence
Evidence     specific numbers from the table, compared against the fixed benchmarks
Change       one structural change, not a parameter nudge
Expected     what it should do to which metric, and why
Overfit risk how this could be curve-fitting, and what would disconfirm it

A higher Sharpe alone is not evidence. Compare against equal-weight buy-and-hold and
plain momentum, not just SPY. Check the 20bps column against the 10bps one, whether
the dev result survives validation, and whether neighbouring parameters behave
similarly. If dev and val disagree, that disagreement is the finding.""",
    "tools": [read_registry],
    "model": MANAGER,
    "permissions": [
        FilesystemPermission(operations=["write"], paths=["/strategies/**"], mode="deny"),
        FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="allow"),
    ],
}

COORDINATOR = f"""You run a quantitative research process and are judged on the honesty
of the process, not on the returns.
{RULES}
Your loop for each version N:
1. plan with write_todos
2. delegate implementation and sweeping to strategy-engineer
3. pass the engineer's table verbatim into the task description for research-critic
4. apply the selection rule yourself and state which gates passed or failed
5. call record_decision with the resulting champion and your rationale

Step 5 is mandatory. The next version is blocked until it is done.

Reject proposals that are parameter tuning dressed up as structure. The champion does
not change just because a newer version exists. Never overwrite an earlier version."""

agent = create_deep_agent(
    model=MANAGER,
    tools=[sweep, read_registry, record_decision],
    system_prompt=COORDINATOR,
    subagents=[engineer, critic],
    backend=FilesystemBackend(root_dir=str(WS), virtual_mode=True),
    checkpointer=InMemorySaver(),
    name="coordinator",
)

THREAD = {"configurable": {"thread_id": "trading-research-v2"}}


def run(prompt: str):
    out = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, THREAD)
    content = out["messages"][-1].content
    print(content if isinstance(content, str) else
          "\n".join(b.get("text", "") for b in content if b.get("type") == "text"))
    return out


if __name__ == "__main__":
    print("subagent models:", engineer["model"].model_name, critic["model"].model_name)
    print(WORKER.invoke("reply with the single word: ok").content)
