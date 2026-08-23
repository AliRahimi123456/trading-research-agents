# Source handbook (pasted by user, saved for reference)

Original article: "How to Build a Multi-Agent Trading Research System with LangChain Deep Agents [Full Handbook]" by Nikhil Adithyan. Saved here in stages as pasted, for reference while building this project. Not all of this code was run as-is -- see src/ for our adapted implementation (yfinance instead of EODHD, refactored module boundaries, etc).

---

## Stage 1: Prerequisites through "Verify Execution Parity and Data Boundaries"

(Covers: Prerequisites, Design the Research Workflow, Set Up the Python Research
Environment, Prepare the EODHD Research Data, Build a Deterministic Strategy
Evaluation Layer parts 1-5. Fully implemented already in src/config.py,
src/data_prep.py, src/engine.py, src/benchmarks.py, src/runner.py, src/isolation.py,
src/check_isolation.py, src/check_probe.py.)

[Content matches what was already pasted at the start of this conversation -- the
EODHD fetch/validation/split pipeline, engine.py's backtest()/metrics()/load_split(),
the four benchmark strategies, runner.py, isolated_environment()/run_isolated(), and
the parity_check.py + probe.py verification scripts. See chat history for full text.]

---

## Stage 2: "Create the Experiment and Decision Layer"

Create the Experiment and Decision Layer
The backtesting engine now gives every strategy the same evaluation path. But we still need to control what happens across repeated experiments.

If an agent can keep testing new configurations indefinitely, ignore failed runs, or move to a new strategy version before the previous one has been reviewed, the research process can still drift toward whatever result looks best. So the next layer will track every experiment, enforce a fixed research budget, and require each version to pass through the same sequence before the next one can begin.

### 1. Create the Experiment Registry
We'll start with a registry that records every configuration tested by the system.

```python
REGISTRY = WS / "registry.csv"
DECISIONS = WS / "decisions.jsonl"
MAX_CONFIGS = 12
COLS = ["version","run","status","params","note","dev_cagr","dev_sharpe","dev_sortino",
        "dev_max_dd","dev_turnover","val_cagr","val_sharpe","val_max_dd","dev_cagr_20bps","error"]

def _used(version):
    if not REGISTRY.exists(): return 0
    return int((pd.read_csv(REGISTRY)["version"] == version).sum())

def _decisions():
    if not DECISIONS.exists(): return []
    return [json.loads(l) for l in DECISIONS.read_text().splitlines() if l.strip()]

def _stage_ok(version):
    """vN cannot begin until v(N-1) is swept, reviewed and decided."""
    if not (version.startswith("v") and version[1:].isdigit()): return True, ""
    n = int(version[1:])
    if n <= 1: return True, ""
    prev = f"v{n-1}"
    if not REGISTRY.exists() or _used(prev) == 0:
        return False, f"stage gate: {prev} has no recorded runs. Complete {prev} first."
    reg = pd.read_csv(REGISTRY)
    if reg[(reg.version == prev) & (reg.status == "ok")].empty:
        return False, f"stage gate: {prev} has no successful runs."
    if not (WS/"reviews"/f"{prev}.md").exists():
        return False, f"stage gate: /reviews/{prev}.md does not exist. Get a critic review first."
    if not any(d["version"] == prev for d in _decisions()):
        return False, f"stage gate: no decision recorded for {prev}. Call record_decision first."
    return True, ""
```

`MAX_CONFIGS = 12` puts a hard ceiling on the number of configurations that can be tested within any strategy version. That matters because validation data can also be overused. If the agent gets unlimited opportunities to search different parameter combinations and keeps selecting whichever one performs best on validation, the validation set gradually becomes another optimization target.

The stage gate controls a different problem. A new version can't start simply because the agent has another idea. Before v2 can be tested, v1 must already have at least one successful run, a critic review, and a recorded decision. The same sequence applies before v3.

### 2. Create the Research Tools
The agents will interact with this layer through three LangChain tools. The most important one is `sweep()`. It's the only route through which an agent can obtain official backtest results.

```python
from langchain.tools import tool

@tool
def sweep(version: str, grid_json: str, note: str = "") -> str:
    """Backtest strategies/<version>.py over several parameter sets in ONE call.

    version   : file stem, e.g. "v1" for strategies/v1.py
    grid_json : JSON list of parameter objects, e.g. [{"top_n":3},{"top_n":4}]
    note      : short reason for this sweep

    Runs each configuration in an isolated subprocess. Returns a CSV table sorted by
    validation Sharpe. Max 12 configurations per version, cumulative. Every row is
    written to registry.csv, including failures. vN is blocked until v(N-1) is swept,
    reviewed and decided.
    """
    ok, why = _stage_ok(version)
    if not ok: return f"error: {why}"
    used = _used(version)
    try:
        grid = json.loads(grid_json)
        if isinstance(grid, dict): grid = [grid]
    except Exception as e:
        return f"error: grid_json is not valid JSON ({e})"
    if used + len(grid) > MAX_CONFIGS:
        return f"error: budget. {used}/{MAX_CONFIGS} used on {version}, you asked for {len(grid)} more."
    path = WS/"strategies"/f"{version}.py"
    if not path.exists():
        return f"error: {path.name} does not exist. Write it first."

    rows = []
    for i, params in enumerate(grid, start=used + 1):
        row = {"version": version, "run": i, "note": note,
               "params": json.dumps(params, separators=(",", ":"))}
        dev = run_isolated(path, params, "dev")
        if not dev["ok"]:
            row.update(status="error", error=dev["error"].strip().split("\n")[-1][:150])
            rows.append(row); continue
        val = run_isolated(path, params, "val")
        c20 = run_isolated(path, params, "dev", cost_bps=20.0)
        dm, vm = dev["metrics"], val["metrics"]
        row.update(status="ok", dev_cagr=dm["cagr"], dev_sharpe=dm["sharpe"],
                   dev_sortino=dm["sortino"], dev_max_dd=dm["max_dd"],
                   dev_turnover=dm["ann_turnover"], val_cagr=vm["cagr"],
                   val_sharpe=vm["sharpe"], val_max_dd=vm["max_dd"],
                   dev_cagr_20bps=c20["metrics"]["cagr"] if c20["ok"] else None)
        tag = f"{version}_run{i}"
        (WS/"results"/f"{tag}.json").write_text(json.dumps({"params": params, "dev": dm, "val": vm}, indent=2))
        eq = pd.Series(dev["equity"], index=pd.to_datetime(dev["dates"]))
        plt.figure(figsize=(8,3)); plt.plot(eq); plt.yscale("log"); plt.title(tag)
        plt.tight_layout(); plt.savefig(WS/"results"/f"{tag}.png", dpi=90); plt.close("all")
        rows.append(row)

    df = pd.DataFrame(rows).reindex(columns=COLS)
    df.to_csv(REGISTRY, mode="a", header=not REGISTRY.exists(), index=False)
    out = df.drop(columns=["version","note"]).round(3).dropna(axis=1, how="all")
    if "val_sharpe" in out:
        out = out.sort_values("val_sharpe", ascending=False, na_position="last")
    return out.to_csv(index=False)

@tool
def read_registry(version: str = "") -> str:
    """Every run recorded so far as CSV, accepted and rejected. Pass a version to filter."""
    if not REGISTRY.exists(): return "empty"
    r = pd.read_csv(REGISTRY)
    if version: r = r[r["version"] == version]
    return r[["version","run","status","params","dev_sharpe","dev_sortino",
              "dev_max_dd","val_sharpe","val_max_dd","error"]].to_csv(index=False)

@tool
def record_decision(version: str, champion: str, rationale: str, params_json: str) -> str:
    """Record the approved outcome of a version. REQUIRED before the next version can be swept.

    version    : the version just reviewed, e.g. "v2"
    champion   : which version is champion after applying the selection rule
    rationale  : cite the selection rule and the specific numbers that decided it
    params_json: the champion's parameters as JSON
    """
    if any(d["version"] == version for d in _decisions()):
        return f"error: a decision for {version} already exists and cannot be overwritten."
    rec = {"version": version, "champion": champion, "rationale": rationale,
           "params": json.loads(params_json), "ts": time.time()}
    with DECISIONS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return f"recorded. champion is now {champion}"
```

For every configuration, `sweep()` runs development and validation through the isolated evaluation path. It also reruns development at 20 basis points of transaction costs, so the critic can see whether a result is especially sensitive to the default 10-bps assumption.

Successful runs produce metrics, JSON result files, and an equity curve. Failed runs still enter registry.csv instead of disappearing from the research history. `read_registry()` lets the agents inspect the recorded evidence; `record_decision()` creates the official outcome of each version and can't be overwritten once written.

### 3. Fix the Strategy Selection Rule
The registry tells us what happened, but we still need to define what counts as an improvement, fixed before any agent-generated version is run.

```python
SELECTION_RULE = """
# Version selection rule (fixed before any version was run)

A challenger replaces the incumbent champion only if it passes ALL THREE gates:

1. Validation Sharpe is not worse than the incumbent's
2. Validation max drawdown is within 2 percentage points of the incumbent's
3. Development annual turnover is no more than 20% above the incumbent's

Ties go to the incumbent. A newer version does not automatically replace an older one.
A higher development Sharpe is not sufficient and is not one of the gates.
"""
(WS/"SELECTION_RULE.md").write_text(SELECTION_RULE)

def select_champion(challenger, incumbent, name_c, name_i):
    if incumbent is None: return name_c, "no incumbent"
    checks = [("validation Sharpe not worse",
               challenger["val_sharpe"] >= incumbent["val_sharpe"]),
              ("validation drawdown within 2pp",
               challenger["val_max_dd"] >= incumbent["val_max_dd"] - 0.02),
              ("turnover within +20%",
               challenger["dev_turnover"] <= incumbent["dev_turnover"] * 1.20)]
    failed = [n for n, ok in checks if not ok]
    if failed:
        return name_i, "incumbent retained; challenger failed: " + "; ".join(failed)
    return name_c, "challenger passed all three gates"

def best_of(version):
    reg = pd.read_csv(REGISTRY)
    rows = reg[(reg.version == version) & (reg.status == "ok")]
    return None if rows.empty else rows.sort_values("val_sharpe", ascending=False).iloc[0]
```

There are two levels of selection: `best_of()` finds the strongest successful configuration within a version using validation Sharpe, but winning that internal sweep doesn't automatically make the strategy the new champion. `select_champion()` then compares that candidate with the incumbent across all three gates. Development Sharpe is intentionally absent from those gates.

---

## Stage 3: "Establish the Manual Baseline" through "Create the Coordinator"

### Establish the Manual Baseline

Before giving the research tools to Deep Agents, we run the initial strategy manually through the same evaluation layer -- a known reference point confirming data, strategy logic, backtesting engine, and benchmark calculations all agree before any agent starts modifying the strategy.

```python
def manual_baseline(data, mom_window=126, vol_short=20, vol_long=120,
                    vol_ratio_min=1.0, top_n=3):
    adj, cls, vol = data["adj_close"], data["close"], data["volume"]
    mom = adj.pct_change(mom_window)
    dv = cls * vol
    ratio = dv.rolling(vol_short).mean() / dv.rolling(vol_long).mean()
    ok = (mom > 0) & (ratio > vol_ratio_min)
    dates = pd.DatetimeIndex(adj.index.to_series().resample("ME").last().dropna())
    w = pd.DataFrame(0.0, index=dates, columns=adj.columns)
    for d in dates:
        picks = mom.loc[d][ok.loc[d]].dropna().nlargest(top_n).index
        if len(picks):
            w.loc[d, picks] = 1.0 / len(picks)
    return w
```

Development result (EODHD data): `{'cagr': 0.0549, 'ann_ret': 0.0642, 'vol': 0.1463, 'sharpe': 0.4387, 'sortino': 0.6047, 'max_dd': -0.2606, 'ann_turnover': 11.6605, 'ann_cost': 0.0117, 'avg_cash': 0.2109, 'bench_cagr': 0.0847}`

5.49% CAGR, 0.4387 Sharpe, -26.06% max drawdown, vs. SPY's 8.47% CAGR over the same window -- deliberately a weaker starting point than an already-optimized result. Turnover 11.6605 (~1.17%/yr cost at 10bps), ~21.09% average cash. These numbers match the volume_mom benchmark exactly, confirming the manual strategy and shared engine agree.

**Note (our build):** our yfinance-based `volume_mom` benchmark produced `cagr: 0.0648, sharpe: 0.5003` for the identical logic/params on `dev` -- different from the EODHD-sourced 0.4387 here, consistent with normal vendor-to-vendor differences in adjusted-close/volume history discussed earlier in this session. Same strategy, same engine, different data source -- the story (volume filter trades Sharpe/CAGR for shallower drawdown, higher turnover) holds either way.

### Configure the Deep Agents Research Team

The deterministic layer is complete: strategies can only be tested through the fixed engine, every experiment is recorded, and the selection rule already defines promotion. Now the agent layer.

Three roles: **strategy-engineer** (implements and tests ideas), **research-critic** (challenges the evidence), **coordinator** (manages sequence, applies the selection rule). Deliberate separation -- the same agent shouldn't propose, evaluate its own work, and decide promotion.

#### 1. Set the Agent Roles and Boundaries

```python
load_dotenv(override=True)
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

MODEL_ID = "openai:gpt-5.6-terra"
WORKER = init_chat_model(MODEL_ID, reasoning={"effort": "low"})
MANAGER = init_chat_model(MODEL_ID, reasoning={"effort": "medium"})
```

Engineer gets lower reasoning effort (mainly implementation); coordinator and critic get higher effort (comparing evidence, challenging conclusions, research decisions).

**NOTE: `"openai:gpt-5.6-terra"` is not a real OpenAI model id.** When we build this for real, swap in an actual current model.

Strategy contract, shared by every version:

```python
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
```

Rules text (surfaces the same boundaries already implemented in Python):

```python
RULES = f"""
Layout: /strategies/vN.py, /results/, /reviews/, /registry.csv, /decisions.jsonl

Stage gates, enforced by the sweep tool:
vN cannot be swept until v(N-1) has successful runs, a review at /reviews/v(N-1).md,
and a decision recorded via record_decision. There is no way around this.

Hard limits: three versions; at most 12 configurations per version; one major
structural change per revision. Engine, universe, splits, benchmark and cost
convention are fixed. The holdout does not exist for you; never ask for it.

{{SELECTION_RULE}}

Fixed benchmarks, computed before any version was written:
{{BENCH_TEXT}}

Do not call ls, glob, grep or read_file unless told a specific file exists and you
need its contents.
"""
```

Strategy engineer:

```python
engineer = {{
    "name": "strategy-engineer",
    "description": "Writes strategy files and sweeps them through the fixed backtester in one batched call. Use for anything that creates code or produces metrics.",
    "system_prompt": f"""You implement strategies. You do not decide what to implement.
{{RULES}}{{CONTRACT}}
Procedure:
1. Write the strategy file with write_file.
2. Call sweep ONCE with the entire parameter grid as a JSON list. Never per configuration.
3. If a run errors, read the message, fix the file, call sweep again. Errors count
   against the budget.
4. Report back in under 200 words: filename, the returned table verbatim, and the one
   configuration you recommend with a one-line reason. Never paste code back.""",
    "tools": [sweep],
    "model": WORKER,
}}
```

Research critic:

```python
critic = {{
    "name": "research-critic",
    "description": "Reads a results table and returns exactly one evidence-backed weakness with one proposed structural change. Use after every version is swept.",
    "system_prompt": f"""You review results. You never write or edit strategy code.
{{RULES}}
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
        FilesystemPermission(operations=["read","write"], paths=["/**"], mode="allow"),
    ],
}}
```

Critic write access to `/strategies/**` is denied at the permission layer, not just by instruction.

#### 2. Create the Coordinator

```python
COORDINATOR = f"""You run a quantitative research process and are judged on the honesty
of the process, not on the returns.
{{RULES}}
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
```

`FilesystemBackend(root_dir=str(WS), virtual_mode=True)` gives the team a shared workspace exposed through agent-facing paths like `/strategies/v1.py`, mapped underneath to the real research directory. The whole v1->v2->v3 sequence runs inside one checkpointed thread via a `run(prompt)` helper that calls `agent.invoke(...)`.

Verification: `subagent models: gpt-5.6-terra gpt-5.6-terra` / worker replies `ok`. At this point the team has everything it needs: engineer implements/tests, critic challenges without changing code, coordinator advances only after test -> review -> decide.

---

## Stage 4: "Reproduce the Manual Baseline as v1" through start of "Let the Agents Revise the Strategy"

### Reproduce the Manual Baseline as v1

v1 doesn't introduce a new idea -- it checks whether the agent workflow can reproduce the manually-verified baseline, run predefined experiments, get an independent critic review, and record a decision before real revision begins. The only search is a **pre-registered nine-configuration grid** (3 momentum windows x 3 portfolio sizes) -- the engineer can't expand the search after seeing first results.

```python
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

_ = run(V1_BRIEF)
```

**Result:** engineer's winning config (highest validation Sharpe 0.542 of the nine): `{mom_window: 126, vol_short: 20, vol_long: 120, vol_ratio_min: 1.0, top_n: 2}`.

| Metric | V1 selected | Equal-weight B&H | Plain momentum |
|---|---:|---:|---:|
| Dev CAGR | 0.0550 | 0.0904 | 0.0750 |
| Dev Sharpe | 0.4240 | 0.5532 | 0.5363 |
| Dev max DD | -0.2760 | -0.5203 | -0.2817 |
| Dev ann turnover | 11.5890 | 0.0000 | 7.2798 |
| Val CAGR | 0.1000 | 0.1769 | 0.2051 |
| Val Sharpe | 0.5420 | 0.8697 | 0.9279 |
| Val max DD | -0.2950 | -0.3371 | -0.2901 |

V1 has a smaller dev drawdown than both benchmarks but underperforms both on CAGR/Sharpe (dev and val), and trades notably more than plain momentum. No incumbent exists yet, so all three gates are N/A -- v1 becomes champion by default.

**Critic's weakness:** the strategy underperformed benchmark risk-adjusted returns despite materially higher turnover; the 189-day variant looked best in dev but that edge didn't survive validation; 126-day variants were more consistent but still well below both fixed benchmarks in validation Sharpe.

**Critic's proposed structural change:** add a **dual-momentum market-regime rule** -- hold the existing relative-momentum portfolio only when broad-market (SPY) absolute momentum is positive, otherwise move to cash. A genuine structural change, not a parameter nudge.

**Decision:** v1 retained as champion (no incumbent to compare against); the regime-filter proposal approved as the v2 hypothesis, subject to the fixed gates.

Post-v1 verification (proves the stage gate's three required artifacts actually exist):
```python
print(pd.read_csv(REGISTRY).groupby(["version","status"]).size())
print("decisions:", [d["version"] for d in _decisions()])
assert (WS/"reviews"/"v1.md").exists(), "v1 review missing"
assert any(d["version"] == "v1" for d in _decisions()), "v1 decision missing"
print("v1 cycle complete")
```

### Let the Agents Revise the Strategy (start)

With v1 established as baseline champion, the loop moves beyond reproduction. Every new version must come from a weakness identified in the previous critic review, and a challenger only replaces the incumbent by passing all three fixed selection gates.

---

## Stage 5 (final): "Test the Market-Regime Filter in v2" through "Conclusion"

### Test the Market-Regime Filter in v2

v1 critic's proposal: keep cross-sectional momentum, but move to cash whenever SPY's absolute momentum is non-positive. Tested using v1's champion config (not a fresh param search) so the v2 comparison is about the regime filter itself, not more tuning.

```python
V2_BRIEF = """Review the critic's v1 proposal in /reviews/v1.md. If you approve it, have
strategy-engineer implement it as /strategies/v2.py. Do not modify v1.py.
Sweep at most 12 configurations in one call. Pass the table to research-critic for a
review written to /reviews/v2.md. Then apply the selection rule between v2 and the
current champion, state which of the three gates passed and which failed, and call
record_decision for v2. Do not proceed past that."""
_ = run(V2_BRIEF)
```

One config swept (v1's champion params). Result:

| Metric | V1 champion | V2 |
|---|---:|---:|
| Dev Sharpe | 0.424 (0.4235) | 0.611 (0.6110) |
| Dev max DD | -0.276 (-27.57%) | -0.190 (-18.99%) |
| Dev turnover | 11.589 | 9.814 |
| Val Sharpe | 0.542 (0.5424) | 0.321 (0.3207) |
| Val max DD | -0.295 | -0.295 |

**Gate 1 (val Sharpe not worse): FAILED** -- 0.321 vs 0.542.
**Gate 2 (val max DD within 2pp): passed** -- identical, 0.0pp difference.
**Gate 3 (dev turnover within +20%): passed** -- 9.814 < 11.589 x 1.20 = 13.907.

Dev looked like a major win across the board; validation reversed the story almost completely. **v2 fails gate 1 -> v1 remains champion**, despite the dramatically better dev numbers. This is the selection rule "earning its place" exactly as designed.

Critic's additional finding: v2 was tested at only **one configuration** -- the large dev improvement has zero neighboring-parameter robustness support. New proposal (not a parameter nudge): replace the binary dollar-volume eligibility filter with **volatility-scaled weights** among the selected momentum assets.

Verification: registry grouped by version/status, decisions list includes v2, `/reviews/v2.md` exists.

### Run the Final Revision in v3

v3 = keep v2's regime filter + remove binary volume filter + weight selected assets inversely to recent realized volatility. Swept top_n in (2, 3, 4). Must freeze the surviving champion (which may be v1, v2, or v3) immediately after.

```python
V3_BRIEF = """Implement the final approved revision as /strategies/v3.py. Do not modify
v1 or v2. Sweep at most 12 configurations in one call, get a critic review at
/reviews/v3.md, apply the selection rule, and call record_decision for v3.

Then write /strategies/frozen.json containing exactly:
{"version": "<champion version>", "params": {...}, "rationale": "..."}
where the version is whichever the selection rule says is champion, which may be v1 or
v2 rather than v3. After writing that file, stop."""
_ = run(V3_BRIEF)
```

Best v3 config: `top_n=3`, validation Sharpe **0.5377** -- extremely close to v1's **0.5424** (diff **0.0047**). v3 improves val drawdown (-0.2884 vs -0.2954) and cuts dev turnover (7.0480 vs 11.5888) -- **2 of 3 gates pass**. But gate 1 (val Sharpe not worse) fails by 0.0047 -- the rule was fixed before v3 existed and doesn't bend for "practically identical." **v1 remains final champion.**

`frozen.json`:
```json
{
    "version": "v1",
    "params": {"mom_window": 126, "vol_short": 20, "vol_long": 120, "vol_ratio_min": 1.0, "top_n": 2},
    "rationale": "V1 remains champion after v3 failed the required validation-Sharpe gate (0.538 versus v1's 0.542), although v3 passed the validation-drawdown and development-turnover gates."
}
```

### Freeze the Champion and Unlock the Holdout

Verification happens **outside the agent layer**, in the main research process -- deliberately, so the holdout boundary doesn't depend on agent behavior:

```python
frozen = json.loads((WS/"strategies"/"frozen.json").read_text())
assert len(_decisions()) == 3, f"expected 3 decisions, found {len(_decisions())}"
for v in ["v1","v2","v3"]:
    assert (WS/"reviews"/f"{v}.md").exists(), f"missing review for {v}"
    assert not pd.read_csv(REGISTRY).query(f"version=='{v}' and status=='ok'").empty, f"no runs for {v}"
print("all three cycles complete")

for field in ["adj_close","close","volume"]:
    DATA["holdout"][field].to_parquet(WS/"data"/f"holdout_{field}.parquet")   # <- unlock happens HERE

final = {}
for split in ["dev","val","holdout"]:
    res = run_isolated(WS/"strategies"/f"{frozen['version']}.py", frozen["params"], split)
    assert res["ok"], res["error"]
    final[split] = res["metrics"]
```

Only after all checks pass are the holdout parquet files written to `WS/"data"` (this is the actual unlock -- confirms our earlier read that the handbook keeps holdout off-disk until this exact moment, unlike our build which persisted it to `private/data/` from the start).

**Holdout result for frozen v1:** CAGR **13.98%**, Sharpe **0.7962**, max DD **-23.04%** -- beats SPY B&H, equal-weight B&H, plain momentum, and volume-momentum on the holdout period. Equal-weight B&H still wins on drawdown alone (-18.23%). Explicitly framed as one unseen evaluation of a precommitted strategy, not a second chance to pick a different one -- "it doesn't change what we learned before the holdout."

### Audit the Complete Research Trail

Final coordinator task, after freeze -- cannot change the strategy, explicitly told **not to defend the outcome**:

```python
REPORT_BRIEF = f"""The holdout has been run once and the strategy is frozen. Nothing can change now.
...
Cite run numbers from the registry. Do not defend the result."""
_ = run(REPORT_BRIEF)
```

**Two real weaknesses the audit surfaces:**
1. v2 tested a substantial regime change at only one configuration -- the dev improvement had almost no robustness evidence.
2. **A genuine bug in the critic itself**: the v3 review recommended replacing the binary volume filter with volatility scaling -- but v3 had *already done exactly that* (removed the filter, implemented inverse-volatility weighting). The critic's explanation sounded reasonable but didn't accurately describe the strategy it was reviewing.

Handbook's own framing: *"That's probably the strongest lesson from the audit. Separating agents by role is useful, but it doesn't guarantee that those agents understand the artifacts they're evaluating. Persisting the strategy code, experiment registry, reviews, and decisions gives us an independent record against which their reasoning can be checked."*

### Conclusion

Full pipeline: raw EODHD data -> fixed data boundaries -> deterministic backtester -> benchmarks -> experiment tracking -> 3 agent roles -> 3 strategy versions -> frozen champion -> 1 holdout test -> final audit.

Explicitly not a clean "AI kept improving the strategy" story: v2 looked much better in dev and failed validation; v3 missed v1 by 0.0047 Sharpe; the critic misunderstood the strategy it reviewed in v3. *"Weirdly, those messy parts are what made the experiment worth doing. They showed exactly why the controls around the agents matter."*

Final line: *"agents can be genuinely useful for generating and challenging research ideas. They just shouldn't get to control the evidence that decides whether those ideas survive."*
