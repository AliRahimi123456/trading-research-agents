import json
import time

import pandas as pd
import matplotlib.pyplot as plt

from src.config import WS, STRATEGIES_DIR, RESULTS_DIR, REVIEWS_DIR
from src.isolation import run_isolated

REGISTRY = WS / "registry.csv"
DECISIONS = WS / "decisions.jsonl"
MAX_CONFIGS = 12
COLS = ["version", "run", "status", "params", "note", "dev_cagr", "dev_sharpe", "dev_sortino",
        "dev_max_dd", "dev_turnover", "val_cagr", "val_sharpe", "val_max_dd", "dev_cagr_20bps", "error"]


def _used(version: str) -> int:
    if not REGISTRY.exists():
        return 0
    return int((pd.read_csv(REGISTRY)["version"] == version).sum())


def _decisions() -> list:
    if not DECISIONS.exists():
        return []
    return [json.loads(line) for line in DECISIONS.read_text().splitlines() if line.strip()]


def _stage_ok(version: str) -> tuple:
    """vN cannot begin until v(N-1) is swept, reviewed and decided."""
    if not (version.startswith("v") and version[1:].isdigit()):
        return True, ""
    n = int(version[1:])
    if n <= 1:
        return True, ""
    prev = f"v{n-1}"
    if not REGISTRY.exists() or _used(prev) == 0:
        return False, f"stage gate: {prev} has no recorded runs. Complete {prev} first."
    reg = pd.read_csv(REGISTRY)
    if reg[(reg.version == prev) & (reg.status == "ok")].empty:
        return False, f"stage gate: {prev} has no successful runs."
    if not (REVIEWS_DIR / f"{prev}.md").exists():
        return False, f"stage gate: /reviews/{prev}.md does not exist. Get a critic review first."
    if not any(d["version"] == prev for d in _decisions()):
        return False, f"stage gate: no decision recorded for {prev}. Call record_decision first."
    return True, ""


def sweep(version: str, grid_json: str, note: str = "") -> str:
    """Backtest strategies/<version>.py over several parameter sets in ONE call."""
    ok, why = _stage_ok(version)
    if not ok:
        return f"error: {why}"
    used = _used(version)
    try:
        grid = json.loads(grid_json)
        if isinstance(grid, dict):
            grid = [grid]
    except Exception as e:
        return f"error: grid_json is not valid JSON ({e})"
    if used + len(grid) > MAX_CONFIGS:
        return f"error: budget. {used}/{MAX_CONFIGS} used on {version}, you asked for {len(grid)} more."
    path = STRATEGIES_DIR / f"{version}.py"
    if not path.exists():
        return f"error: {path.name} does not exist. Write it first."

    rows = []
    for i, params in enumerate(grid, start=used + 1):
        row = {"version": version, "run": i, "note": note,
               "params": json.dumps(params, separators=(",", ":"))}
        dev = run_isolated(path, params, "dev")
        if not dev["ok"]:
            row.update(status="error", error=dev["error"].strip().split("\n")[-1][:150])
            rows.append(row)
            continue
        val = run_isolated(path, params, "val")
        c20 = run_isolated(path, params, "dev", cost_bps=20.0)
        dm, vm = dev["metrics"], val["metrics"]
        row.update(status="ok", dev_cagr=dm["cagr"], dev_sharpe=dm["sharpe"],
                    dev_sortino=dm["sortino"], dev_max_dd=dm["max_dd"],
                    dev_turnover=dm["ann_turnover"], val_cagr=vm["cagr"],
                    val_sharpe=vm["sharpe"], val_max_dd=vm["max_dd"],
                    dev_cagr_20bps=c20["metrics"]["cagr"] if c20["ok"] else None)
        tag = f"{version}_run{i}"
        (RESULTS_DIR / f"{tag}.json").write_text(json.dumps({"params": params, "dev": dm, "val": vm}, indent=2))
        eq = pd.Series(dev["equity"], index=pd.to_datetime(dev["dates"]))
        plt.figure(figsize=(8, 3))
        plt.plot(eq)
        plt.yscale("log")
        plt.title(tag)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{tag}.png", dpi=90)
        plt.close("all")
        rows.append(row)

    df = pd.DataFrame(rows).reindex(columns=COLS)
    df.to_csv(REGISTRY, mode="a", header=not REGISTRY.exists(), index=False)
    out = df.drop(columns=["version", "note"]).round(3).dropna(axis=1, how="all")
    if "val_sharpe" in out:
        out = out.sort_values("val_sharpe", ascending=False, na_position="last")
    return out.to_csv(index=False)


def read_registry(version: str = "") -> str:
    """Every run recorded so far as CSV, accepted and rejected. Pass a version to filter."""
    if not REGISTRY.exists():
        return "empty"
    r = pd.read_csv(REGISTRY)
    if version:
        r = r[r["version"] == version]
    return r[["version", "run", "status", "params", "dev_sharpe", "dev_sortino",
              "dev_max_dd", "val_sharpe", "val_max_dd", "error"]].to_csv(index=False)


def record_decision(version: str, champion: str, rationale: str, params_json: str) -> str:
    """Record the approved outcome of a version. REQUIRED before the next version can be swept."""
    if any(d["version"] == version for d in _decisions()):
        return f"error: a decision for {version} already exists and cannot be overwritten."
    rec = {"version": version, "champion": champion, "rationale": rationale,
           "params": json.loads(params_json), "ts": time.time()}
    with DECISIONS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return f"recorded. champion is now {champion}"
