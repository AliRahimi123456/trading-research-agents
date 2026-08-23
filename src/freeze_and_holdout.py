import json
import shutil

import pandas as pd

from src.config import WS, PRIVATE, DATA_DIR
from src.registry import REGISTRY, _decisions
from src.isolation import run_isolated
from src.benchmarks import benchmark_table

frozen = json.loads((WS / "strategies" / "frozen.json").read_text())
print("frozen:", frozen)

assert len(_decisions()) == 3, f"expected 3 decisions, found {len(_decisions())}"
reg = pd.read_csv(REGISTRY)
for v in ["v1", "v2", "v3"]:
    assert (WS / "reviews" / f"{v}.md").exists(), f"missing review for {v}"
    assert not reg.query(f"version=='{v}' and status=='ok'").empty, f"no successful runs for {v}"
print("all three cycles complete\n")

# prove the boundary held: no holdout file has ever reached the workspace until this line
assert not list(DATA_DIR.glob("holdout_*")), "holdout leaked into workspace before freeze!"

for field in ["adj_close", "close", "volume"]:
    shutil.copy(PRIVATE / "data" / f"holdout_{field}.parquet", DATA_DIR / f"holdout_{field}.parquet")
print("holdout unlocked into", DATA_DIR, "\n")

final = {}
for split in ["dev", "val", "holdout"]:
    res = run_isolated(WS / "strategies" / f"{frozen['version']}.py", frozen["params"], split)
    assert res["ok"], res["error"]
    final[split] = res["metrics"]

(WS / "results" / "holdout.json").write_text(json.dumps(final, indent=2))
print(json.dumps(final, indent=2))

bench_hold = benchmark_table("holdout")
cols = ["cagr", "sharpe", "sortino", "max_dd", "ann_turnover"]
print("\nholdout benchmarks:")
print(bench_hold[cols])
