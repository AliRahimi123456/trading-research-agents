import json

from src.registry import sweep, REGISTRY
from src.selection import best_of

grid = [{"mom_window": mw, "top_n": tn} for mw in (63, 126, 189) for tn in (2, 3, 4)]

result = sweep("v1", json.dumps(grid), note="pre-registered baseline grid")
print(result)

print("\nbest config by validation Sharpe:")
print(best_of("v1"))



from src.registry import record_decision

record_result = record_decision(
    "v1", "v1",
    "V1 is the initial champion; no incumbent exists yet, so it is selected by default.",
    '{"mom_window": 126, "vol_short": 20, "vol_long": 120, "vol_ratio_min": 1.0, "top_n": 2}',
)
print("\n", record_result)
