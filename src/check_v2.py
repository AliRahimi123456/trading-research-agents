import json

from src.registry import sweep
from src.selection import best_of

params = {"mom_window": 126, "vol_short": 20, "vol_long": 120, "vol_ratio_min": 1.0, "top_n": 2}
result = sweep("v2", json.dumps([params]), note="regime filter, v1 champion params")
print(result)
print("\nbest v2 config:")
print(best_of("v2"))




from src.selection import select_champion, best_of

v1_best = best_of("v1")
v2_best = best_of("v2")
champion, rationale = select_champion(v2_best, v1_best, "v2", "v1")
print(f"\nselection rule result: champion = {champion}")
print(rationale)

from src.registry import record_decision

v1_params = '{"mom_window": 126, "vol_short": 20, "vol_long": 120, "vol_ratio_min": 1.0, "top_n": 2}'
print("\n", record_decision("v2", champion, rationale, v1_params))
