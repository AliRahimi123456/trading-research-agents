import json

from src.registry import sweep, record_decision
from src.selection import best_of, select_champion

grid = [{"top_n": tn} for tn in (2, 3, 4)]
result = sweep("v3", json.dumps(grid), note="volatility-scaled weights, no volume filter")
print(result)

v1_best = best_of("v1")
v3_best = best_of("v3")
print("\nbest v3 config:")
print(v3_best)

champion, rationale = select_champion(v3_best, v1_best, "v3", "v1")
print(f"\nselection rule result: champion = {champion}")
print(rationale)



from src.config import WS

v1_params = '{"mom_window": 126, "vol_short": 20, "vol_long": 120, "vol_ratio_min": 1.0, "top_n": 2}'
print("\n", record_decision("v3", champion, rationale, v1_params))

import json as _json
frozen = {
    "version": champion,
    "params": _json.loads(v1_params),
    "rationale": rationale + f" (v3 val_sharpe {v3_best['val_sharpe']:.4f} vs v1's {v1_best['val_sharpe']:.4f})",
}
(WS / "strategies" / "frozen.json").write_text(_json.dumps(frozen, indent=2))
print("\nfrozen:", (WS / "strategies" / "frozen.json").read_text())
