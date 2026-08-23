import pandas as pd

from src.config import WS
from src.registry import REGISTRY

SELECTION_RULE = """
# Version selection rule (fixed before any version was run)

A challenger replaces the incumbent champion only if it passes ALL THREE gates:

1. Validation Sharpe is not worse than the incumbent's
2. Validation max drawdown is within 2 percentage points of the incumbent's
3. Development annual turnover is no more than 20% above the incumbent's

Ties go to the incumbent. A newer version does not automatically replace an older one.
A higher development Sharpe is not sufficient and is not one of the gates.
"""
(WS / "SELECTION_RULE.md").write_text(SELECTION_RULE)


def select_champion(challenger, incumbent, name_c: str, name_i: str):
    if incumbent is None:
        return name_c, "no incumbent"
    checks = [
        ("validation Sharpe not worse", challenger["val_sharpe"] >= incumbent["val_sharpe"]),
        ("validation drawdown within 2pp", challenger["val_max_dd"] >= incumbent["val_max_dd"] - 0.02),
        ("turnover within +20%", challenger["dev_turnover"] <= incumbent["dev_turnover"] * 1.20),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        return name_i, "incumbent retained; challenger failed: " + "; ".join(failed)
    return name_c, "challenger passed all three gates"


def best_of(version: str):
    reg = pd.read_csv(REGISTRY)
    rows = reg[(reg.version == version) & (reg.status == "ok")]
    return None if rows.empty else rows.sort_values("val_sharpe", ascending=False).iloc[0]
