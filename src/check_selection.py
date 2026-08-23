from src.selection import select_champion

v1 = {"val_sharpe": 0.5424, "val_max_dd": -0.2954, "dev_turnover": 11.5888}
v2 = {"val_sharpe": 0.3207, "val_max_dd": -0.2954, "dev_turnover": 9.8139}
v3 = {"val_sharpe": 0.5377, "val_max_dd": -0.2884, "dev_turnover": 7.0480}

print("v2 vs v1:", select_champion(v2, v1, "v2", "v1"))
print("v3 vs v1:", select_champion(v3, v1, "v3", "v1"))
