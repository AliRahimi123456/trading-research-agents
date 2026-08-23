import importlib.util

from src.config import STRATEGIES_DIR, DATA_DIR
from src.engine import backtest, metrics, load_split
from src.isolation import run_isolated

params = {"mom_window": 126, "top_n": 3, "vol_short": 20, "vol_long": 120, "vol_ratio_min": 1.0}
strategy_path = STRATEGIES_DIR / "parity_check.py"

# --- path 1: run it directly, in this process ---
spec = importlib.util.spec_from_file_location("strategy", strategy_path)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)

data = load_split(DATA_DIR, "dev")
w = strategy.target_weights(data, **params)
bt = backtest(w, data["returns"], cost_bps=10.0)
in_process_sharpe = metrics(bt.loc[data["eval_start"]:])["sharpe"]

# --- path 2: run the exact same file through the isolated sandbox ---
iso = run_isolated(strategy_path, params, "dev")

print("in-process sharpe:", in_process_sharpe)
print("isolated result:  ", iso)

assert iso["ok"], f"isolated run failed: {iso.get('error')}"
assert iso["metrics"]["sharpe"] == in_process_sharpe, "isolated and in-process disagree!"
print("\nparity check passed")
