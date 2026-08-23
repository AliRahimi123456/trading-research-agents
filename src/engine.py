import pandas as pd

import numpy as np
import json

PERIODS = 252  # trading days per year
RF_ANNUAL = 0.0
MAR_ANNUAL = 0.0


def metrics(bt: pd.DataFrame, rf_annual: float = RF_ANNUAL, mar_annual: float = MAR_ANNUAL) -> dict:
    """Turn a day-by-day backtest() result into the numbers used to compare strategies."""
    r = bt["ret"]
    rf_d = (1 + rf_annual) ** (1 / PERIODS) - 1
    mar_d = (1 + mar_annual) ** (1 / PERIODS) - 1
    ex = r - rf_d

    eq = (1 + r).cumprod()
    yrs = len(r) / PERIODS
    sd = ex.std(ddof=1)
    downside = np.sqrt((np.minimum(r - mar_d, 0.0) ** 2).mean()) * np.sqrt(PERIODS)

    out = {
        "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
        "ann_ret": r.mean() * PERIODS,
        "vol": r.std(ddof=1) * np.sqrt(PERIODS),
        "sharpe": (ex.mean() / sd) * np.sqrt(PERIODS) if sd > 0 else 0.0,
        "sortino": (r.mean() * PERIODS - mar_annual) / downside if downside > 0 else 0.0,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "ann_turnover": bt["turnover"].sum() / yrs,
        "ann_cost": bt["cost"].sum() / yrs,
        "avg_cash": bt["cash"].mean(),
    }
    return {k: round(float(v), 4) for k, v in out.items()}



def backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """Turn target weights into daily portfolio returns, with a one-day execution lag."""
    scheduled = pd.Series(returns.index.isin(weights.index), index=returns.index, dtype=bool)
    w = weights.reindex(returns.index).ffill().shift(1).fillna(0.0)
    is_rebal = scheduled.shift(1, fill_value=False)


    held = pd.Series(0.0, index=returns.columns)
    rows = []

    for d in returns.index:
        target = w.loc[d] if is_rebal.loc[d] else held

        traded = float((target - held).abs().sum())
        cost = traded * cost_bps / 1e4

        r = returns.loc[d]
        gross = float((target *  r).sum())
        net = gross - cost
        rows.append((net, traded, cost, float(1.0 - target.sum())))


        denominator = 1.0 + gross 
        if denominator <= 0:
            raise RuntimeError(f"portfolio value went non-positive on {d}: gross return={gross}")
        held = (target * (1.0 + r)) / denominator 

    return pd.DataFrame(rows, index=returns.index, columns=["ret", "turnover", "cost", "cash"])


def load_split(data_dir, split: str) -> dict:
    """Load one split's panels from a given directory. No dependency on this project's
    config — this file gets copied standalone into an isolated sandbox later."""
    from pathlib import Path
    data_dir = Path(data_dir)
    adj_close = pd.read_parquet(data_dir / f"{split}_adj_close.parquet")
    close = pd.read_parquet(data_dir / f"{split}_close.parquet")
    volume = pd.read_parquet(data_dir / f"{split}_volume.parquet")
    returns = adj_close.pct_change().fillna(0.0)
    eval_start = pd.Timestamp(json.load(open(data_dir / "splits.json"))[split])
    return {"adj_close": adj_close, "close": close, "volume": volume,
            "returns": returns, "eval_start": eval_start}



if __name__ == "__main__":
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {"SPY": [0.0, 0.01, -0.005, 0.002], "XLK": [0.0, 0.02, 0.005, -0.001]},
        index=dates,
    )
    weights = pd.DataFrame({"SPY": [0.5], "XLK": [0.5]}, index=[dates[0]])
    bt = backtest(weights, returns, cost_bps=10.0)
    assert bt["turnover"].sum() == 1.0
    print("toy example: turnover check passed\n")

    from src.config import DATA_DIR
    dev = load_split(DATA_DIR, "dev")
    spy_weights = pd.DataFrame(
        {c: [1.0 if c == "SPY" else 0.0] for c in dev["adj_close"].columns},
        index=[dev["adj_close"].index[0]],
    )
    real_bt = backtest(spy_weights, dev["returns"], cost_bps=10.0)
    real_bt_scored = real_bt.loc[dev["eval_start"]:]
    print(f"SPY buy-and-hold, dev split (eval_start={dev['eval_start'].date()}):")
    print(metrics(real_bt_scored))


