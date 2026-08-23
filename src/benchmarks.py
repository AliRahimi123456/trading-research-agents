import pandas as pd

from src.config import TICKERS, DATA_DIR
from src.engine import backtest, metrics, load_split


def bh_weights(data: dict, tickers: list) -> pd.DataFrame:
    """Buy-and-hold: one purchase, equal-weighted across the given tickers, never rebalanced."""
    adj = data["adj_close"]
    w = pd.DataFrame(0.0, index=[adj.index[0]], columns=adj.columns)
    w.loc[w.index[0], tickers] = 1.0 / len(tickers)
    return w


def plain_momentum(data: dict, mom_window: int = 126, top_n: int = 3) -> pd.DataFrame:
    """Monthly rebalance into the top-N tickers by trailing momentum, if positive."""
    adj = data["adj_close"]
    mom = adj.pct_change(mom_window)
    month_ends = pd.DatetimeIndex(adj.index.to_series().resample("ME").last().dropna())
    w = pd.DataFrame(0.0, index=month_ends, columns=adj.columns)
    for dt in month_ends:
        picks = mom.loc[dt][mom.loc[dt] > 0].dropna().nlargest(top_n).index
        if len(picks):
            w.loc[dt, picks] = 1.0 / len(picks)
    return w


def volume_momentum(data: dict, mom_window: int = 126, top_n: int = 3,
                     vol_short: int = 20, vol_long: int = 120, vol_ratio_min: float = 1.0) -> pd.DataFrame:
    """Same as plain_momentum, but only eligible when recent dollar volume is above its own longer average."""
    adj, cls, vol = data["adj_close"], data["close"], data["volume"]
    mom = adj.pct_change(mom_window)
    dollar_vol = cls * vol
    ratio = dollar_vol.rolling(vol_short).mean() / dollar_vol.rolling(vol_long).mean()
    eligible = (mom > 0) & (ratio > vol_ratio_min)
    month_ends = pd.DatetimeIndex(adj.index.to_series().resample("ME").last().dropna())
    w = pd.DataFrame(0.0, index=month_ends, columns=adj.columns)
    for dt in month_ends:
        picks = mom.loc[dt][eligible.loc[dt]].dropna().nlargest(top_n).index
        if len(picks):
            w.loc[dt, picks] = 1.0 / len(picks)
    return w


BENCHMARKS = {
    "spy_bh": lambda d: bh_weights(d, ["SPY"]),
    "ew_bh": lambda d: bh_weights(d, TICKERS),
    "plain_mom": plain_momentum,
    "volume_mom": volume_momentum,
}


def benchmark_table(split: str) -> pd.DataFrame:
    data = load_split(DATA_DIR, split)
    rows = {}
    for name, fn in BENCHMARKS.items():
        w = fn(data)
        bt = backtest(w, data["returns"], cost_bps=10.0)
        scored = bt.loc[data["eval_start"]:]
        rows[name] = metrics(scored)
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from src.config import WS

    cols = ["cagr", "sharpe", "sortino", "max_dd", "ann_turnover"]
    lines = ["# Fixed benchmarks\n", "Computed once, before any strategy version exists.\n"]
    for split in ["dev", "val"]:
        table = benchmark_table(split)[cols]
        print(f"\n{split.upper()}")
        print(table)
        lines.append(f"\n## {split.upper()}\n\n```\n{table.to_string()}\n```\n")

    (WS / "BENCHMARKS.md").write_text("\n".join(lines))
    print(f"\nwritten to {WS / 'BENCHMARKS.md'}")
