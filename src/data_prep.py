import json
import time
import pandas as pd
import yfinance as yf

from src.config import (
    TICKERS, START, END, RAW, WS, PRIVATE, DATA_DIR, SPLITS, WARMUP_DAYS,
)


def fetch_and_cache(symbol: str) -> None:
    """Download one ticker's full history once, then reuse the cached file on every later run."""
    out = RAW / f"{symbol}.parquet"
    if out.exists():
        print(f"{symbol}: using cached file")
        return

    df = yf.download(symbol, start=START, end=END, auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.to_parquet(out)
    print(f"{symbol}: fetched {len(df)} rows")


def load_raw(symbol: str) -> pd.DataFrame:
    """Read one ticker's cached parquet back off disk."""
    df = pd.read_parquet(RAW / f"{symbol}.parquet")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def validate(symbol: str, df: pd.DataFrame) -> dict:
    """Same four checks the handbook runs before trusting any price series."""
    prices = df[["Close", "Adj Close"]]
    return {
        "symbol": symbol,
        "rows": len(df),
        "first": df.index[0].date(),
        "last": df.index[-1].date(),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "missing": int(df[["Close", "Adj Close", "Volume"]].isna().sum().sum()),
        "nonpositive_price": int((prices <= 0).sum().sum()),
        "zero_volume_days": int((df["Volume"] <= 0).sum()),
    }


def build_panels_and_splits() -> None:
    """Align all 9 tickers into shared panels, then cut dev/val/holdout.

    dev and val are written into the agent-visible workspace. holdout is written
    into private/ instead — never into workspace/ — so the boundary is real on
    disk, not just a rule we intend to follow later.
    """
    frames = {sym: load_raw(sym) for sym in TICKERS}

    adj_close = pd.concat({s: frames[s]["Adj Close"] for s in TICKERS}, axis=1).dropna()
    close = pd.concat({s: frames[s]["Close"] for s in TICKERS}, axis=1).loc[adj_close.index]
    volume = pd.concat({s: frames[s]["Volume"] for s in TICKERS}, axis=1).loc[adj_close.index]

    idx = adj_close.index
    split_starts = {}

    for name, (lo, hi) in SPLITS.items():
        pos = idx.searchsorted(pd.Timestamp(lo))
        first = idx[max(0, pos - WARMUP_DAYS)]
        keep = (idx >= first) & (idx <= pd.Timestamp(hi))

        out_dir = DATA_DIR if name != "holdout" else (PRIVATE / "data")
        out_dir.mkdir(parents=True, exist_ok=True)

        for field_name, panel in [("adj_close", adj_close), ("close", close), ("volume", volume)]:
            panel[keep].to_parquet(out_dir / f"{name}_{field_name}.parquet")

        split_starts[name] = lo
        print(f"{name}: {int(keep.sum())} rows (incl. warmup) -> {out_dir}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(split_starts, open(DATA_DIR / "splits.json", "w"))


def check_holdout_boundary() -> None:
    """Prove, by search, that no holdout file exists anywhere under the agent-visible workspace."""
    leaked = list(WS.rglob("holdout*"))
    if leaked:
        raise RuntimeError(f"holdout data leaked into workspace: {leaked}")
    print("holdout boundary ok: no holdout files under workspace/")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for sym in TICKERS:
        fetch_and_cache(sym)
        time.sleep(0.5)

    report = pd.DataFrame([validate(sym, load_raw(sym)) for sym in TICKERS])
    print()
    print(report.to_string(index=False))

    problems = report[["duplicate_dates", "missing", "nonpositive_price"]].sum().sum()
    if problems > 0:
        raise ValueError(f"data validation failed: {problems} total problems found, see table above")
    print("\nall 9 tickers passed validation\n")

    build_panels_and_splits()
    check_holdout_boundary()
