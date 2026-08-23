from pathlib import Path

# --- universe & date range ---
TICKERS = ["SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLV", "XLP", "XLY"]
START = "2004-01-01"
END = "2025-12-31"

# --- research splits ---
# dev:  strategies get written and iterated here
# val:  versions compete for promotion here
# holdout: never seen until the champion is frozen
SPLITS = {
    "dev":     ("2005-01-01", "2017-12-31"),
    "val":     ("2018-01-01", "2021-12-31"),
    "holdout": ("2022-01-01", "2025-12-31"),
}
WARMUP_DAYS = 250  # trading days of history before eval_start, for rolling indicators

# --- filesystem layout ---
ROOT = Path(__file__).resolve().parent.parent / "project"
RAW = ROOT / "raw_cache"
WS = ROOT / "workspace"
PRIVATE = ROOT / "private"

DATA_DIR = WS / "data"
STRATEGIES_DIR = WS / "strategies"
RESULTS_DIR = WS / "results"
REVIEWS_DIR = WS / "reviews"
