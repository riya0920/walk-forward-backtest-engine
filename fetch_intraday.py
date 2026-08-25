"""Fetch real 5-minute bars so the intraday volume curve is measured.

    python fetch_intraday.py

`engine/execution.py` says outright that it has "no intraday scheduling ... The
horizon is days and the unit is one day's volume." Building the intraday half
needs an intraday volume profile, and the profile is the whole point: assuming a
U-shape and then demonstrating consequences of the assumption would be circular.

Yahoo serves 5-minute bars for roughly the last 60 days, which is short but real.
Cached to parquet so the analysis is reproducible offline and so a rerun does not
depend on what the API feels like returning today.

WHAT THIS SAMPLE CAN AND CANNOT SUPPORT. Sixty days of five-minute bars on a
handful of large caps is enough to measure the SHAPE of the intraday volume
curve, which is stable and well documented. It is not enough to say anything
about a particular day, an earnings date, an index rebalance, or a name that
does not trade like these. The shape is the input the execution model needs; the
sample is not a claim about anything else.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "intraday_5m.parquet"

# The same names as the daily cache, so the intraday work sits on the same
# universe rather than a conveniently liquid subset chosen afterwards.
TICKERS = ["SPY", "AAPL", "MSFT", "JNJ", "XOM", "JPM", "PG", "KO"]


def main() -> int:
    if CACHE.exists() and "--force" not in sys.argv:
        df = pd.read_parquet(CACHE)
        print("cached: {:,} rows, {} tickers, {} .. {}".format(
            len(df), df.ticker.nunique(), df.index.min(), df.index.max()))
        return 0

    import yfinance as yf
    warnings.filterwarnings("ignore")

    frames = []
    for t in TICKERS:
        raw = yf.download(t, period="60d", interval="5m", progress=False,
                          auto_adjust=False)
        if raw is None or not len(raw):
            print("  {}: no data returned".format(t))
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        out = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        out.columns = ["open", "high", "low", "close", "volume"]
        out["ticker"] = t
        frames.append(out)
        print("  {}: {:,} bars".format(t, len(out)))

    if not frames:
        print("nothing fetched")
        return 1

    df = pd.concat(frames)
    df = df[df.volume > 0]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    print("\nwrote {} -- {:,} rows, {} .. {}".format(
        CACHE.relative_to(ROOT), len(df), df.index.min(), df.index.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
