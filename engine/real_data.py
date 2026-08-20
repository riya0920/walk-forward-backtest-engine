"""Real market data, with the survivorship warning stated up front.

**yfinance cannot give you a survivorship-free universe.** It returns data for
tickers that exist *today*. Ask it for the S&P 500 as of 2015 and you will get
today's members' histories, which is precisely the contaminated sample
`engine/universe.py` was written to demonstrate. Every "I backtested the S&P over
20 years with yfinance" result carries this bias whether or not the author knows.

So this module does two things and refuses to do a third:

  1. Fetches real prices for a named ticker list, cached to disk so the repo runs
     offline after the first call and results are reproducible.
  2. Reports, loudly, what the resulting sample can and cannot support.
  3. Does NOT pretend the result is survivorship-free. The delisted-universe
     experiment stays on generated data, because that is the only place this repo
     can actually observe a name dying.

Corporate actions: `auto_adjust=False` and the Adj Close column is used for
returns. Splits and dividends are already reflected there. Using raw Close would
put a 4:1 split through the engine as a -75% day, which momentum reads as a
crash and mean-reversion buys with both hands.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data" / "prices.parquet"

# A deliberately small, liquid, still-listed set. Every name here survived, which
# is the point being made rather than a selection I am hiding.
DEFAULT_TICKERS = ["SPY", "AAPL", "MSFT", "JNJ", "XOM", "JPM", "PG", "KO"]


class SurvivorshipWarning(UserWarning):
    pass


def load(tickers: list[str] | None = None, start: str = "2015-01-01",
         end: str = "2024-12-31", use_cache: bool = True) -> pd.DataFrame:
    """Returns a frame with (open, close) columns per ticker, split/dividend
    adjusted. Cached after the first fetch."""
    tickers = tickers or DEFAULT_TICKERS
    if use_cache and CACHE.exists():
        df = pd.read_parquet(CACHE)
        have = {c[1] for c in df.columns}
        if set(tickers).issubset(have):
            return df

    import yfinance as yf
    warnings.warn(
        "yfinance returns only CURRENTLY LISTED tickers. Any universe built from "
        "it is survivorship-biased by construction, and no amount of careful "
        "backtesting downstream removes that. See docs/BIAS_AUDIT.md.",
        SurvivorshipWarning, stacklevel=2)

    raw = yf.download(tickers, start=start, end=end, progress=False,
                      auto_adjust=False)
    if raw.empty:
        raise RuntimeError("no data returned; check network or ticker list")

    # Adjusted close for returns; the open is scaled by the same adjustment
    # factor so intraday open->close arithmetic stays consistent.
    adj = raw["Adj Close"]
    close = raw["Close"]
    factor = adj / close
    open_adj = raw["Open"] * factor

    frame = pd.concat({"close": adj, "open": open_adj}, axis=1).dropna()
    CACHE.parent.mkdir(exist_ok=True)
    frame.to_parquet(CACHE)
    return frame


def single(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one instrument as the (open, close, high, low) frame the
    single-instrument engine expects."""
    out = pd.DataFrame({
        "open": frame[("open", ticker)],
        "close": frame[("close", ticker)],
    }).dropna()
    out["high"] = out[["open", "close"]].max(axis=1) * 1.001
    out["low"] = out[["open", "close"]].min(axis=1) * 0.999
    return out


def describe(frame: pd.DataFrame) -> dict:
    tickers = sorted({c[1] for c in frame.columns})
    return {
        "tickers": tickers,
        "n_tickers": len(tickers),
        "start": str(frame.index[0].date()),
        "end": str(frame.index[-1].date()),
        "bars": len(frame),
        "survivorship_free": False,
        "caveat": ("every ticker in this sample is still listed today; names "
                   "that delisted during the window are absent and cannot be "
                   "recovered from this source"),
    }
