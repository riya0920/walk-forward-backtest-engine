"""A universe with delistings, so survivorship bias can be MEASURED rather than
apologised for.

The bias in one sentence: if you build today's ticker list and backtest it over
the last ten years, every company that went bankrupt, got acquired at a
discount, or was delisted for non-compliance is invisible. Your sample is the
winners. Returns are inflated, drawdowns are shallow, and the effect is largest
in exactly the strategies that buy distress.

Most backtests handle this by not mentioning it. This module handles it by
generating a universe where names DIE -- with a delisting reason and a terminal
return -- and then running the same strategy twice:

    as_it_was    the universe an investor could actually have traded on each
                 date, including names that later delisted
    survivors    only names still listed at the end of the sample

The gap between those two runs is the survivorship bias, in Sharpe and in
return, for that strategy on that data. That number is the deliverable.

Delisting return conventions matter and are stated: a bankruptcy is booked at
-100%, an acquisition at a premium, a compliance delisting at a haircut. Setting
delisting returns to 0 (or dropping the row, which is the same thing) is the
single most common way survivorship bias sneaks back in after you thought you
had removed it -- see Shumway (1997) on CRSP delisting returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DELIST_REASONS = {
    "bankruptcy": -1.00,        # equity wiped
    "compliance": -0.55,        # forced off the exchange at a discount
    "acquisition": +0.18,       # taken out at a premium
    "merger": +0.05,
}


@dataclass
class Listing:
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp | None          # None = still listed at sample end
    delist_reason: str | None
    delist_return: float


class Universe:
    """Point-in-time membership. `members_on(date)` returns only names that were
    actually listed then -- no forward knowledge of who survives."""

    def __init__(self, prices: pd.DataFrame, listings: list[Listing]):
        self.prices = prices
        self.listings = {l.ticker: l for l in listings}
        self.index = prices.index

    def members_on(self, date: pd.Timestamp) -> list[str]:
        return [t for t, l in self.listings.items()
                if l.start <= date and (l.end is None or date <= l.end)]

    def survivors(self) -> list[str]:
        """The look-ahead-contaminated view: names still listed at the end."""
        return [t for t, l in self.listings.items() if l.end is None]

    def delisted(self) -> list[str]:
        return [t for t, l in self.listings.items() if l.end is not None]

    def delist_event(self, ticker: str, date: pd.Timestamp):
        l = self.listings[ticker]
        if l.end is not None and date == l.end:
            return l.delist_reason, l.delist_return
        return None, 0.0

    def summary(self) -> dict:
        by_reason = {}
        for l in self.listings.values():
            if l.delist_reason:
                by_reason[l.delist_reason] = by_reason.get(l.delist_reason, 0) + 1
        return {
            "total_names": len(self.listings),
            "survived": len(self.survivors()),
            "delisted": len(self.delisted()),
            "delist_rate": len(self.delisted()) / len(self.listings),
            "by_reason": by_reason,
        }


def make_universe(n_names: int = 60, n_bars: int = 1500, seed: int = 11,
                  annual_delist_rate: float = 0.06) -> Universe:
    """Random-walk names, some of which die.

    Names that will delist are given a downward drift beforehand for the
    distress reasons -- a company does not go from healthy to bankrupt in one
    bar. That drift is what a momentum strategy sees and a survivor-only sample
    never contains, and it is why the bias is not a small correction.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_bars)
    bars_per_year = 252
    listings: list[Listing] = []
    closes, opens = {}, {}

    for i in range(n_names):
        ticker = "N{:03d}".format(i)
        p_delist = 1 - (1 - annual_delist_rate) ** (n_bars / bars_per_year)
        will_delist = rng.random() < p_delist

        if will_delist:
            end_i = int(rng.integers(int(n_bars * 0.15), n_bars - 1))
            reason = rng.choice(list(DELIST_REASONS),
                                p=[0.35, 0.25, 0.28, 0.12])
        else:
            end_i, reason = n_bars - 1, None

        mu = 0.0002
        sigma = 0.013 + 0.004 * rng.random()
        rets = rng.normal(mu, sigma, n_bars)

        if reason in ("bankruptcy", "compliance"):
            # Distress builds over the final ~120 bars.
            decay_start = max(0, end_i - 120)
            ramp = np.linspace(0, 1, end_i - decay_start + 1)
            rets[decay_start:end_i + 1] -= 0.0035 * ramp

        close = 100 * np.exp(np.cumsum(rets))
        gap = rng.normal(0, sigma / 3, n_bars)
        open_ = close * np.exp(-rets + gap)

        closes[ticker] = pd.Series(close, index=idx)
        opens[ticker] = pd.Series(open_, index=idx)
        listings.append(Listing(
            ticker=ticker,
            start=idx[0],
            end=None if reason is None else idx[end_i],
            delist_reason=reason,
            delist_return=DELIST_REASONS[reason] if reason else 0.0,
        ))

    prices = pd.concat({"close": pd.DataFrame(closes),
                        "open": pd.DataFrame(opens)}, axis=1)
    return Universe(prices, listings)
