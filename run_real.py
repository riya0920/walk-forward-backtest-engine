"""Run the engine on REAL prices, and check the leak detector still holds.

The point of this script is not the strategy results. It is that the same
harness, unchanged, runs on real data and the planted-leak test still trips --
because a detector calibrated only on synthetic data proves nothing about the
data you actually care about.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.backtest import CostModel, buy_and_hold, run
from engine.multiple_testing import assess
from engine.portfolio import ParticipationCost, momentum_rank, run_portfolio
from engine.real_data import SurvivorshipWarning, describe, load, single
from engine.strategies import mean_reversion, momentum, planted_leak

IMPOSSIBLE_SHARPE = 3.0


def main() -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SurvivorshipWarning)
        frame = load()
        surv = [w for w in caught if issubclass(w.category, SurvivorshipWarning)]

    meta = describe(frame)
    print("=" * 78)
    print("REAL DATA")
    print("-" * 78)
    print("tickers : {}".format(", ".join(meta["tickers"])))
    print("window  : {} to {}  ({:,} bars)".format(
        meta["start"], meta["end"], meta["bars"]))
    print("adjusted: split/dividend adjusted (Adj Close); the open is scaled by")
    print("          the same factor so open->close arithmetic stays consistent.")
    print("          Using raw Close would put a 4:1 split through the engine as")
    print("          a -75% day, which momentum reads as a crash.")
    if surv or True:
        print("\n!! SURVIVORSHIP: {}".format(meta["caveat"]))
        print("   yfinance returns only currently-listed tickers, so this sample")
        print("   cannot support a survivorship-free claim. The delisted-universe")
        print("   experiment stays on generated data in run_survivorship.py,")
        print("   because that is the only place a name can actually die here.")

    # ---- 1. the leak detector, on real prices -----------------------------
    spy = single(frame, "SPY")
    leak = run(spy, planted_leak(1), CostModel(0, 0, 0))
    mom = run(spy, momentum())
    rev = run(spy, mean_reversion())

    print("\n" + "=" * 78)
    print("1. PLANTED-LEAK TEST ON REAL DATA")
    print("-" * 78)
    print("peeks one bar ahead : Sharpe {:8.2f}   <- must exceed {:.1f}".format(
        leak.sharpe(), IMPOSSIBLE_SHARPE))
    print("momentum (honest)   : Sharpe {:8.2f}".format(mom.sharpe()))
    print("mean reversion      : Sharpe {:8.2f}".format(rev.sharpe()))
    ok = leak.sharpe() > IMPOSSIBLE_SHARPE and abs(mom.sharpe()) < IMPOSSIBLE_SHARPE
    print("\ndetector holds on real data: {}".format(ok))
    print("A leak detector calibrated only on synthetic data proves nothing about")
    print("the data you care about, which is why this runs on both.")

    # ---- 2. benchmark-relative --------------------------------------------
    bh = buy_and_hold(spy)
    print("\n" + "=" * 78)
    print("2. SPY, {} to {}".format(meta["start"], meta["end"]))
    print("-" * 78)
    print("{:<22}{:>10}{:>14}{:>14}{:>12}".format(
        "strategy", "Sharpe", "total return", "max drawdown", "turnover"))
    for name, r in (("buy and hold", bh), ("momentum(20)", mom),
                    ("mean reversion(20)", rev)):
        dd, _ = r.max_drawdown()
        print("{:<22}{:>10.2f}{:>13.1%}{:>14.1%}{:>12.1f}".format(
            name, r.sharpe(), r.total_return(), dd, r.turnover))
    print("\nBoth strategies are test cargo. On real data over a decade that")
    print("included a bull market, buy-and-hold is the honest benchmark and")
    print("beating it is the only result that would mean anything.")

    # ---- 3. cross-sectional on the real universe --------------------------
    print("\n" + "=" * 78)
    print("3. CROSS-SECTIONAL MOMENTUM ON THE REAL (SURVIVOR-ONLY) UNIVERSE")
    print("-" * 78)
    from engine.universe import Listing, Universe
    import pandas as pd

    tickers = [t for t in meta["tickers"] if t != "SPY"]
    sub = pd.concat({"close": frame["close"][tickers],
                     "open": frame["open"][tickers]}, axis=1)
    listings = [Listing(t, frame.index[0], None, None, 0.0) for t in tickers]
    uni = Universe(sub, listings)

    for aum in (1e6, 1e7, 1e8):
        r = run_portfolio(uni, momentum_rank, top_n=3,
                          costs=ParticipationCost(portfolio_usd=aum,
                                                  daily_volume_usd=5e8))
        print("AUM {:>12}  Sharpe {:>6.2f}  return {:>8.1%}  turnover {:>7.1f}".format(
            "${:,.0f}".format(aum), r.sharpe(), r.total_return(), r.turnover))
    print("\nEvery name here survived to today. Read these as an engine")
    print("demonstration, not as evidence about cross-sectional momentum.")

    print("\n" + "=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
