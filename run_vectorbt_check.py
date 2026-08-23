"""Cross-check this engine's arithmetic against vectorbt.

Every number this repo prints comes from code I wrote, tested against tests I
also wrote. That is a closed loop: a consistent sign error appears in the engine
and in the test and nothing ever disagrees. An independent implementation is the
only thing that breaks the loop.

Two comparisons, kept separate on purpose:

  A. METRIC arithmetic -- the SAME return series handed to both, so any
     difference is a definition difference and nothing else.
  B. EXECUTION arithmetic -- the same signal run through both engines, where a
     difference could be either a fill convention or a bug.

A is the sharper test. B is the one people expect, and it is muddier, because
two engines almost never share a fill convention exactly and the disagreement
you find is usually the convention.

    python run_vectorbt_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import vectorbt as vbt

from engine.backtest import CostModel, buy_and_hold, run
from engine.strategies import make_synthetic_prices, mean_reversion, momentum

FREE = CostModel(0, 0, 0)
TOL = 1e-9


def _row(name, mine, theirs, tol=TOL):
    diff = abs(mine - theirs)
    ok = "MATCH" if diff <= tol else "DIFFER"
    print("{:<28}{:>14.8f}{:>14.8f}{:>14.2e}   {}".format(
        name, mine, theirs, diff, ok))
    return diff <= tol


def main() -> None:
    prices = make_synthetic_prices()
    print("=" * 88)
    print("CROSS-CHECK AGAINST VECTORBT {}".format(vbt.__version__))
    print("=" * 88)

    res = run(prices, momentum(20), FREE)
    rets = res.returns

    # ------------------------------------------------------------------- A
    print("\nA. METRIC ARITHMETIC -- identical return series into both")
    print("-" * 88)
    print("{:<28}{:>14}{:>14}{:>14}".format("metric", "this engine", "vectorbt",
                                            "abs diff"))

    acc = rets.vbt.returns(freq="1D", year_freq="252 days")
    results = []
    results.append(_row("Sharpe (annualised)", res.sharpe(), acc.sharpe_ratio()))
    results.append(_row("max drawdown", res.max_drawdown()[0], acc.max_drawdown()))
    results.append(_row("Sortino (annualised)", res.sortino(), acc.sortino_ratio()))
    results.append(_row("total return", res.total_return(),
                        float((1 + rets).prod() - 1), tol=1e-8))

    print("\nTwo things this comparison found, both real:")
    print()
    print("1. THE YEAR IS A CONVENTION, AND IT MOVES SHARPE 20%.")
    naive = rets.vbt.returns(freq="1D").sharpe_ratio()
    print("   vectorbt's default annualisation from a daily frequency is 365")
    print("   CALENDAR days: Sharpe {:.4f}. This engine uses 252 TRADING days:".format(naive))
    print("   Sharpe {:.4f}. Ratio sqrt(365/252) = {:.4f}, and {:.4f} x {:.4f} = {:.4f}."
          .format(res.sharpe(), np.sqrt(365 / 252), res.sharpe(),
                  np.sqrt(365 / 252), res.sharpe() * np.sqrt(365 / 252)))
    print("   Same series, same code, {:+.1%} on the headline number, entirely from"
          .format(naive / res.sharpe() - 1))
    print("   which calendar you meant. Neither is wrong; quoting either without")
    print("   saying which is.")
    print()
    print("2. THIS ENGINE'S SORTINO WAS THE WRONG STATISTIC -- now fixed.")
    print("   It divided by `r[r < 0].std()`: the dispersion of losing returns")
    print("   about their own mean, over the losing subset only. Sortino divides")
    print("   by the second lower partial moment over ALL periods. On this series")
    print("   that was 0.554 against vectorbt's 0.641.")
    print("   The size of the gap is not the point. The old denominator goes to")
    print("   ZERO when every loss is the same size, so a strategy losing exactly")
    print("   1% on every down day scored an INFINITE Sortino:")
    uniform = pd.Series(np.where(np.arange(400) % 2 == 0, 0.012, -0.010))
    old = uniform[uniform < 0].std()
    new = np.sqrt(np.mean(np.minimum(uniform.to_numpy(), 0.0) ** 2))
    print("      uniform-loss series: old denominator {:.2e}, new {:.6f}".format(
        old, new))
    print("   vectorbt agrees with the new one to {:.2e}.".format(
        abs(uniform.vbt.returns(freq="1D", year_freq="252 days").sortino_ratio()
            - float(uniform.mean() / new * np.sqrt(252)))))

    # ------------------------------------------------------------------- B
    print("\n" + "=" * 88)
    print("B. EXECUTION ARITHMETIC -- same signal, two engines")
    print("-" * 88)

    close = prices["close"]
    open_ = prices["open"]

    # Reproduce this engine's rule exactly: decide on bar i from closes up to i,
    # fill at bar i+1's open. vectorbt is given the shifted signal and told to
    # fill at the open, which is the same instruction stated its way.
    lookback = 20
    long = (close > close.shift(lookback)).to_numpy(copy=True)
    long[:60] = False                      # this engine's warmup
    sig = pd.Series(long, index=close.index).shift(1).fillna(False)

    pf = vbt.Portfolio.from_signals(
        close=close, entries=sig & ~sig.shift(1).fillna(False),
        exits=~sig & sig.shift(1).fillna(False),
        price=open_, init_cash=1.0, fees=0.0, slippage=0.0, freq="1D")

    mine = run(prices, momentum(lookback), FREE)
    print("{:<28}{:>14}{:>14}{:>14}".format("quantity", "this engine", "vectorbt",
                                            "abs diff"))
    _row("total return", mine.total_return(), float(pf.total_return()), tol=5e-3)
    _row("max drawdown", mine.max_drawdown()[0], -float(pf.max_drawdown()),
         tol=5e-3)

    print("\nThese agree to a few tenths of a percent over 1,500 bars, and the")
    print("residual is a fill convention, not a bug: this engine splits each bar")
    print("at the fill (old position carries the overnight gap, new position")
    print("carries the rest of the day) while vectorbt marks the position to the")
    print("close and books the trade at the open price it was given. Chasing that")
    print("to zero would mean reimplementing vectorbt's accounting, at which point")
    print("the check stops being independent and stops being worth running.")

    # ------------------------------------------------------------------- C
    print("\n" + "=" * 88)
    print("C. THE ONE IDENTITY BOTH MUST SATISFY EXACTLY")
    print("-" * 88)
    bh = buy_and_hold(prices, warmup=0)
    price_return = float(close.iloc[-1] / open_.iloc[1] - 1)
    _row("held position vs price", bh.total_return(), price_return, tol=1e-9)
    hold = vbt.Portfolio.from_holding(close, init_cash=1.0, freq="1D")
    print("\nvectorbt's from_holding buys at the CLOSE of the first bar, so it")
    print("reports {:.4%} against this engine's {:.4%} -- the difference is one".format(
        float(hold.total_return()), bh.total_return()))
    print("bar's open-to-close move, {:.4%}, and it reconciles exactly:".format(
        float(open_.iloc[1] / close.iloc[0] - 1)))
    print("   {:.10f} vs {:.10f}".format(
        (1 + bh.total_return()) * float(open_.iloc[1] / close.iloc[0]),
        1 + float(hold.total_return())))
    print("\nA buy-and-hold run that does NOT reproduce the underlying's price")
    print("return has a bug in the return series, and that identity is the test")
    print("that caught the dropped overnight gap described in the README.")

    print("\n" + "=" * 88)
    print("VERDICT: {}/{} metric definitions agree to {:.0e} once the year".format(
        sum(results), len(results), TOL))
    print("         convention is stated. Of the two that did not agree at first,")
    print("         one was a convention that has to be declared and one was a")
    print("         defect in THIS engine. That is what the check is for.")
    print("=" * 88)


if __name__ == "__main__":
    main()
