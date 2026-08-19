"""Quantify survivorship bias and capacity decay.

Two experiments, each producing one number that is hard to argue with:

  1. SURVIVORSHIP. Same strategy, same dates, same engine. Once on the universe
     as it was (delistings settled at their delisting return), once on only the
     names that survived to the end. The gap is the bias.

  2. CAPACITY. Same strategy at increasing AUM under a participation-based
     impact model. The AUM at which the edge disappears is the capacity, and a
     strategy without one is a strategy nobody has sized.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.portfolio import (ParticipationCost, momentum_rank, reversal_rank,
                              run_portfolio)
from engine.universe import make_universe


def main() -> None:
    uni = make_universe()
    s = uni.summary()

    print("=" * 78)
    print("UNIVERSE")
    print("-" * 78)
    print("names             : {}".format(s["total_names"]))
    print("survived to end   : {}".format(s["survived"]))
    print("delisted          : {}  ({:.1%} of the universe)".format(
        s["delisted"], s["delist_rate"]))
    for reason, n in sorted(s["by_reason"].items()):
        print("   {:<14} {:>3}".format(reason, n))
    print("\nDelisting returns are booked, not dropped: bankruptcy -100%,")
    print("compliance -55%, acquisition +18%, merger +5%. Dropping the row")
    print("instead would convert a total loss into 'no position', which is")
    print("most of survivorship bias in one line of code.")

    # ---- experiment 1: survivorship ---------------------------------------
    print("\n" + "=" * 78)
    print("1. SURVIVORSHIP BIAS")
    print("-" * 78)
    print("{:<22}{:>12}{:>14}{:>14}{:>10}".format(
        "run", "Sharpe", "total return", "max drawdown", "delists"))

    rows = {}
    for label, strat in (("momentum", momentum_rank), ("reversal", reversal_rank)):
        honest = run_portfolio(uni, strat)
        biased = run_portfolio(uni, strat, restrict_to=uni.survivors())
        rows[label] = (honest, biased)
        print("{:<22}{:>12.2f}{:>13.1%}{:>14.1%}{:>10}".format(
            label + " (as it was)", honest.sharpe(), honest.total_return(),
            honest.max_drawdown(), honest.delist_hits))
        print("{:<22}{:>12.2f}{:>13.1%}{:>14.1%}{:>10}".format(
            label + " (survivors)", biased.sharpe(), biased.total_return(),
            biased.max_drawdown(), biased.delist_hits))

    print("-" * 78)
    for label, (honest, biased) in rows.items():
        print("{:<12} survivorship inflation: Sharpe {:+.2f}, return {:+.1%}, "
              "drawdown {:+.1%}".format(
                  label, biased.sharpe() - honest.sharpe(),
                  biased.total_return() - honest.total_return(),
                  biased.max_drawdown() - honest.max_drawdown()))
    print("\nThe survivor run never holds a name that later dies, so it never")
    print("takes the loss. Note the drawdown column especially: survivorship")
    print("flatters risk more than it flatters return, and risk is what the")
    print("position size is set from.")

    # ---- experiment 2: capacity -------------------------------------------
    print("\n" + "=" * 78)
    print("2. CAPACITY UNDER PARTICIPATION-BASED IMPACT")
    print("-" * 78)
    print("Impact model: total_bps = 3 + 120 * sqrt(participation), where")
    print("participation = order notional / $5m daily volume. Square-root impact")
    print("is the standard shape; the point is that it is NOT linear in size.\n")
    print("{:>14}{:>12}{:>14}{:>12}  {}".format(
        "AUM", "Sharpe", "total return", "costs", "note"))
    prev_sharpe, crossover = None, None
    for aum in (1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 5e8):
        cost = ParticipationCost(portfolio_usd=aum)
        r = run_portfolio(uni, momentum_rank, costs=cost)
        # Participation is capped at 100% of a day's volume, so past the AUM
        # where one rebalance leg IS the whole day, the impact term stops
        # growing and every larger AUM prints an identical number.
        leg_notional = aum / 10        # top_n = 10, so one name is ~10% of book
        note = "impact model SATURATED" if leg_notional >= cost.daily_volume_usd else ""
        if prev_sharpe is not None and prev_sharpe > 0 >= r.sharpe() and crossover is None:
            crossover = aum
        prev_sharpe = r.sharpe()
        print("{:>14}{:>12.2f}{:>13.1%}{:>12.3f}  {}".format(
            "${:,.0f}".format(aum), r.sharpe(), r.total_return(), r.costs_paid, note))

    if crossover:
        print("\nCapacity: the edge crosses zero at about ${:,.0f}.".format(crossover))
    print("\nRead the SATURATED rows as a model limit, not a result. Once one")
    print("rebalance leg is the entire day's volume the impact term stops growing,")
    print("so every larger AUM reports the same Sharpe. The honest statement at")
    print("that size is not 'Sharpe -0.90' -- it is 'this trade cannot be executed")
    print("in a day'. Saying so properly needs multi-day execution scheduling,")
    print("which this engine does not have.")
    print("\nA strategy quoted without an AUM is a strategy nobody has sized.")
    print("=" * 78)


if __name__ == "__main__":
    main()
