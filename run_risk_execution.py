"""Risk limits, sector neutrality, borrow costs, and the capacity table's missing row.

    python run_risk_execution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.execution import (ImpactModel, days_to_execute, optimal_schedule,
                              schedule_cost)
from engine.risk import RiskLimits, apply_limits, exposures, neutralise

SECTORS = {"AAA": "tech", "BBB": "tech", "CCC": "tech", "DDD": "energy",
           "EEE": "energy", "FFF": "financials", "GGG": "financials"}

RAW = {"AAA": 0.35, "BBB": 0.22, "CCC": 0.18, "DDD": -0.20,
       "EEE": -0.15, "FFF": 0.24, "GGG": -0.06}


def _line(t):
    print("\n" + "=" * 82)
    print(t)
    print("-" * 82)


def main() -> None:
    print("=" * 82)
    print("RISK LIMITS AND MULTI-DAY EXECUTION")
    print("=" * 82)

    _line("1. AN UNCONSTRAINED SIGNAL MEETS A MANDATE")
    limits = RiskLimits(max_name_weight=0.10, max_sector_weight=0.30,
                        max_gross=1.00, max_net=0.20)
    kept, bound = apply_limits(RAW, SECTORS, limits)

    print("{:<8}{:>12}{:>12}{:>12}".format("name", "raw", "constrained", "sector"))
    for n in RAW:
        print("{:<8}{:>12.3f}{:>12.3f}{:>12}".format(n, RAW[n], kept[n], SECTORS[n]))

    before, after = exposures(RAW, SECTORS), exposures(kept, SECTORS)
    print("\n{:<16}{:>14}{:>14}".format("exposure", "raw", "constrained"))
    for k in ("gross", "net", "long", "short", "max_name"):
        print("{:<16}{:>14.3f}{:>14.3f}".format(k, before[k], after[k]))
    print("\nsector net (raw -> constrained)")
    for s in before["sector_net"]:
        print("  {:<14}{:>10.3f}  ->{:>10.3f}".format(
            s, before["sector_net"][s], after["sector_net"][s]))
    print("\nconstraints that bound, in the order they bound: {}".format(
        " -> ".join(bound) if bound else "none"))
    print("\nThe order is a decision, not an implementation detail. A name cap is")
    print("a clip so one oversized position does not shrink every other one; a")
    print("sector cap scales inside the offending sector only; gross comes before")
    print("net because scaling for gross can only shrink net, while fixing net")
    print("first can be undone by the gross scale. Reorder these and you get a")
    print("different portfolio -- a defensible one, but a different one.")

    _line("2. SECTOR NEUTRALITY IS NOT THE SAME AS A SECTOR LIMIT")
    neutral, _ = apply_limits(neutralise(RAW, SECTORS), SECTORS, limits)
    ne = exposures(neutral, SECTORS)
    print("{:<14}{:>12}{:>12}".format("sector", "capped", "neutralised"))
    for s in after["sector_net"]:
        print("{:<14}{:>12.3f}{:>12.3f}".format(
            s, after["sector_net"][s], ne["sector_net"][s]))
    print("\nA cap bounds the bet. Neutralising removes it: every sector nets to")
    print("zero, so the book keeps its within-sector view and stops being paid")
    print("for the direction of the sector itself. It still HOLDS energy risk --")
    print("long the best names, short the worst -- and calling that 'no oil")
    print("exposure' is how a market-neutral book finds out it was long beta.")

    _line("3. WHAT THE SHORT SIDE COSTS")
    print("{:<26}{:>12}{:>16}{:>14}".format(
        "borrow assumption", "bps/yr", "annual cost", "on Sharpe 0.60"))
    short_notional = abs(exposures(kept)["short"])
    for label, lim in [
            ("general collateral", RiskLimits(borrow_bps_annual=50)),
            ("moderately tight", RiskLimits(borrow_bps_annual=300)),
            ("hard to borrow (DDD)", RiskLimits(borrow_bps_annual=50,
                                                hard_to_borrow_bps={"DDD": 2500}))]:
        annual = lim.borrow_cost(kept, days=252)
        # A 10% annual-vol book: Sharpe drag = cost / vol.
        print("{:<26}{:>12}{:>15.3%}{:>14.3f}".format(
            label, lim.borrow_bps_annual, annual, 0.60 - annual / 0.10))
    print("\nShort notional in the book: {:.1%}. Financing is charged on that".format(
        short_notional))
    print("only, not on gross -- in a market-neutral book the shorts fund the")
    print("longs, so a symmetric rate on gross double-counts. A single crowded")
    print("name at 2,500bps costs more than the entire rest of the book, which is")
    print("why the rate is per name and not a portfolio constant.")

    # ------------------------------------------------------------------- 4
    _line("4. THE CAPACITY TABLE'S MISSING ROW")
    adv = 5e6
    daily_vol = 0.011
    impact = ImpactModel()
    print("One leg = 10% of the book. ADV ${:,.0f}, daily vol {:.1%}, "
          "20% participation cap.".format(adv, daily_vol))
    print()
    print("{:>14}{:>15}{:>11}{:>13}{:>13}{:>15}".format(
        "AUM", "leg notional", "min days", "impact bps", "timing sd", "verdict"))
    for aum in (1e6, 1e7, 5e7, 1e8, 5e8, 1e9):
        leg = aum / 10
        need = days_to_execute(leg, adv)
        sched = schedule_cost(leg, adv, need, daily_vol, impact)
        verdict = "OK" if need <= 60 else "does not fit"
        print("{:>14}{:>15}{:>11}{:>13.1f}{:>13.1f}{:>15}".format(
            "${:,.0f}".format(aum), "${:,.0f}".format(leg), need,
            sched.impact_bps, sched.timing_sigma_bps, verdict))

    print("\nThat is what the SATURATED label was standing in for. At $1bn the")
    print("answer is not a worse Sharpe: one rebalance leg needs 100 trading days")
    print("at a 20% participation cap, which is longer than the strategy's own")
    print("holding period -- the position can never finish being built. Impact")
    print("bps stop being the interesting column at that point.")

    # ------------------------------------------------------------------- 5
    _line("5. THE OPTIMUM IS A PREFERENCE, NOT A MEASUREMENT")
    leg = 2.5e7
    print("Working ${:,.0f} at ${:,.0f} ADV. Impact falls as 1/sqrt(days);".format(
        leg, adv))
    print("timing risk is a standard DEVIATION that grows as sqrt(days). Those")
    print("are not the same unit, so the total depends on what a unit of risk is")
    print("worth -- lambda below. That number is a mandate, not a fact.")
    print()
    print("{:>10}{:>14}{:>14}{:>14}{:>18}".format(
        "lambda", "optimal days", "impact bps", "lambda*sd", "shape"))
    for lam in (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00):
        best, curve = optimal_schedule(leg, adv, daily_vol, impact,
                                       risk_aversion=lam)
        feas = [c for c in curve if c.feasible]
        if not feas:
            shape = "nothing feasible"
        elif best.days == feas[0].days:
            shape = "fastest allowed"
        elif best.days == feas[-1].days:
            shape = "slowest allowed"
        else:
            shape = "interior"
        print("{:>10.2f}{:>14}{:>14.1f}{:>14.1f}{:>18}".format(
            lam, best.days, best.impact_bps, lam * best.timing_sigma_bps, shape))

    print("\nAt lambda = 0 the answer is always 'as slowly as the venue allows'.")
    print("Above a small threshold it is always 'as fast as the cap allows'. The")
    print("interior optimum that execution papers draw exists only in the band")
    print("between, and nothing in the price data tells you where in that band")
    print("you sit. A first draft of this module summed the two terms directly")
    print("and announced an interior minimum; there was not one, because adding")
    print("a cost to a standard deviation is not an operation. Sweeping lambda is")
    print("what the numbers actually support.")

    # ------------------------------------------------------------------- 6
    _line("6. THE CURVE AT ONE LAMBDA")
    lam = 0.10
    best, curve = optimal_schedule(leg, adv, daily_vol, impact, risk_aversion=lam)
    print("lambda = {:.2f}".format(lam))
    print()
    print("{:>8}{:>16}{:>14}{:>14}{:>12}".format(
        "days", "participation", "impact bps", "lambda*sd", "total bps"))
    for sc in curve:
        if sc.days in (1, 2, 3, 5, 8, 13, 21, 34, 60) or sc.days == best.days:
            mark = "  <- cheapest" if sc.days == best.days else (
                "  infeasible" if not sc.feasible else "")
            print("{:>8}{:>15.1%}{:>14.1f}{:>14.1f}{:>12.1f}{}".format(
                sc.days, sc.participation, sc.impact_bps,
                lam * sc.timing_sigma_bps, sc.total_bps, mark))
    print("\nA backtest that rebalances instantly is standing on row one of this")
    print("table without paying for it, and row one is infeasible here.")
    print("=" * 82)


if __name__ == "__main__":
    main()
