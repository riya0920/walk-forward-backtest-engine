"""Multi-day execution scheduling -- the answer the capacity table could not give.

`run_survivorship.py` prints `impact model SATURATED` once a single rebalance leg
reaches an entire day's volume. That is honest but it is not an answer: the
square-root impact model stops growing there, so every larger AUM reports the
same Sharpe, and the true statement is not "Sharpe -0.90" but "this trade cannot
be done in a day".

Which raises the question the table dodges: then how many days DOES it take, and
what does the delay cost? Splitting an order over N days trades one cost for
another and both are real:

    IMPACT falls, because impact is driven by participation and participation
    falls roughly as 1/N. Under a square-root law, per-day impact falls as
    1/sqrt(N) and the total paid over the schedule falls as sqrt(N)/N = 1/sqrt(N).

    TIMING RISK rises, because the unexecuted remainder is exposed to the market
    for longer. Its standard deviation grows as sqrt(days), and on average half
    the order is still outstanding, so the exposure scales as sigma*sqrt(N)/2.

Those two are NOT in the same units, and pretending otherwise is the mistake
this module exists to avoid. Impact is a cost, in basis points paid. Timing risk
is a standard deviation, in basis points of dispersion. Adding them requires a
price for risk -- the risk-aversion parameter lambda of Almgren-Chriss -- and
that parameter is a preference, not a measurement:

    total = impact_bps + lambda * timing_sigma_bps

A first draft of this module summed them with lambda implicitly 1 and announced
an interior minimum. There was no interior minimum: timing risk dominated at
every horizon and the optimum was always "as fast as the cap allows", which the
prose had already claimed was wrong. `run_risk_execution.py` now sweeps lambda
instead, because the honest finding is that the answer is a preference and the
schedule only has an interior optimum inside a narrow band of it.

WHAT THIS IS NOT. There is no intraday scheduling, no venue routing, no limit-
order model and no adverse-selection term. The horizon is days and the unit is
one day's volume. It answers "is this position executable, and at what cost",
which is the question capacity analysis needs, and nothing finer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ImpactModel:
    """Square-root market impact, the same law used in the capacity table."""
    fixed_bps: float = 3.0
    coefficient_bps: float = 120.0

    def bps(self, participation: float) -> float:
        p = max(0.0, min(participation, 1.0))
        return self.fixed_bps + self.coefficient_bps * math.sqrt(p)


@dataclass
class Schedule:
    days: int
    participation: float
    impact_bps: float
    timing_sigma_bps: float
    risk_aversion: float
    feasible: bool
    note: str = ""

    @property
    def total_bps(self) -> float:
        """Impact plus the PRICE of the timing risk, not plus the risk."""
        return self.impact_bps + self.risk_aversion * self.timing_sigma_bps


def schedule_cost(notional: float, daily_volume: float, days: int,
                  daily_vol: float, impact: ImpactModel,
                  max_participation: float = 0.20,
                  risk_aversion: float = 0.10) -> Schedule:
    """Cost of working `notional` over `days` at an even daily rate."""
    days = max(1, int(days))
    per_day = notional / days
    participation = per_day / max(daily_volume, 1.0)

    impact_bps = impact.bps(participation) * 1.0     # paid on every slice
    # Timing risk: the average outstanding fraction is (days-1)/(2*days) of the
    # order -- on day one everything is outstanding, on the last day nothing is.
    # Sigma over the horizon is daily_vol * sqrt(days).
    outstanding = (days - 1) / (2.0 * days)
    timing_bps = daily_vol * math.sqrt(days) * outstanding * 1e4

    feasible = participation <= max_participation
    note = "" if feasible else "{:.0%} of ADV -- above the {:.0%} cap".format(
        participation, max_participation)
    return Schedule(days=days, participation=participation,
                    impact_bps=impact_bps, timing_sigma_bps=timing_bps,
                    risk_aversion=risk_aversion, feasible=feasible, note=note)


def optimal_schedule(notional: float, daily_volume: float, daily_vol: float,
                     impact: ImpactModel | None = None,
                     max_days: int = 60, max_participation: float = 0.20,
                     risk_aversion: float = 0.10
                     ) -> tuple[Schedule, list[Schedule]]:
    """Cheapest feasible schedule, plus the whole curve so it can be plotted.

    If NO schedule inside `max_days` respects the participation cap, the honest
    answer is that the position does not fit -- so the cheapest infeasible
    schedule is returned with `feasible=False` rather than silently relaxing the
    cap. A capacity number produced by relaxing the constraint that produced it
    is not a capacity number.
    """
    impact = impact or ImpactModel()
    curve = [schedule_cost(notional, daily_volume, d, daily_vol, impact,
                           max_participation, risk_aversion)
             for d in range(1, max_days + 1)]
    feasible = [s for s in curve if s.feasible]
    best = min(feasible or curve, key=lambda s: s.total_bps)
    return best, curve


def days_to_execute(notional: float, daily_volume: float,
                    max_participation: float = 0.20) -> int:
    """Minimum days to work an order without breaching the participation cap."""
    return max(1, math.ceil(notional / (daily_volume * max_participation)))
