"""Intraday execution, on a volume curve measured rather than assumed.

`engine/execution.py` states its own limit: "There is no intraday scheduling, no
venue routing, no limit-order model and no adverse-selection term. The horizon is
days and the unit is one day's volume." This is the intraday half, and the first
decision was not to assume the shape of the day.

THE CURVE IS MEASURED. Assuming a U-shape and then demonstrating consequences of
the assumption would be circular -- the conclusion would be a property of the
assumed curve. `fetch_intraday.py` pulls 60 days of real 5-minute bars for the
same eight names the daily cache holds, and the profile below comes out of them:

    15:55  8.80% of the day's volume, in one five-minute bar   (6.9x uniform)
    09:30  6.97%                                               (5.4x uniform)
    13:50  0.70%  -- the midday trough                         (0.55x uniform)

A 12.6x spread between the busiest and quietest bucket of the day.

WHY THAT SHAPE MAKES A STATED PARTICIPATION RATE WRONG. "We traded at 5% of
volume" is a statement about the day. A TWAP schedule -- equal quantity per time
slice, the obvious default -- holds the QUANTITY constant while the volume under
it swings 12-fold, so the realised participation swings 12-fold the other way. At
the midday trough a nominal 5% schedule is participating at roughly 9%; into the
close it is participating at under 1%. The aggregate number is true and every
interval disagrees with it.

That matters because impact is driven by participation, not by quantity. Under
the square-root law the engine already uses, cost per slice goes as
sqrt(participation), so a schedule that concentrates participation into the
thinnest part of the day pays more than its average rate suggests. `twap_vs_vwap`
computes how much more on the measured curve rather than asserting it.

WHAT A VWAP SCHEDULE IS AND IS NOT, and this section was written wrong first.

Slicing in proportion to expected volume holds participation flat, which
minimises impact for a given order. The textbook cost is that it DEFERS
execution -- tracking the volume curve pushes the order into the close, leaving
the remainder exposed to timing risk for longer, which is the same trade
`execution.py` prices over days.

That is not what the measured curve does. Timing exposure -- the average
fraction of the order still outstanding across the day -- comes out at 0.484 for
VWAP against 0.494 for TWAP, and VWAP is 46.1% complete by noon where TWAP is
39.7%. VWAP is slightly LESS deferred, not more.

The reason is that a U is not a ramp. The textbook trade-off assumes a
back-loaded curve, and this one is front-loaded as well: the open auction at
6.97% roughly offsets the close at 8.80%, so following the curve gets an order
started sooner as well as finished later. On this profile VWAP is better on
impact and no worse on timing, and the trade-off simply is not there at this
granularity.

Where the trade-off IS real is with alpha. Neither schedule knows anything about
the next few hours; a signal that decays within the day is a reason to trade
faster than either, and that is a reason the timing-exposure number above cannot
see, because it prices dispersion rather than drift.

THE CLOSING AUCTION IS NOT CONTINUOUS TRADING, and the 15:55 bucket is mostly
it. An auction is a single price formed once, so "participating at 5% of the
auction" is a different act from participating in continuous flow -- you submit
into it and find out. Treating that bucket as more of the same overstates how
much of the schedule can actually be placed there, and this module flags the
share of a schedule that lands in it rather than pretending otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data" / "intraday_5m.parquet"

# Buckets treated as auction rather than continuous flow. The US close auction
# prints in the final bar; the open auction prints in the first.
AUCTION_BUCKETS = ("09:30", "15:55")


def load_profile(path: Path | str = CACHE, ticker: str | None = None
                 ) -> pd.Series:
    """Average share of daily volume by time of day, from real bars.

    Computed as a share WITHIN each ticker-day and then averaged across days, not
    as total volume per bucket over total volume. The difference matters: summing
    raw volume lets the highest-volume ticker and the highest-volume days
    dominate the shape, so the curve would describe SPY on a busy week rather
    than the typical day.
    """
    df = pd.read_parquet(path)
    if ticker:
        df = df[df.ticker == ticker]
    df = df.copy()
    df["tod"] = df.index.strftime("%H:%M")
    df["day"] = df.index.date
    total = df.groupby(["ticker", "day"])["volume"].transform("sum")
    df["share"] = df.volume / total
    prof = df.groupby("tod")["share"].mean()
    return prof / prof.sum()


@dataclass
class IntradaySchedule:
    name: str
    quantity: np.ndarray          # shares per bucket
    participation: np.ndarray     # fraction of that bucket's volume
    impact_bps: float
    auction_share: float
    max_participation: float
    mean_participation: float


def _impact_bps(participation: np.ndarray, quantity: np.ndarray,
                coefficient: float = 100.0) -> float:
    """Quantity-weighted square-root impact, in basis points.

    Same functional form as `engine/execution.ImpactModel` so the intraday and
    multi-day answers are commensurable. Weighted by quantity because impact is
    paid on the shares traded in each slice, not once per slice -- an unweighted
    average would let a tiny slice at high participation dominate a large one at
    low participation.
    """
    q = quantity.sum()
    if q <= 0:
        return 0.0
    return float((coefficient * np.sqrt(np.clip(participation, 0, None))
                  * quantity).sum() / q)


def build(profile: pd.Series, order_shares: float, daily_volume: float,
          style: str = "vwap", coefficient: float = 100.0) -> IntradaySchedule:
    """Slice an order across the day.

    style='twap'  equal quantity per bucket -- the obvious default
    style='vwap'  quantity proportional to expected volume, holding
                  participation flat
    """
    shares_by_bucket = profile.to_numpy() * daily_volume
    n = len(profile)

    if style == "twap":
        qty = np.full(n, order_shares / n)
    elif style == "vwap":
        qty = profile.to_numpy() * order_shares
    else:
        raise ValueError("style must be twap or vwap")

    part = np.divide(qty, shares_by_bucket,
                     out=np.zeros_like(qty), where=shares_by_bucket > 0)
    auction_mask = np.array([t in AUCTION_BUCKETS for t in profile.index])
    return IntradaySchedule(
        name=style,
        quantity=qty,
        participation=part,
        impact_bps=_impact_bps(part, qty, coefficient),
        auction_share=float(qty[auction_mask].sum() / order_shares)
        if order_shares else 0.0,
        max_participation=float(part.max()),
        mean_participation=float(order_shares / daily_volume)
        if daily_volume else 0.0,
    )


def twap_vs_vwap(profile: pd.Series, order_shares: float, daily_volume: float,
                 coefficient: float = 100.0) -> dict:
    """How much more does the obvious default cost on the measured curve?

    The answer is a property of the curve's dispersion and nothing else, which
    is why it is computed rather than quoted. On a flat curve the two schedules
    are identical; the penalty grows with how uneven the day is.
    """
    t = build(profile, order_shares, daily_volume, "twap", coefficient)
    v = build(profile, order_shares, daily_volume, "vwap", coefficient)
    return {
        "twap_impact_bps": t.impact_bps,
        "vwap_impact_bps": v.impact_bps,
        "excess_bps": t.impact_bps - v.impact_bps,
        "excess_pct": (t.impact_bps / v.impact_bps - 1) * 100
        if v.impact_bps else 0.0,
        "twap_max_participation": t.max_participation,
        "vwap_max_participation": v.max_participation,
        "stated_participation": t.mean_participation,
        # The gap between what a schedule CLAIMS and what its worst interval
        # actually does. A compliance limit expressed as "never exceed 10% of
        # volume" is breached by a schedule whose daily average is 5%.
        "twap_overshoot_ratio": t.max_participation / t.mean_participation
        if t.mean_participation else float("nan"),
    }


def curve_dispersion(profile: pd.Series) -> dict:
    """Summary of how uneven the day is. The input to everything above."""
    p = profile.to_numpy()
    uniform = 1.0 / len(p)
    return {
        "buckets": len(p),
        "uniform_share": uniform,
        "max_share": float(p.max()),
        "min_share": float(p.min()),
        "max_over_min": float(p.max() / p.min()),
        "max_over_uniform": float(p.max() / uniform),
        "min_over_uniform": float(p.min() / uniform),
        # Concentration of the day's volume. 1.0 means one bucket holds it all.
        "herfindahl": float((p ** 2).sum()),
        "effective_buckets": float(1.0 / (p ** 2).sum()),
    }


def completion_curve(sched: IntradaySchedule, profile: pd.Series) -> pd.Series:
    """Fraction of the order executed by the end of each bucket.

    The other half of the trade-off, and the reason VWAP is not simply better.
    Tracking the volume curve means deferring execution into the close, so the
    unexecuted remainder is exposed to the market for longer -- exactly the
    timing risk `engine/execution.py` prices over days, now inside one.
    """
    cum = np.cumsum(sched.quantity)
    total = cum[-1] if len(cum) else 0.0
    return pd.Series(cum / total if total else cum, index=profile.index)


def timing_exposure(sched: IntradaySchedule, profile: pd.Series) -> float:
    """Average fraction of the order still outstanding, across the day.

    A single number for how long the order sits unexecuted. `execution.py` uses
    the same idea over days, where on average half the order is outstanding
    under a uniform schedule -- so 0.5 is the TWAP benchmark and anything above
    it is deferral.
    """
    done = completion_curve(sched, profile).to_numpy()
    return float((1.0 - done).mean())
