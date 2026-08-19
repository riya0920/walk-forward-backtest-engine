"""Tests for the survivorship machinery.

The point of contention these pin down: it is easy to write a universe object
that *claims* point-in-time membership and quietly leaks the future.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.portfolio import ParticipationCost, momentum_rank, run_portfolio
from engine.universe import DELIST_REASONS, make_universe


@pytest.fixture(scope="module")
def uni():
    return make_universe()


def test_universe_actually_contains_deaths(uni):
    """A survivorship test on a universe where nothing dies proves nothing."""
    s = uni.summary()
    assert s["delisted"] > 0
    assert s["survived"] > 0
    assert "bankruptcy" in s["by_reason"]


def test_membership_is_point_in_time(uni):
    """A name must not be a member after it delists, or before it lists."""
    for ticker, listing in uni.listings.items():
        if listing.end is None:
            continue
        after = uni.index[uni.index > listing.end]
        if len(after):
            assert ticker not in uni.members_on(after[0]), ticker
        assert ticker in uni.members_on(listing.end), ticker


def test_membership_never_grows_from_future_knowledge(uni):
    """Every member on date D must have listed on or before D."""
    for d in (uni.index[100], uni.index[700], uni.index[-2]):
        for t in uni.members_on(d):
            assert uni.listings[t].start <= d


def test_bankruptcy_is_booked_at_total_loss(uni):
    """Dropping the row instead of booking -100% is most of survivorship bias
    in one line of code, so the convention is asserted rather than assumed."""
    assert DELIST_REASONS["bankruptcy"] == -1.00
    bankrupt = [t for t, l in uni.listings.items() if l.delist_reason == "bankruptcy"]
    assert bankrupt
    for t in bankrupt:
        reason, ret = uni.delist_event(t, uni.listings[t].end)
        assert reason == "bankruptcy" and ret == -1.00


def test_survivor_only_run_is_more_flattering(uni):
    """The direction of the bias is not a coin flip. Restricting to survivors
    must not make a strategy look WORSE -- if it does, the experiment is wired
    backwards."""
    honest = run_portfolio(uni, momentum_rank)
    biased = run_portfolio(uni, momentum_rank, restrict_to=uni.survivors())
    assert biased.delist_hits == 0, "the survivor run should never hold a dying name"
    assert honest.delist_hits > 0, "the honest run must actually take delisting hits"
    assert biased.total_return() >= honest.total_return()
    assert biased.max_drawdown() >= honest.max_drawdown()   # shallower = larger


def test_participation_cost_is_nonlinear_in_size():
    """Square-root impact: doubling size must cost MORE than double per unit,
    but less than quadratic. A linear model cannot show capacity decay."""
    c = ParticipationCost(daily_volume_usd=5_000_000, portfolio_usd=1_000_000)
    small = c.cost_bps(0.01)
    big = c.cost_bps(0.04)      # 4x the weight
    assert big > small
    assert big < 4 * small, "impact should be concave in size, not linear"


def test_capacity_decays_with_aum(uni):
    small = run_portfolio(uni, momentum_rank,
                          costs=ParticipationCost(portfolio_usd=1e6))
    large = run_portfolio(uni, momentum_rank,
                          costs=ParticipationCost(portfolio_usd=5e7))
    assert large.costs_paid > small.costs_paid
    assert large.sharpe() < small.sharpe()
