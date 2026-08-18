"""The planted-leak test: an engine that can catch fraud-against-yourself.

The threshold below is the whole design decision. On a random walk with costs, a
legitimate strategy's annualised Sharpe lives in roughly [-1, 1]. A strategy that
sees one bar ahead produces a Sharpe an order of magnitude higher. So
"impossible Sharpe" is a detectable event, and the harness asserts it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtest import CostModel, run
from engine.data import LookAheadError, PointInTimeView
from engine.strategies import (make_synthetic_prices, mean_reversion, momentum,
                               planted_leak)

IMPOSSIBLE_SHARPE = 3.0


@pytest.fixture(scope="module")
def prices():
    return make_synthetic_prices()


def test_view_refuses_data_after_the_cursor(prices):
    cursor = prices.index[100]
    view = PointInTimeView(prices, cursor)
    assert len(view.history("close")) == 101
    with pytest.raises(LookAheadError):
        view.at("close", prices.index[101])


def test_history_never_includes_the_future(prices):
    """Property: for every cursor, max(returned index) <= cursor."""
    for i in (5, 50, 500, 1400):
        view = PointInTimeView(prices, prices.index[i])
        assert len(view.history("close")) == i + 1
        assert view.last("close") == pytest.approx(prices["close"].iloc[i])


def test_planted_leak_is_caught_by_impossible_sharpe(prices):
    """Inject a future-peeking signal; the harness must flag it."""
    leaked = run(prices, planted_leak(horizon=1), CostModel(0, 0, 0))
    assert leaked.sharpe() > IMPOSSIBLE_SHARPE, (
        "the leak detector failed: a strategy that sees tomorrow produced a "
        "Sharpe of {:.2f}, which is not implausible enough to flag".format(
            leaked.sharpe()))


def test_honest_strategies_stay_in_the_plausible_band(prices):
    """The other half of the test. A detector that flags everything is useless;
    these must NOT trip the same threshold on data with no signal in it."""
    for name, strat in [("momentum", momentum()), ("mean_reversion", mean_reversion())]:
        res = run(prices, strat)
        assert abs(res.sharpe()) < IMPOSSIBLE_SHARPE, (
            "{} tripped the leak threshold on a random walk -- that is a bug in "
            "the engine, not an edge".format(name))


def test_fills_happen_at_the_next_open_not_at_the_signal_bar(prices):
    """Door #2: executing at the price that generated the signal.

    A strategy that goes long exactly when today closes up would be free money
    if filled at today's close. Filled at tomorrow's open on a random walk, it
    must not be.
    """
    def close_chaser(view):
        h = view.history("close", 2)
        return 1.0 if len(h) == 2 and h[-1] > h[0] else 0.0

    res = run(prices, close_chaser, CostModel(0, 0, 0))
    assert abs(res.sharpe()) < IMPOSSIBLE_SHARPE


def test_costs_reduce_returns_monotonically(prices):
    """Sanity: the cost model has to actually bite, and bite harder as it rises."""
    sharpes = [run(prices, momentum(), CostModel(0, 0, bps)).sharpe()
               for bps in (0, 5, 10, 20)]
    assert sharpes == sorted(sharpes, reverse=True)
