"""The out-of-sample window must be scored on every one of its bars.

Regression: walk_forward used to backtest each test window as a stand-alone
100-bar series. The engine's default 60-bar warm-up then left the first 60 test
bars untraded, the last bar could not fill, and the strategy could not read the
training-window prices just before the test -- prices that are in the past and
perfectly legal to use. At most 39 of 100 "out-of-sample" bars were traded, a
lookback-90 strategy could act on at most 9 of them, and the Sharpe was taken
over a series that was mostly padding zeros.

Neither failure is a look-ahead leak, so the leak-guard tests could not catch
it. These two tests pin the behaviour directly.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtest import CostModel
from engine.strategies import make_synthetic_prices
from run_audit import walk_forward

TRAIN, TEST = 400, 100


@pytest.fixture(scope="module")
def prices():
    return make_synthetic_prices()


def _windows(prices):
    start = 0
    while start + TRAIN + TEST <= len(prices):
        yield start
        start += TEST


def test_oos_window_is_traded_on_every_test_bar(prices):
    """An always-long strategy must earn the whole test window's price move.

    Entered at the open of the first test bar and held to the close of the last
    one, after paying one entry cost. The old code entered at bar 61 of the
    window and earned only the last 39 bars, so this equality failed for every
    window.
    """
    always_long = lambda _p: (lambda view: 1.0)
    wf = walk_forward(prices, always_long, "long", [0], TRAIN, TEST)
    cost = CostModel().total_bps / 10_000

    assert len(wf) == len(list(_windows(prices)))
    for row, start in zip(wf.itertuples(), _windows(prices)):
        first = start + TRAIN
        last = first + TEST - 1
        expected = (1 - cost) * prices["close"].iloc[last] / prices["open"].iloc[first] - 1
        assert row.oos_return == pytest.approx(expected, rel=1e-9), (
            "window starting {}: OOS return {:.4%} is not the test window's "
            "price move {:.4%} -- some test bars were not traded".format(
                row.window_start, row.oos_return, expected))


def test_oos_window_can_read_the_prices_before_it(prices):
    """A strategy that needs 300 bars of history must be able to trade from
    the first test bar, because the training window is in the past.

    The old code gave it at most 100 bars of history, so it never traded and
    every OOS return was exactly zero.
    """
    def needs_history(_p):
        return lambda view: 1.0 if len(view.history("close")) > 300 else 0.0

    wf = walk_forward(prices, needs_history, "history", [0], TRAIN, TEST)
    assert (wf.oos_return != 0).all(), (
        "a strategy that needs 300 bars of history never traded out of sample "
        "-- the test window cannot see the prices before it")
    assert (wf.turnover > 0).all()
