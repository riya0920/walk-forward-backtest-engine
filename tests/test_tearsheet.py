"""The blotter has to reassemble into the headline numbers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtest import CostModel, run
from engine.strategies import make_synthetic_prices, momentum
from engine.tearsheet import blotter, round_trips, tearsheet


@pytest.fixture(scope="module")
def result():
    return run(make_synthetic_prices(), momentum(20), CostModel())


def test_blotter_ties_to_the_summary(result):
    """If these do not tie, the summary describes a portfolio the fills did not
    build -- and the summary is what gets quoted."""
    b = blotter(result)
    assert b["traded"].sum() == pytest.approx(result.turnover, abs=1e-12)
    assert b["cost"].sum() == pytest.approx(result.costs_paid, abs=1e-12)


def test_every_fill_is_decided_before_it_prints(result):
    """The blotter is also a look-ahead check: a fill dated at or before its own
    decision bar is the leak the engine exists to prevent."""
    b = blotter(result)
    assert (b["bar"] > b["decided_at"]).all()


def test_round_trips_are_fewer_than_fills(result):
    """A round trip is entry plus exit. Reporting fills as trades doubles the
    count and halves the average size."""
    assert 0 < len(round_trips(result)) <= len(result.trades)


def test_a_flat_strategy_produces_no_fills():
    res = run(make_synthetic_prices(), lambda v: 0.0)
    assert res.trades == []
    assert blotter(res).empty
    assert res.turnover == 0


def test_tearsheet_renders_and_reports_its_reconciliation(result):
    doc = tearsheet(result)
    assert "Reconciliation" in doc and "Monthly returns" in doc
    assert "hit rate" in doc
