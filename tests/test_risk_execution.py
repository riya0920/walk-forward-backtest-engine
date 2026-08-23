"""Risk limits and execution scheduling."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.execution import (ImpactModel, days_to_execute, optimal_schedule,
                              schedule_cost)
from engine.risk import RiskLimits, apply_limits, exposures, neutralise

SEC = {"A": "tech", "B": "tech", "C": "energy", "D": "energy"}


# ------------------------------------------------------------------- limits
def test_name_cap_clips_only_the_offender():
    w, bound = apply_limits({"A": 0.40, "B": 0.05}, None,
                            RiskLimits(max_name_weight=0.10, max_gross=10,
                                       max_net=10))
    assert w["A"] == pytest.approx(0.10)
    assert w["B"] == pytest.approx(0.05), "an innocent position was rescaled"
    assert any("name" in b for b in bound)


def test_sector_cap_scales_inside_the_sector_only():
    lim = RiskLimits(max_name_weight=1.0, max_sector_weight=0.30,
                     max_gross=10, max_net=10)
    w, bound = apply_limits({"A": 0.30, "B": 0.30, "C": 0.10, "D": 0.05},
                            SEC, lim)
    assert w["A"] + w["B"] == pytest.approx(0.30)
    assert w["C"] == pytest.approx(0.10) and w["D"] == pytest.approx(0.05)
    assert any("sector" in b for b in bound)


def test_gross_and_net_caps_both_hold_after_the_pass():
    lim = RiskLimits(max_name_weight=1.0, max_sector_weight=10.0,
                     max_gross=1.0, max_net=0.2)
    w, _ = apply_limits({"A": 0.9, "B": 0.8, "C": -0.1, "D": 0.1}, SEC, lim)
    e = exposures(w)
    assert e["gross"] <= 1.0 + 1e-9
    assert abs(e["net"]) <= 0.2 + 1e-9


def test_a_book_already_inside_every_limit_is_untouched():
    lim = RiskLimits()
    raw = {"A": 0.05, "B": -0.05}
    w, bound = apply_limits(raw, SEC, lim)
    assert bound == []
    assert w == pytest.approx(raw)


def test_the_order_the_constraints_bind_is_reported():
    lim = RiskLimits(max_name_weight=0.10, max_sector_weight=0.15, max_gross=0.25)
    _, bound = apply_limits({"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5}, SEC, lim)
    assert bound[0].startswith("name")
    assert any(b.startswith("sector") for b in bound)


# ------------------------------------------------------------- neutrality
def test_neutralising_nets_every_sector_to_zero():
    n = neutralise({"A": 0.3, "B": 0.1, "C": -0.2, "D": 0.4}, SEC)
    for sec in set(SEC.values()):
        assert sum(v for k, v in n.items() if SEC[k] == sec) == pytest.approx(0)


def test_neutralising_keeps_the_within_sector_ranking():
    raw = {"A": 0.3, "B": 0.1, "C": -0.2, "D": 0.4}
    n = neutralise(raw, SEC)
    assert (n["A"] > n["B"]) == (raw["A"] > raw["B"])


# ------------------------------------------------------------------ borrow
def test_borrow_is_charged_on_shorts_only():
    lim = RiskLimits(borrow_bps_annual=100)
    assert lim.borrow_cost({"A": 0.5}, days=252) == 0.0
    assert lim.borrow_cost({"A": -0.5}, days=252) == pytest.approx(0.005)


def test_a_hard_to_borrow_name_uses_its_own_rate():
    lim = RiskLimits(borrow_bps_annual=50, hard_to_borrow_bps={"C": 2500})
    cheap = lim.borrow_cost({"A": -0.1}, days=252)
    dear = lim.borrow_cost({"C": -0.1}, days=252)
    assert dear == pytest.approx(cheap * 50)


# --------------------------------------------------------------- execution
def test_impact_falls_as_the_order_is_spread_out():
    a = schedule_cost(1e7, 5e6, 1, 0.01, ImpactModel())
    b = schedule_cost(1e7, 5e6, 20, 0.01, ImpactModel())
    assert b.impact_bps < a.impact_bps


def test_timing_risk_rises_as_the_order_is_spread_out():
    a = schedule_cost(1e7, 5e6, 2, 0.01, ImpactModel())
    b = schedule_cost(1e7, 5e6, 20, 0.01, ImpactModel())
    assert b.timing_sigma_bps > a.timing_sigma_bps


def test_a_single_day_order_carries_no_timing_risk():
    assert schedule_cost(1e6, 5e6, 1, 0.01, ImpactModel()).timing_sigma_bps == 0


def test_days_to_execute_respects_the_participation_cap():
    # $10m against $5m ADV at 20% participation = $1m/day = 10 days.
    assert days_to_execute(1e7, 5e6, 0.20) == 10


def test_a_schedule_over_the_participation_cap_is_marked_infeasible():
    s = schedule_cost(1e8, 5e6, 1, 0.01, ImpactModel(), max_participation=0.20)
    assert not s.feasible and "ADV" in s.note


def test_risk_aversion_zero_always_wants_the_slowest_schedule():
    """With no price on risk there is no reason to hurry, and a model that
    still picks an interior day is pricing something it has not declared."""
    best, curve = optimal_schedule(2.5e7, 5e6, 0.011, max_days=60,
                                   risk_aversion=0.0)
    assert best.days == max(c.days for c in curve if c.feasible)


def test_high_risk_aversion_always_wants_the_fastest_feasible_schedule():
    best, curve = optimal_schedule(2.5e7, 5e6, 0.011, max_days=60,
                                   risk_aversion=5.0)
    assert best.days == min(c.days for c in curve if c.feasible)


def test_the_interior_optimum_exists_only_in_a_band_of_lambda():
    """Measured: slowest at lambda<=0.05, interior at 0.10, fastest at >=0.20."""
    def shape(lam):
        best, curve = optimal_schedule(2.5e7, 5e6, 0.011, max_days=60,
                                       risk_aversion=lam)
        feas = [c for c in curve if c.feasible]
        if best.days == feas[0].days:
            return "fast"
        return "slow" if best.days == feas[-1].days else "interior"

    assert shape(0.0) == "slow"
    assert shape(0.10) == "interior"
    assert shape(0.50) == "fast"


def test_total_cost_prices_risk_rather_than_adding_it():
    s = schedule_cost(1e7, 5e6, 10, 0.01, ImpactModel(), risk_aversion=0.25)
    assert s.total_bps == pytest.approx(s.impact_bps + 0.25 * s.timing_sigma_bps)
