"""Intraday execution on a measured volume curve."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.intraday import (CACHE, build, completion_curve, curve_dispersion,
                             load_profile, timing_exposure, twap_vs_vwap)

BUCKETS = ["{:02d}:{:02d}".format(9 + (30 + 5 * i) // 60, (30 + 5 * i) % 60)
           for i in range(78)]


def _flat():
    return pd.Series(np.full(78, 1 / 78), index=BUCKETS)


def _ushape():
    x = np.linspace(-1, 1, 78)
    w = 0.5 + x ** 2
    return pd.Series(w / w.sum(), index=BUCKETS)


DV = 50_000_000


# ------------------------------------------------------- on a flat curve
def test_twap_and_vwap_are_identical_on_a_flat_curve():
    """The control. Any difference between the two is a property of the curve's
    dispersion, so a flat curve must produce none."""
    r = twap_vs_vwap(_flat(), DV * 0.05, DV)
    assert r["excess_bps"] == pytest.approx(0.0, abs=1e-9)
    assert r["twap_max_participation"] == pytest.approx(
        r["stated_participation"])


def test_participation_is_flat_under_vwap_by_construction():
    s = build(_ushape(), DV * 0.05, DV, "vwap")
    assert s.participation.std() == pytest.approx(0.0, abs=1e-12)
    assert s.max_participation == pytest.approx(0.05)


# ------------------------------------------------------ on an uneven curve
def test_twap_overshoots_its_stated_participation_where_volume_is_thin():
    """"We traded at 5% of volume" is a statement about the day. TWAP holds
    QUANTITY constant while the volume under it swings, so the realised
    participation swings the other way."""
    s = build(_ushape(), DV * 0.05, DV, "twap")
    assert s.mean_participation == pytest.approx(0.05)
    assert s.max_participation > 0.05 * 1.5


def test_twap_costs_more_than_vwap_on_an_uneven_curve():
    r = twap_vs_vwap(_ushape(), DV * 0.05, DV)
    assert r["excess_bps"] > 0
    assert r["twap_impact_bps"] > r["vwap_impact_bps"]


def test_the_penalty_is_scale_invariant():
    """A consequence of the square-root law that also checks the arithmetic: the
    TWAP penalty is a pure function of the CURVE, not of order size. If this
    ever stops holding, the impact weighting is wrong."""
    p = _ushape()
    small = twap_vs_vwap(p, DV * 0.001, DV)["excess_pct"]
    large = twap_vs_vwap(p, DV * 0.20, DV)["excess_pct"]
    assert small == pytest.approx(large, rel=1e-9)


def test_the_penalty_grows_with_how_uneven_the_day_is():
    mild = pd.Series(np.linspace(0.9, 1.1, 78), index=BUCKETS)
    mild = mild / mild.sum()
    severe = _ushape()
    assert (twap_vs_vwap(severe, DV * 0.05, DV)["excess_pct"]
            > twap_vs_vwap(mild, DV * 0.05, DV)["excess_pct"])


# ---------------------------------------------------------- book-keeping
def test_both_schedules_execute_the_whole_order():
    for style in ("twap", "vwap"):
        s = build(_ushape(), DV * 0.05, DV, style)
        assert s.quantity.sum() == pytest.approx(DV * 0.05)


def test_an_unknown_style_is_refused():
    with pytest.raises(ValueError, match="twap or vwap"):
        build(_flat(), 1000, DV, "iceberg")


def test_a_zero_order_costs_nothing_rather_than_dividing_by_zero():
    s = build(_flat(), 0, DV, "vwap")
    assert s.impact_bps == 0.0 and s.auction_share == 0.0


def test_the_auction_share_is_reported():
    """An auction is a single price formed once, so "participating at 5% of the
    auction" is a different act from participating in continuous flow. Treating
    it as more of the same overstates how much of a schedule can land there."""
    s = build(_ushape(), DV * 0.05, DV, "vwap")
    assert 0 < s.auction_share < 1


def test_vwap_puts_more_into_the_auction_buckets_than_twap():
    p = _ushape()
    assert (build(p, DV * 0.05, DV, "vwap").auction_share
            > build(p, DV * 0.05, DV, "twap").auction_share)


# ----------------------------------------------------------- completion
def test_the_completion_curve_ends_at_one():
    c = completion_curve(build(_ushape(), DV * 0.05, DV, "vwap"), _ushape())
    assert c.iloc[-1] == pytest.approx(1.0)
    assert (c.diff().dropna() >= -1e-12).all(), "completion went backwards"


def test_timing_exposure_is_a_half_for_a_flat_schedule():
    """Under a uniform schedule half the order is outstanding on average, which
    is the benchmark `engine/execution.py` uses over days."""
    p = _flat()
    assert timing_exposure(build(p, DV * 0.05, DV, "twap"), p) == pytest.approx(
        0.5, abs=0.01)


# ------------------------------------------------- against the real data
real_only = pytest.mark.skipif(
    not CACHE.exists(),
    reason="intraday cache not built -- run fetch_intraday.py")


@real_only
def test_the_real_curve_is_a_u_and_not_a_ramp():
    """The measured shape, pinned. Both ends are heavy; the trough is midday."""
    p = load_profile()
    assert p.index[0] == "09:30" and p.index[-1] == "15:55"
    assert p.iloc[-1] > p.mean() * 3, "the close is not heavy"
    assert p.iloc[0] > p.mean() * 3, "the open is not heavy"
    assert p.loc["13:00":"14:00"].mean() < p.mean(), "midday is not the trough"


@real_only
def test_the_real_curve_is_uneven_enough_to_matter():
    d = curve_dispersion(load_profile())
    assert d["max_over_min"] > 5, (
        "the measured day is nearly flat, and the whole TWAP-vs-VWAP argument "
        "is about dispersion -- if this now holds, recheck the data")
    assert d["effective_buckets"] < d["buckets"]


@real_only
def test_the_measured_penalty_is_material():
    r = twap_vs_vwap(load_profile(), DV * 0.05, DV)
    assert r["excess_pct"] > 5, (
        "TWAP costs {:.1f}% more on the measured curve".format(r["excess_pct"]))
    assert r["twap_overshoot_ratio"] > 1.5


@real_only
def test_vwap_is_NOT_more_deferred_on_the_measured_curve():
    """The claim this module made before it measured anything.

    The textbook cost of VWAP is that it defers execution into the close. That
    assumes a back-loaded curve. The measured one is a U -- front-loaded as well
    -- so following it starts the order sooner as well as finishing it later,
    and VWAP comes out slightly LESS deferred than TWAP rather than more.

    If this ever fails, the profile has changed shape and the docstring's
    correction needs re-deriving rather than trusting.
    """
    p = load_profile()
    twap = timing_exposure(build(p, DV * 0.05, DV, "twap"), p)
    vwap = timing_exposure(build(p, DV * 0.05, DV, "vwap"), p)
    assert vwap <= twap, (
        "vwap exposure {:.4f} vs twap {:.4f}".format(vwap, twap))

    c_v = completion_curve(build(p, DV * 0.05, DV, "vwap"), p)
    c_t = completion_curve(build(p, DV * 0.05, DV, "twap"), p)
    assert c_v["12:00"] > c_t["12:00"], "vwap should be further along by noon"
