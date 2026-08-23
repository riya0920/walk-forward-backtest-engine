"""The bootstrap data-snooping tests, checked in both directions.

A significance test needs two proofs, not one: it must FIRE on a planted edge
and it must STAY QUIET on noise. Only checking the first gives you a test that
always rejects, which is worse than no test at all because it looks rigorous.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.reality_check import (hansens_spa, outperformance_matrix,
                                  stationary_bootstrap_indices,
                                  whites_reality_check)

T = 800


def _noise(k, seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(T, k))


# ------------------------------------------------------------ the resampler
def test_indices_are_in_range_and_the_right_length():
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(500, 5.0, rng)
    assert len(idx) == 500
    assert idx.min() >= 0 and idx.max() < 500


def test_block_length_one_degenerates_to_the_iid_bootstrap():
    """A mean block length of 1 must destroy serial structure, otherwise the
    block parameter is decoration."""
    rng = np.random.default_rng(2)
    idx = stationary_bootstrap_indices(2000, 1.0, rng)
    consecutive = np.mean(np.diff(idx) == 1)
    assert consecutive < 0.05


def test_longer_blocks_preserve_more_consecutive_runs():
    rng = np.random.default_rng(3)
    short = stationary_bootstrap_indices(4000, 2.0, rng)
    long_ = stationary_bootstrap_indices(4000, 20.0, rng)
    assert np.mean(np.diff(long_) == 1) > np.mean(np.diff(short) == 1) + 0.3


# ------------------------------------------------------------------- power
def test_a_planted_edge_is_detected_by_both_tests():
    d = _noise(6, seed=5)
    d[:, 3] += 0.0025          # ~0.25% per bar of genuine outperformance
    assert whites_reality_check(d, n_boot=400)["p_value"] < 0.05
    assert hansens_spa(d, n_boot=400)["p_value"] < 0.05


def test_the_winning_column_is_identified():
    d = _noise(6, seed=6)
    d[:, 2] += 0.0025
    assert whites_reality_check(d, n_boot=400)["best_index"] == 2
    assert hansens_spa(d, n_boot=400)["best_index"] == 2


# -------------------------------------------------------------------- size
@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_pure_noise_is_not_declared_significant(seed):
    """20 strategies with no edge. A test that rejects here is manufacturing
    the exact finding it exists to prevent."""
    d = _noise(20, seed=seed)
    assert whites_reality_check(d, n_boot=400)["p_value"] > 0.05
    assert hansens_spa(d, n_boot=400)["p_value"] > 0.05


def test_a_single_lucky_column_out_of_many_does_not_reject():
    """The whole point: the best of 40 noise series looks good in isolation and
    must not look good against the maximum of 40 draws."""
    d = _noise(40, seed=21)
    best = d.mean(axis=0).argmax()
    naive_t = d[:, best].mean() / (d[:, best].std() / np.sqrt(T))
    assert naive_t > 1.9, "the setup needs a column that a t-test would pass"
    assert whites_reality_check(d, n_boot=400)["p_value"] > 0.05


# --------------------------------------------------- RC vs SPA, the padding
def _padded(d, n_bad, drag=0.004, seed=31):
    rng = np.random.default_rng(seed)
    bad = rng.normal(-drag, 0.01, size=(d.shape[0], n_bad))
    return np.column_stack([d, bad])


def test_spa_is_insensitive_to_padding_with_hopeless_strategies():
    d = _noise(8, seed=41)
    d[:, 0] += 0.0012
    base = hansens_spa(d, n_boot=400)["p_value"]
    padded = hansens_spa(_padded(d, 25), n_boot=400)["p_value"]
    assert abs(padded - base) < 0.02


@pytest.mark.parametrize("seed", [41, 43, 44])
def test_reality_check_IS_sensitive_to_that_padding(seed):
    """Not a bug in my implementation -- it is RC's documented weakness, and
    pinning it here is what makes the SPA test above meaningful.

    The assertion is RELATIVE because the effect is multiplicative and the base
    p-value is often near zero: measured inflation over these seeds is 3.4x to
    7x (and one 0.0000 -> 0.0150). An absolute threshold looks stricter and is
    actually just seed-dependent.
    """
    d = _noise(8, seed=seed)
    d[:, 0] += 0.0012
    base = whites_reality_check(d, n_boot=400)["p_value"]
    padded = whites_reality_check(_padded(d, 25), n_boot=400)["p_value"]
    assert padded >= 3 * base
    assert padded - base > 0.005


def test_spa_reports_how_many_columns_it_nulled():
    d = _padded(_noise(5, seed=51), 12)
    res = hansens_spa(d, n_boot=400)
    assert res["n_treated_as_null"] >= 12
    assert (res["n_treated_as_null"] + res["n_recentred_on_own_mean"]
            == res["n_strategies"])


# ---------------------------------------------------------------- plumbing
def test_outperformance_matrix_subtracts_the_benchmark():
    a = np.array([0.01, 0.02, 0.03])
    b = np.array([0.005, 0.005, 0.005])
    m = outperformance_matrix([a], b)
    assert np.allclose(m[:, 0], a - b)


def test_outperformance_matrix_trims_to_a_common_length():
    m = outperformance_matrix([np.ones(10), np.ones(7)], np.zeros(12))
    assert m.shape == (7, 2)


def test_results_are_reproducible_for_a_fixed_seed():
    d = _noise(5, seed=61)
    assert (whites_reality_check(d, n_boot=200, seed=99)["p_value"]
            == whites_reality_check(d, n_boot=200, seed=99)["p_value"])
