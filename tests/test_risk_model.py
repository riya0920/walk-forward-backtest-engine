"""Risk estimated from returns, and why the sample covariance is not enough."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.risk_model import (condition_number, estimate,
                               pca_effective_bets, shrunk_covariance)


def _correlated(n_assets=6, n_obs=400, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, size=(n_obs, 1))
    idio = rng.normal(0, 0.01, size=(n_obs, n_assets))
    return np.sqrt(rho) * common + np.sqrt(1 - rho) * idio


NAMES = ["A", "B", "C", "D", "E", "F"]
EQUAL = {n: 1 / 6 for n in NAMES}


def test_correlated_books_are_riskier_than_uncorrelated_ones_at_equal_gross():
    """The thing a gross cap cannot see: two books at the same gross can differ
    enormously in risk depending on correlation."""
    low = estimate(_correlated(rho=0.0, seed=1), EQUAL, NAMES)
    high = estimate(_correlated(rho=0.9, seed=1), EQUAL, NAMES)
    assert high.volatility_annual > low.volatility_annual * 1.5


def test_the_diversification_ratio_collapses_toward_one_when_everything_moves_together():
    """1.0 means the book is one bet wearing many tickers."""
    low = estimate(_correlated(rho=0.0, seed=2), EQUAL, NAMES)
    high = estimate(_correlated(rho=0.95, seed=2), EQUAL, NAMES)
    assert high.diversification_ratio < low.diversification_ratio
    assert high.diversification_ratio < 1.3


def test_risk_contributions_sum_to_one():
    """That identity is what makes 'which position drives the risk' a
    decomposition rather than a ranking."""
    est = estimate(_correlated(seed=3), EQUAL, NAMES)
    assert sum(est.risk_contributions.values()) == pytest.approx(1.0, abs=1e-9)


def test_the_biggest_position_is_not_automatically_the_biggest_risk():
    """Size and risk are different things, which is the entire reason to
    decompose rather than sort by weight."""
    rets = _correlated(rho=0.0, seed=4)
    rets[:, 0] *= 0.1                      # A is tiny-vol
    rets[:, 5] *= 6.0                      # F is huge-vol
    weights = {"A": 0.50, "B": 0.10, "C": 0.10, "D": 0.10, "E": 0.10, "F": 0.10}
    est = estimate(rets, weights, NAMES)
    biggest_weight = max(weights, key=weights.get)
    biggest_risk = max(est.risk_contributions, key=est.risk_contributions.get)
    assert biggest_weight == "A"
    assert biggest_risk != biggest_weight


# --------------------------------- the metric this rejected, and what rejected it
def test_pca_effective_bets_is_correct_on_the_population_matrix():
    """The theory is not in doubt: six independent assets, equally weighted, are
    six bets, and on the population covariance the measure says so exactly."""
    C = np.eye(6)
    assert pca_effective_bets(C, np.repeat(1 / 6, 6)) == pytest.approx(6.0)


def test_pca_effective_bets_breaks_on_a_sample_matrix_of_the_same_assets():
    """And this is why it is not reported. 4,000 observations of six INDEPENDENT
    assets -- as clean a sample as this repo will ever see -- and it reads under
    3 instead of 6.

    Near-equal eigenvalues make the eigenvectors an arbitrary rotation, and an
    equal-weight book lands mostly on whichever one happens to point its way. The
    measure is basis-dependent exactly where the answer should be easiest.
    """
    rets = _correlated(rho=0.0, n_obs=4000, seed=11)
    cov = np.cov(rets, rowvar=False)
    measured = pca_effective_bets(cov, np.repeat(1 / 6, 6))
    assert measured < 3.0, (
        "if this now reads near 6, the sampling problem that disqualified the "
        "metric has gone away and it is worth reinstating")


def test_pca_effective_bets_is_a_hair_trigger_and_cannot_discriminate():
    """The disqualifying result. At a pairwise correlation of 0.2 it already
    reads ~1.0, and every equity book shares a market factor well above that. It
    returns 1.0 for a diversified book and 1.0 for a concentrated one -- a number
    that gives the same answer to both questions is not a measurement."""
    w = np.repeat(1 / 6, 6)
    mild = (1 - 0.2) * np.eye(6) + 0.2 * np.ones((6, 6))
    severe = (1 - 0.95) * np.eye(6) + 0.95 * np.ones((6, 6))
    assert pca_effective_bets(mild, w) < 1.05
    assert pca_effective_bets(severe, w) < 1.05


def test_the_diversification_ratio_does_discriminate_where_the_entropy_measure_does_not():
    """Which is the whole reason the crude measure is the one reported. Same two
    correlation levels, and this one separates them."""
    mild = estimate(_correlated(rho=0.2, n_obs=2000, seed=15), EQUAL, NAMES)
    severe = estimate(_correlated(rho=0.95, n_obs=2000, seed=15), EQUAL, NAMES)
    assert mild.diversification_ratio > severe.diversification_ratio * 1.3


def test_the_reported_estimate_does_not_carry_an_effective_bet_count():
    """Pinned so it does not quietly come back."""
    est = estimate(_correlated(seed=16), EQUAL, NAMES)
    assert not hasattr(est, "effective_bets")


def test_shrinkage_is_applied_and_reported():
    est = estimate(_correlated(seed=6), EQUAL, NAMES)
    assert 0.0 < est.shrinkage <= 1.0, (
        "shrinkage of exactly 0 means none was applied, and the caller has to "
        "be able to tell that from 'none was needed'")


def test_shrinkage_conditions_a_matrix_the_sample_estimate_cannot_invert():
    """T < N is where the sample covariance stops being merely noisy and becomes
    RANK DEFICIENT: 60 assets from 40 observations gives at most 39 non-zero
    eigenvalues, so 21 directions have exactly zero estimated variance. An
    optimiser reads those as riskless and puts the whole book there.

    Note the threshold. At T=80, N=60 the sample matrix is still invertible --
    badly conditioned but not singular -- so the honest demonstration needs
    T < N and not merely T < N(N+1)/2.
    """
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, size=(40, 60))      # T < N: genuinely singular

    sample = np.cov(rets, rowvar=False)
    shrunk, intensity = shrunk_covariance(rets)

    rank = np.linalg.matrix_rank(sample)
    assert rank < 60, "expected a rank-deficient sample covariance"
    assert np.linalg.matrix_rank(shrunk) == 60, "shrinkage should restore rank"
    assert intensity > 0
    assert condition_number(shrunk) < condition_number(sample) / 1e6


def test_at_t_greater_than_n_the_sample_matrix_is_merely_bad_not_singular():
    """Stated as its own test because the distinction is what the doc claims,
    and an overstated claim is the failure mode this project is written
    against."""
    rng = np.random.default_rng(14)
    rets = rng.normal(0, 0.01, size=(80, 60))
    assert np.linalg.matrix_rank(np.cov(rets, rowvar=False)) == 60


def test_var_99_exceeds_var_95():
    est = estimate(_correlated(seed=8), EQUAL, NAMES)
    assert est.var_99 > est.var_95


def test_an_empty_book_has_no_risk():
    est = estimate(_correlated(seed=9), {n: 0.0 for n in NAMES}, NAMES)
    assert est.volatility_annual == pytest.approx(0.0)


def test_a_long_short_pair_of_identical_names_is_nearly_riskless():
    """The sanity check on the whole apparatus: if the covariance is right, a
    perfectly hedged pair should show almost no portfolio risk even at full
    gross."""
    rets = _correlated(rho=0.999, n_assets=2, seed=10)
    est = estimate(rets, {"A": 1.0, "B": -1.0}, ["A", "B"])
    unhedged = estimate(rets, {"A": 1.0, "B": 0.0}, ["A", "B"])
    assert est.volatility_annual < unhedged.volatility_annual / 5
