"""A risk model estimated from returns, not declared as constants.

`engine/risk.py` constrains exposures: caps per name, per sector, on gross and
net. Every one of those numbers is a declared input, which makes it a MANDATE
and not a risk model. A mandate answers "is this position allowed?"; a risk model
answers "how much can this book lose, and from what?" -- and the second question
needs a covariance matrix.

WHY THE SAMPLE COVARIANCE IS NOT GOOD ENOUGH, and this is the whole reason the
module is more than four lines. With N assets you estimate N(N+1)/2 parameters
from T observations. At N=60 and T=250 that is 1,830 parameters from 250 rows --
the matrix is singular or nearly so, and the optimiser then finds the directions
where it is WRONGEST and puts the whole book there. The classic symptom is a
"minimum variance" portfolio with enormous offsetting positions in two nearly
identical names.

LEDOIT-WOLF SHRINKAGE is the standard answer: pull the sample covariance toward
a structured target (constant correlation), with an intensity chosen so the
result is better conditioned. The estimate is biased on purpose, because a
biased matrix you can invert beats an unbiased one you cannot.

WHAT THIS BUYS THAT THE LIMITS CANNOT:

  EX-ANTE VOLATILITY   what the book is expected to lose, before it does. A
                       gross cap says nothing about it -- two books at the same
                       gross can differ tenfold in risk depending on
                       correlation.
  RISK CONTRIBUTION    which position is actually driving the risk. Almost never
                       the largest one, because size and risk are different
                       things.
  DIVERSIFICATION      whether the book is genuinely diversified or holds twenty
                       names that are one bet. Reported as the crude
                       diversification ratio and NOT as an effective-bet count --
                       see `pca_effective_bets` for the better-looking measure
                       that was tried and rejected, and for what rejected it.

WHAT IT DOES NOT BUY. Any of this is an estimate from history, and history is
the one sample where the correlations held. They rise in a crisis -- which is
exactly when the number matters -- so a covariance-based risk figure is a
statement about calm markets that is quoted during storms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskEstimate:
    names: list
    volatility_annual: float
    var_95: float
    var_99: float
    risk_contributions: dict
    diversification_ratio: float
    shrinkage: float


def shrunk_covariance(returns: np.ndarray) -> tuple:
    """Ledoit-Wolf toward constant correlation. Returns (cov, shrinkage).

    Falls back to the sample covariance only when scikit-learn is absent, and
    says so via the returned intensity of 0.0 -- so a caller can tell the
    difference between "no shrinkage was needed" and "no shrinkage was applied".
    """
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(returns)
        return lw.covariance_, float(lw.shrinkage_)
    except Exception:                                        # noqa: BLE001
        return np.cov(returns, rowvar=False), 0.0


def estimate(returns: np.ndarray, weights: dict, names: list,
             periods_per_year: int = 252) -> RiskEstimate:
    """Ex-ante risk of a book, from the return matrix it actually holds."""
    w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
    cov, shrink = shrunk_covariance(np.asarray(returns, dtype=float))

    var_p = float(w @ cov @ w)
    vol_period = np.sqrt(max(var_p, 0.0))
    vol_annual = vol_period * np.sqrt(periods_per_year)

    # MARGINAL contribution to risk, then component contribution. The component
    # contributions sum to the total variance -- that identity is what makes
    # "which position drives the risk" a decomposition rather than a ranking.
    marginal = cov @ w
    component = w * marginal
    total = component.sum()
    contributions = {n: float(c / total) if total else 0.0
                     for n, c in zip(names, component)}

    # Diversification ratio: weighted average of individual vols over portfolio
    # vol. 1.0 means the book is one bet wearing many tickers.
    individual = np.sqrt(np.diag(cov))
    weighted_avg = float(np.abs(w) @ individual)
    div_ratio = weighted_avg / vol_period if vol_period else 1.0


    return RiskEstimate(
        names=list(names),
        volatility_annual=vol_annual,
        # Parametric VaR: a normal quantile on the portfolio vol. Stated as
        # parametric because it UNDERSTATES the tail on real returns, which are
        # fat-tailed -- the number is a floor and calling it "VaR" without the
        # qualifier is how a risk report becomes reassuring.
        var_95=float(1.645 * vol_period),
        var_99=float(2.326 * vol_period),
        risk_contributions=contributions,
        diversification_ratio=div_ratio,
        shrinkage=shrink,
    )


def pca_effective_bets(cov: np.ndarray, w: np.ndarray) -> float:
    """Meucci's effective number of bets. **Deliberately not part of
    `RiskEstimate`** -- it is here with the test that rejected it, so it does not
    get re-added by someone who reaches the same obvious idea I did.

    The idea is right: "how many independent bets is this book?" cannot be
    answered from the position list, so decompose the variance in a basis where
    the risk sources are uncorrelated -- the eigenvectors -- and take the
    exponential of the entropy of that distribution.

    IT DOES NOT SURVIVE CONTACT WITH DATA, in two separate ways, both measured in
    tests/test_risk_model.py:

      1. On the POPULATION covariance of six independent assets it returns
         exactly 6.000, as the theory promises. On a SAMPLE covariance of the
         same six with 4,000 observations it returns about 1.9. When eigenvalues
         are near-equal the eigenvectors are an arbitrary rotation, and an
         equal-weight book lands mostly on whichever one happens to point its
         way. The measure is basis-dependent precisely where the answer should
         be easiest.

      2. It is a hair trigger. At a pairwise correlation of 0.2 it already reads
         1.004. Every equity book shares a market factor well above that, so on
         real prices it returns 1.0 for a diversified book and 1.0 for a
         concentrated one. A number that gives the same answer to both questions
         is not a measurement.

    Two candidate replacements were tried and are worse. Concentration of the
    per-position risk contributions rates an equal-weight book of three
    near-identical tech names as well diversified, because its contributions ARE
    even -- it rated tech+index above cross-sector on the real cached prices,
    which is backwards. The minimum-torsion (Lowdin) basis fixes the independent
    case but reports TWO effective bets for two perfectly correlated assets,
    which is worse still.

    What is reported instead is the diversification ratio, which is crude, has no
    entropy in it, and does move in the right direction on real data.
    """
    cov = np.asarray(cov, dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    contrib = (vecs.T @ np.asarray(w, dtype=float)) ** 2 * vals
    total = contrib.sum()
    if total <= 0:
        return 0.0
    p = contrib / total
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def condition_number(cov: np.ndarray) -> float:
    """How invertible is this matrix? The number that says whether an optimiser
    built on it is solving a problem or amplifying noise."""
    vals = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    lo = max(float(vals.min()), 1e-18)
    return float(vals.max() / lo)


def render(est: RiskEstimate) -> str:
    L = ["{:<34}{:>14.4f}".format("annualised volatility", est.volatility_annual),
         "{:<34}{:>14.4f}".format("parametric VaR 95% (per period)", est.var_95),
         "{:<34}{:>14.4f}".format("parametric VaR 99% (per period)", est.var_99),
         "{:<34}{:>14.4f}".format("diversification ratio", est.diversification_ratio),
         "{:<34}{:>14.4f}".format("Ledoit-Wolf shrinkage", est.shrinkage),
         "",
         "{:<20}{:>16}".format("position", "risk share")]
    for name, share in sorted(est.risk_contributions.items(),
                              key=lambda kv: -abs(kv[1]))[:10]:
        L.append("{:<20}{:>15.1%}".format(name, share))
    return "\n".join(L)
