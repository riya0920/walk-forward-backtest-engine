"""Multiple-testing adjustment.

If you try 47 parameter settings and report the best one, the reported Sharpe is
not an estimate of that strategy's edge -- it is an estimate of the maximum of 47
draws from a distribution whose true mean is probably zero. The fix is not to
stop searching. It is to compare the winner against what the *search itself*
would have produced on noise.

Benchmark used: the expected maximum Sharpe under the null of zero true skill,
following Bailey & Lopez de Prado's deflated-Sharpe construction:

    E[max SR_N] ~= sigma_SR * [ (1-g) * z(1 - 1/N) + g * z(1 - 1/(N*e)) ]

with g the Euler-Mascheroni constant and sigma_SR the standard error of an
annualised Sharpe estimated from T daily observations, ~ sqrt(252/T).

This is the simple form, not the full deflated-Sharpe statistic (which also
adjusts for skew and kurtosis of the return series). It is named, and its
limitations are stated, which is the point -- an unnamed adjustment is worse
than none.
"""
from __future__ import annotations

import math

from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649
TRADING_DAYS = 252


def sharpe_standard_error(n_days: int) -> float:
    """SE of an annualised Sharpe from n_days daily observations, under the
    null. Ignores autocorrelation and non-normality -- both make the true SE
    larger, so this adjustment is optimistic, not conservative."""
    return math.sqrt(TRADING_DAYS / max(n_days, 2))


def expected_max_sharpe_under_null(n_trials: int, n_days: int) -> float:
    if n_trials <= 1:
        return 0.0
    se = sharpe_standard_error(n_days)
    g = EULER_MASCHERONI
    a = norm.ppf(1 - 1 / n_trials)
    b = norm.ppf(1 - 1 / (n_trials * math.e))
    return se * ((1 - g) * a + g * b)


def assess(best_sharpe: float, n_trials: int, n_days: int) -> dict:
    threshold = expected_max_sharpe_under_null(n_trials, n_days)
    se = sharpe_standard_error(n_days)
    z = (best_sharpe - threshold) / se if se else 0.0
    return {
        "best_sharpe": best_sharpe,
        "n_trials": n_trials,
        "n_days": n_days,
        "sharpe_standard_error": se,
        "expected_max_under_null": threshold,
        "excess_over_null_max": best_sharpe - threshold,
        "z_vs_null_max": z,
        "verdict": ("indistinguishable from search noise"
                    if best_sharpe <= threshold else
                    "exceeds the null-search benchmark (necessary, not sufficient)"),
    }
