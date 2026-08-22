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

import numpy as np
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


# --------------------------------------------------------------- deflated SR
def deflated_sharpe_ratio(returns, n_trials: int, benchmark_sr: float = 0.0):
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio, full form.

    `expected_max_sharpe_under_null` above uses only the number of trials. That
    is the simple form and it ignores two things that matter on real return
    series:

      SKEW      enters as -skew*SR, so its effect DEPENDS ON THE SIGN OF SR.
                For a profitable strategy (SR > 0) with negative skew -- the
                usual shape for anything that sells insurance -- the term is
                positive, the standard error widens, and the same Sharpe becomes
                less impressive. For a losing strategy the sign flips and the
                interval narrows. Stating this as "negative skew always widens
                the interval" would be the tidy version and it is wrong.
      KURTOSIS  enters as (kurt-1)/4 * SR^2, always non-negative, so fat tails
                always widen the interval regardless of direction.

    The standard error of a per-period Sharpe under non-normality is

        SE = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1) )

    which reduces to sqrt(1/(T-1)) when skew=0 and kurt=3. DSR is then the
    probability that the observed Sharpe exceeds the benchmark, given that
    benchmark and that standard error.

    Returns a probability. Below ~0.95 the strategy has not cleared the bar that
    the SEARCH ITSELF sets, and reporting the raw Sharpe would be reporting the
    maximum of N draws as if it were an estimate of one.
    """
    import math

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3 or r.std() == 0:
        return {"dsr": float("nan"), "sr": 0.0, "T": T}

    sr = float(r.mean() / r.std())                 # per-period, NOT annualised
    skew = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)

    sr0 = expected_max_sharpe_under_null(n_trials, T) / math.sqrt(TRADING_DAYS)
    if benchmark_sr:
        sr0 = max(sr0, benchmark_sr)

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return {"dsr": float("nan"), "sr": sr, "skew": skew, "kurtosis": kurt,
                "T": T, "note": "variance estimate degenerate"}
    se = math.sqrt(denom / (T - 1))
    z = (sr - sr0) / se
    dsr = float(norm.cdf(z))

    return {
        "dsr": dsr,
        "sr_per_period": sr,
        "sr_annualised": sr * math.sqrt(TRADING_DAYS),
        "benchmark_sr_per_period": sr0,
        "skew": skew,
        "kurtosis": kurt,
        "standard_error": se,
        "n_trials": n_trials,
        "T": T,
        "verdict": ("clears the search-adjusted bar" if dsr >= 0.95 else
                    "does NOT clear the search-adjusted bar"),
        "normal_se": math.sqrt(1.0 / (T - 1)),
    }

