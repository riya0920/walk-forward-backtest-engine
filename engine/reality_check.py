"""White's Reality Check and Hansen's SPA -- the bootstrap answer to data snooping.

`multiple_testing.py` answers "how big a Sharpe would the SEARCH have produced on
noise?" with a closed form that needs only the trial count. That form assumes the
trials are independent and the returns are i.i.d. normal. Neither is true: 40
momentum variants with lookbacks 5..90 are enormously correlated with each other,
and daily returns are neither normal nor independent.

The bootstrap alternatives fix exactly that, because they resample the ACTUAL
joint return matrix and therefore carry the cross-strategy correlation and the
serial dependence with them, without ever having to model either.

RESAMPLING SCHEME. Plain i.i.d. bootstrap destroys serial dependence, which is
the thing that makes a Sharpe standard error wrong in the first place. This uses
the STATIONARY BOOTSTRAP of Politis & Romano (1994): blocks of geometric length
(mean `block_len`) sampled with wraparound, which preserves short-range dependence
while keeping the resampled series stationary. `block_len = 1` degenerates to the
i.i.d. bootstrap, and the test `test_block_length_matters` shows the p-value moves
when it should.

WHITE'S REALITY CHECK (2000). Null: the best of the K strategies does not beat
the benchmark. With d[k,t] the per-period outperformance of strategy k over the
benchmark, the statistic is

    V = max_k  sqrt(T) * mean_t d[k,t]

and its null distribution comes from resampling d and RE-CENTRING each column on
its own observed mean:

    V*_b = max_k  sqrt(T) * ( mean_t d*[k,t] - mean_t d[k,t] )

p = P(V*_b >= V). The re-centring is the whole trick: it imposes the null
(nobody beats the benchmark) on the bootstrap world while keeping the real
dependence structure.

RC's known weakness is that it is a MAXIMUM over all K columns, so adding
strategies that are obviously terrible still inflates the null distribution and
pushes the p-value up. A researcher can therefore make a good strategy look
insignificant by including junk -- or, read the other way, RC's verdict depends
on the company you keep.

HANSEN'S SPA (2005) fixes that in two steps:

  1. STUDENTISE. Divide each column by its own bootstrap standard error, so a
     high-variance strategy does not dominate the maximum purely by being noisy.
  2. RECENTRE SELECTIVELY. Only columns whose observed mean clears

        -sqrt( omega_k^2 / T * 2 * log log T )

     are recentred on their own mean; columns far below the benchmark are
     recentred to zero, i.e. treated as if they were exactly at the null rather
     than as evidence against it. That threshold is the standard rate from the
     law of the iterated logarithm -- it lets in strategies whose
     underperformance is within sampling noise and excludes the hopeless ones.

`run_reality_check.py` demonstrates the divergence directly: pad the candidate
set with deliberately bad variants and watch the RC p-value climb while the SPA
p-value stays put.

WHAT NEITHER FIXES. Both test the strategies you actually ran. Variants
abandoned before they were recorded, features chosen by eye, and the decision to
study this asset at all are invisible to both. The trial counter in
`run_audit.py` is the only defence against those, and it is a discipline, not a
statistic.
"""
from __future__ import annotations

import math

import numpy as np

TRADING_DAYS = 252


# ------------------------------------------------------------------ resampler
def stationary_bootstrap_indices(n: int, block_len: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap index path of length n.

    Geometric block lengths with p = 1/block_len: at each step, continue the
    current block with probability 1-p or start a new one at a uniform position.
    Wraparound keeps every original observation equally likely to be sampled,
    which the non-circular version does not (it under-samples the tails).
    """
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    p = 1.0 / block_len
    idx = np.empty(n, dtype=np.int64)
    cur = int(rng.integers(0, n))
    for t in range(n):
        if t > 0:
            if rng.random() < p:
                cur = int(rng.integers(0, n))
            else:
                cur = (cur + 1) % n
        idx[t] = cur
    return idx


def _bootstrap_means(d: np.ndarray, n_boot: int, block_len: float,
                     seed: int) -> np.ndarray:
    """(n_boot, K) matrix of resampled column means."""
    rng = np.random.default_rng(seed)
    T = d.shape[0]
    out = np.empty((n_boot, d.shape[1]), dtype=float)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, block_len, rng)
        out[b] = d[idx].mean(axis=0)
    return out


# ------------------------------------------------------------ Reality Check
def whites_reality_check(d: np.ndarray, n_boot: int = 1000,
                         block_len: float = 5.0, seed: int = 7) -> dict:
    """d: (T, K) per-period outperformance of each strategy over the benchmark."""
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    T, K = d.shape
    root_t = math.sqrt(T)
    obs = d.mean(axis=0)
    v = float(np.max(root_t * obs))

    boot = _bootstrap_means(d, n_boot, block_len, seed)
    v_star = np.max(root_t * (boot - obs), axis=1)
    p = float(np.mean(v_star >= v))

    return {
        "test": "White's Reality Check (stationary bootstrap)",
        "statistic": v,
        "p_value": p,
        "n_strategies": K,
        "n_periods": T,
        "n_boot": n_boot,
        "block_len": block_len,
        "best_mean_outperformance": float(obs.max()),
        "best_index": int(np.argmax(obs)),
        "verdict": ("rejects the no-superior-strategy null" if p < 0.05 else
                    "cannot reject: consistent with data snooping"),
    }


# --------------------------------------------------------------------- SPA
def hansens_spa(d: np.ndarray, n_boot: int = 1000, block_len: float = 5.0,
                seed: int = 7) -> dict:
    """Hansen's Superior Predictive Ability test, consistent (SPA_c) variant."""
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    T, K = d.shape
    root_t = math.sqrt(T)
    obs = d.mean(axis=0)

    boot = _bootstrap_means(d, n_boot, block_len, seed)
    # omega_k: bootstrap standard error of sqrt(T)*mean. Estimating it FROM the
    # bootstrap rather than from a HAC formula keeps the serial dependence in the
    # denominator too -- studentising by an i.i.d. standard error would undo the
    # block resampling in the numerator.
    omega = root_t * boot.std(axis=0, ddof=1)
    omega = np.where(omega < 1e-12, 1e-12, omega)

    t_stats = root_t * obs / omega
    t_obs = float(np.max(t_stats))

    # Selective recentring. log log T is tiny at these sample sizes, which is why
    # the threshold is a rate result and not a magic number -- at T=1500 it
    # admits any strategy within ~1.4 bootstrap standard errors below zero.
    llt = math.log(math.log(T)) if T > 15 else 1.0
    threshold = -np.sqrt(omega ** 2 / T * 2.0 * max(llt, 1e-9))
    keep = obs >= threshold
    g = np.where(keep, obs, 0.0)

    t_star = np.max(root_t * (boot - g) / omega, axis=1)
    p = float(np.mean(t_star >= t_obs))

    return {
        "test": "Hansen's SPA (consistent variant, stationary bootstrap)",
        "statistic": t_obs,
        "p_value": p,
        "n_strategies": K,
        "n_periods": T,
        "n_boot": n_boot,
        "block_len": block_len,
        "n_recentred_on_own_mean": int(keep.sum()),
        "n_treated_as_null": int((~keep).sum()),
        "best_index": int(np.argmax(t_stats)),
        "verdict": ("rejects the no-superior-strategy null" if p < 0.05 else
                    "cannot reject: consistent with data snooping"),
    }


def outperformance_matrix(strategy_returns: list[np.ndarray],
                          benchmark_returns: np.ndarray) -> np.ndarray:
    """Stack per-period (strategy - benchmark) columns, trimmed to a common length.

    The performance measure is the per-period return differential, which is what
    RC and SPA were written for. Using a ratio like Sharpe as the loss would need
    the delta method inside the bootstrap and is not what the tests assume.
    """
    b = np.asarray(benchmark_returns, dtype=float)
    cols = []
    for r in strategy_returns:
        r = np.asarray(r, dtype=float)
        n = min(len(r), len(b))
        cols.append(r[-n:] - b[-n:])
    n = min(len(c) for c in cols)
    return np.column_stack([c[-n:] for c in cols])
