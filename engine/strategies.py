"""Test cargo, not alpha claims.

These two strategies exist to exercise the engine. Neither is presented as a
signal worth trading, and if either shows a good Sharpe the correct reading is
"the engine ran", not "I found something".

The third one is a deliberate cheater, used by the planted-leak test.
"""
from __future__ import annotations

import numpy as np

from .data import PointInTimeView


def momentum(lookback: int = 20):
    """Long if the last `lookback` bars are up, else flat."""
    def strat(view: PointInTimeView) -> float:
        h = view.history("close", lookback + 1)
        if len(h) < lookback + 1:
            return 0.0
        return 1.0 if h[-1] > h[0] else 0.0
    return strat


def mean_reversion(lookback: int = 20, z_entry: float = 1.0):
    """Fade moves beyond `z_entry` standard deviations from the rolling mean."""
    def strat(view: PointInTimeView) -> float:
        h = view.history("close", lookback)
        if len(h) < lookback or h.std() == 0:
            return 0.0
        z = (h[-1] - h.mean()) / h.std()
        if z > z_entry:
            return -1.0
        if z < -z_entry:
            return 1.0
        return 0.0
    return strat


def planted_leak(horizon: int = 1):
    """A strategy that peeks at tomorrow. Used ONLY by the leak test.

    It reaches through `_unsafe_full_frame()` -- the one deliberate hole in the
    data layer -- because a leak has to be *possible* for a leak detector to
    prove anything. If the harness cannot catch this, it cannot catch the subtle
    version of the same bug in real strategy code.
    """
    def strat(view: PointInTimeView) -> float:
        full = view._unsafe_full_frame()
        i = int(full.index.searchsorted(view.cursor, side="right")) - 1
        j = min(i + horizon, len(full) - 1)
        future = float(full["close"].to_numpy()[j])
        now = float(full["close"].to_numpy()[i])
        return 1.0 if future > now else -1.0
    return strat


def make_synthetic_prices(n: int = 1500, seed: int = 3, mu: float = 0.0002,
                          sigma: float = 0.011):
    """A random walk with a small drift. Chosen on purpose: there is no signal
    in it, so any strategy that shows a strong Sharpe here has a bug, not an
    edge. That makes it a better engine test than real data."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    close = 100 * np.exp(np.cumsum(rets))
    gap = rng.normal(0, sigma / 3, n)
    open_ = close * np.exp(-rets + gap)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"open": open_, "close": close,
                         "high": np.maximum(open_, close) * 1.001,
                         "low": np.minimum(open_, close) * 0.999}, index=idx)
