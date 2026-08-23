"""Position limits, leverage, sector neutrality and the cost of being short.

A backtest without these is measuring a portfolio no risk committee would sign.
The three things it silently assumes:

  * that you may put an unbounded fraction of the book in one name,
  * that gross exposure is free, and
  * that a short position costs the same as a long one.

None of those is true, and the third is the one that quietly flatters
long/short backtests: a short is a borrow, the borrow has a fee, and on hard-to-
borrow names that fee is the entire edge.

ORDER OF APPLICATION MATTERS AND IS A DECISION. Caps are applied name -> sector
-> book, because that is the order the constraints actually bind in a mandate:
a name limit is a concentration rule, a sector limit is a diversification rule,
and gross leverage is a balance-sheet rule. Applying them in a different order
gives a DIFFERENT portfolio, not a rounding difference, so `apply_limits`
returns which constraints bound and in what order rather than just the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskLimits:
    max_name_weight: float = 0.10        # |w_i| per name
    max_sector_weight: float = 0.30      # |sum of w in a sector|
    max_gross: float = 1.00              # sum |w|
    max_net: float = 1.00                # |sum w|
    borrow_bps_annual: float = 50.0      # financing on short notional
    hard_to_borrow_bps: dict = field(default_factory=dict)

    def borrow_cost(self, weights: dict, days: float = 1.0) -> float:
        """Financing on SHORT notional only, as a fraction of book per period.

        Longs are funded by the cash the shorts raise in a market-neutral book,
        so charging a symmetric financing rate on gross would double-count. What
        a short actually costs is the stock-loan fee, and that fee is per name:
        a general-collateral name is a few bps and a crowded short is hundreds.
        """
        total = 0.0
        for name, w in weights.items():
            if w >= 0:
                continue
            bps = self.hard_to_borrow_bps.get(name, self.borrow_bps_annual)
            total += abs(w) * bps / 1e4 * days / 252.0
        return total


def _scale_to(weights: np.ndarray, cap: float, measure) -> tuple[np.ndarray, bool]:
    value = measure(weights)
    if value <= cap + 1e-12 or value == 0:
        return weights, False
    return weights * (cap / value), True


def apply_limits(target: dict, sectors: dict | None, limits: RiskLimits
                 ) -> tuple[dict, list[str]]:
    """Return (feasible weights, ordered list of constraints that bound)."""
    names = list(target)
    w = np.array([float(target[n]) for n in names])
    bound: list[str] = []

    # 1. per-name concentration -- a clip, not a scale. Scaling the whole book
    #    to fix one oversized name would punish every other position for it.
    clipped = np.clip(w, -limits.max_name_weight, limits.max_name_weight)
    if not np.allclose(clipped, w):
        bound.append("name<={:.0%}".format(limits.max_name_weight))
    w = clipped

    # 2. sector exposure -- scale WITHIN the offending sector only.
    if sectors:
        for sec in sorted(set(sectors.get(n, "?") for n in names)):
            mask = np.array([sectors.get(n, "?") == sec for n in names])
            net = float(w[mask].sum())
            if abs(net) > limits.max_sector_weight + 1e-12:
                w[mask] *= limits.max_sector_weight / abs(net)
                bound.append("sector[{}]<={:.0%}".format(sec, limits.max_sector_weight))

    # 3. book-level gross, then net. Gross first: scaling for gross can only
    #    shrink net, so a net breach after a gross scale is still a real breach,
    #    whereas fixing net first can be undone by the gross scale.
    w, hit = _scale_to(w, limits.max_gross, lambda v: float(np.abs(v).sum()))
    if hit:
        bound.append("gross<={:.2f}".format(limits.max_gross))
    w, hit = _scale_to(w, limits.max_net, lambda v: abs(float(v.sum())))
    if hit:
        bound.append("net<={:.2f}".format(limits.max_net))

    return {n: float(v) for n, v in zip(names, w)}, bound


def neutralise(target: dict, sectors: dict) -> dict:
    """Demean weights within each sector so every sector nets to zero.

    This removes the sector BET, not the sector exposure: a book that is long
    the best three energy names and short the worst three is still exposed to
    oil, it just is not paid for the direction of oil. Confusing those two is
    how a "market-neutral" fund discovers it was long beta all along.
    """
    out = dict(target)
    for sec in set(sectors.values()):
        members = [n for n in target if sectors.get(n) == sec]
        if not members:
            continue
        mean = sum(target[n] for n in members) / len(members)
        for n in members:
            out[n] = target[n] - mean
    return out


def exposures(weights: dict, sectors: dict | None = None) -> dict:
    v = np.array(list(weights.values()), dtype=float)
    out = {"gross": float(np.abs(v).sum()), "net": float(v.sum()),
           "long": float(v[v > 0].sum()), "short": float(v[v < 0].sum()),
           "n_positions": int((np.abs(v) > 1e-12).sum()),
           "max_name": float(np.abs(v).max()) if len(v) else 0.0}
    if sectors:
        out["sector_net"] = {
            s: float(sum(w for n, w in weights.items() if sectors.get(n) == s))
            for s in sorted(set(sectors.values()))}
    return out
