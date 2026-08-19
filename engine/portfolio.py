"""Cross-sectional portfolio backtest over a universe with delistings.

Differences from the single-instrument loop in backtest.py, each of which is a
place a naive multi-asset backtest goes wrong:

  * Membership is point-in-time. On each rebalance the strategy ranks only the
    names LISTED on that date. It cannot buy a name that has not listed yet, and
    it cannot avoid one that is about to die.
  * Delisting is a cash event, not a missing row. When a held name delists, the
    position is closed at the delisting return -- -100% for a bankruptcy. The
    common bug is to drop the row, which silently converts a total loss into "no
    position", and that single line is most of survivorship bias.
  * Costs are charged on turnover, including the forced liquidation at delisting.
  * Participation-based slippage: the cost of trading scales with how much of a
    day's volume you are. A strategy that looks fine at $1m and dies at $100m is
    the normal case, and a fixed-bps model cannot show it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .universe import Universe

TRADING_DAYS = 252


@dataclass
class ParticipationCost:
    """Cost model with a size-dependent term.

    total_bps = fixed + impact_coefficient * sqrt(participation)

    The square-root form is the standard market-impact shape (Almgren et al.).
    `participation` is order size over daily volume. At 1% participation the
    impact term is a tenth of its value at 100%, which is why capacity analysis
    has to be non-linear -- doubling AUM does not double costs, it does worse
    than that per dollar as you climb.
    """
    fixed_bps: float = 3.0
    impact_bps_at_full_participation: float = 120.0
    daily_volume_usd: float = 5_000_000.0
    portfolio_usd: float = 1_000_000.0

    def cost_bps(self, weight_traded: float) -> float:
        if weight_traded <= 0:
            return 0.0
        notional = abs(weight_traded) * self.portfolio_usd
        participation = min(notional / max(self.daily_volume_usd, 1.0), 1.0)
        return self.fixed_bps + self.impact_bps_at_full_participation * np.sqrt(participation)


@dataclass
class PortfolioResult:
    equity: pd.Series
    returns: pd.Series
    turnover: float
    costs_paid: float
    delist_hits: int
    delist_pnl: float
    n_rebalances: int
    meta: dict = field(default_factory=dict)

    def sharpe(self) -> float:
        r = self.returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))

    def max_drawdown(self) -> float:
        eq = self.equity
        return float((eq / eq.cummax() - 1).min())

    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)

    def summary(self) -> dict:
        return {"sharpe": self.sharpe(), "max_drawdown": self.max_drawdown(),
                "total_return": self.total_return(), "turnover": self.turnover,
                "costs_paid": self.costs_paid, "delist_hits": self.delist_hits,
                "delist_pnl": self.delist_pnl, **self.meta}


def run_portfolio(universe: Universe, rank_fn, *, top_n: int = 10,
                  rebalance_every: int = 20, lookback: int = 60,
                  costs: ParticipationCost | None = None,
                  restrict_to: list[str] | None = None) -> PortfolioResult:
    """`rank_fn(history_df) -> pd.Series` of scores; highest scores are bought.

    `restrict_to` is how the survivorship experiment is run: pass the survivor
    list to reproduce the contaminated backtest, or None for the honest one.
    """
    costs = costs or ParticipationCost()
    idx = universe.index
    close = universe.prices["close"]
    open_ = universe.prices["open"]

    weights: dict[str, float] = {}
    equity = 1.0
    eq_curve, ret_curve = [], []
    turnover_total = costs_total = delist_pnl = 0.0
    delist_hits = n_rebalances = 0

    for i in range(len(idx)):
        date = idx[i]
        if i < lookback or i + 1 >= len(idx):
            eq_curve.append(equity)
            ret_curve.append(0.0)
            continue

        # ---- delisting events settle first, at the delisting return ----------
        day_pnl = 0.0
        for ticker in list(weights):
            reason, dret = universe.delist_event(ticker, date)
            if reason is not None:
                day_pnl += weights[ticker] * dret
                delist_pnl += weights[ticker] * dret
                delist_hits += 1
                turnover_total += abs(weights[ticker])
                costs_total += abs(weights[ticker]) * costs.cost_bps(
                    abs(weights[ticker])) / 10_000
                del weights[ticker]

        # ---- rebalance -------------------------------------------------------
        if (i - lookback) % rebalance_every == 0:
            eligible = universe.members_on(date)
            if restrict_to is not None:
                eligible = [t for t in eligible if t in set(restrict_to)]
            hist = close.loc[idx[i - lookback]:date, eligible]
            if hist.shape[1] >= 2:
                scores = rank_fn(hist).dropna().sort_values(ascending=False)
                picks = list(scores.index[:top_n])
                target = {t: 1.0 / len(picks) for t in picks} if picks else {}

                traded = 0.0
                for t in set(target) | set(weights):
                    delta = abs(target.get(t, 0.0) - weights.get(t, 0.0))
                    if delta > 0:
                        traded += delta
                        costs_total += delta * costs.cost_bps(delta) / 10_000
                turnover_total += traded
                cost_hit = sum(
                    abs(target.get(t, 0.0) - weights.get(t, 0.0))
                    * costs.cost_bps(abs(target.get(t, 0.0) - weights.get(t, 0.0)))
                    / 10_000
                    for t in set(target) | set(weights))
                day_pnl -= cost_hit
                weights = target
                n_rebalances += 1

        # ---- hold: next bar's open -> close ---------------------------------
        for t, w in weights.items():
            o, c = open_.at[idx[i + 1], t], close.at[idx[i + 1], t]
            if np.isfinite(o) and np.isfinite(c) and o > 0:
                day_pnl += w * (c / o - 1.0)

        equity *= (1 + day_pnl)
        eq_curve.append(equity)
        ret_curve.append(day_pnl)

    return PortfolioResult(
        equity=pd.Series(eq_curve, index=idx[:len(eq_curve)]),
        returns=pd.Series(ret_curve, index=idx[:len(ret_curve)]),
        turnover=turnover_total, costs_paid=costs_total,
        delist_hits=delist_hits, delist_pnl=delist_pnl,
        n_rebalances=n_rebalances)


def momentum_rank(hist: pd.DataFrame) -> pd.Series:
    """Total return over the lookback. Test cargo, not an alpha claim."""
    return hist.iloc[-1] / hist.iloc[0] - 1.0


def reversal_rank(hist: pd.DataFrame) -> pd.Series:
    return -(hist.iloc[-1] / hist.iloc[0] - 1.0)
