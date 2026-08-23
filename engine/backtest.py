"""Event loop with strict time discipline, costs, and metrics.

Time discipline: a signal computed from data up to and including bar t is filled
at bar t+1's OPEN. Not t's close. The strategy never gets to choose this; the
loop applies it.

Costs are not optional and not a footnote. Commission + spread + slippage are
applied on every fill, and `run(costs_bps=0)` exists only to produce the
with/without delta that docs/BIAS_AUDIT.md reports -- the delta is the honesty
exhibit, not the zero-cost number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PointInTimeView

TRADING_DAYS = 252


@dataclass
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 2.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    def cost_of(self, notional_turnover: float) -> float:
        return notional_turnover * self.total_bps / 10_000


@dataclass
class Result:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    turnover: float
    costs_paid: float
    n_bars: int
    initial_equity: float = 1.0
    trades: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- metrics -----------------------------------------------------------
    def sharpe(self) -> float:
        """Annualised from DAILY returns as mean/std * sqrt(252). Stated because
        'Sharpe 1.4' means nothing without the frequency and the method, and
        monthly-compounded vs daily-scaled are not the same number."""
        r = self.returns.dropna()
        if r.std() == 0 or len(r) < 2:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))

    def sortino(self, mar: float = 0.0) -> float:
        """Annualised Sortino against a minimum acceptable return `mar`.

        The denominator is the second lower partial moment computed over ALL
        periods, with upside set to zero:

            DD = sqrt( mean_t( min(r_t - mar, 0)^2 ) )

        An earlier version used `r[r < 0].std()` -- the dispersion of the losing
        returns about THEIR OWN mean, over the losing subset only. That is a
        different statistic and it is not Sortino. The vectorbt cross-check in
        `run_vectorbt_check.py` is what caught it: identical return series,
        0.554 from this engine against 0.641 from vectorbt, a 13.4% gap with no
        modelling difference behind it.

        Its failure mode is worse than the 13% suggests. If every loss is the
        same size the subset standard deviation is ZERO and the ratio is
        infinite -- a strategy with perfectly uniform losses scored as flawless.
        The lower partial moment cannot do that, because a loss enters by its
        magnitude rather than by how much it differs from other losses.
        """
        r = self.returns.dropna().to_numpy()
        if len(r) < 2:
            return float("nan")
        downside = np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2))
        if downside == 0:
            return float("nan")
        return float((r.mean() - mar) / downside * np.sqrt(TRADING_DAYS))

    def max_drawdown(self) -> tuple[float, int]:
        eq = self.equity
        # The running peak has to start at the STARTING capital, not at the
        # first recorded bar. `equity` holds end-of-bar values, so if the very
        # first bar loses money that loss is the peak on the old reading and the
        # drawdown from par is invisible.
        peak = eq.cummax().clip(lower=self.initial_equity)
        dd = eq / peak - 1.0
        trough = dd.idxmin()
        peak_before = eq.loc[:trough].idxmax()
        duration = int(len(eq.loc[peak_before:trough]))
        return float(dd.min()), duration

    def total_return(self) -> float:
        """Against the STARTING capital, not against the first recorded bar.

        This read `equity.iloc[0]` until the vectorbt cross-check disagreed on
        the buy-and-hold identity. `equity` is an end-of-bar series, so
        `iloc[0]` is already 1 + the first bar's return and dividing by it
        silently discards that bar. With the default 60-bar warmup the first
        recorded value IS the starting capital and the bug is invisible; at
        `warmup=0` it removed a -2.46% first bar and turned a 134.07% hold into
        139.97%. A bug that only appears at a non-default argument is still a
        bug, and it is exactly the kind an independent implementation finds and
        a self-written test does not.
        """
        return float(self.equity.iloc[-1] / self.initial_equity - 1)

    def summary(self) -> dict:
        dd, dur = self.max_drawdown()
        return {"sharpe": self.sharpe(), "sortino": self.sortino(),
                "max_drawdown": dd, "drawdown_bars": dur,
                "total_return": self.total_return(),
                "turnover": self.turnover, "costs_paid": self.costs_paid,
                "bars": self.n_bars, **self.meta}


def run(frame: pd.DataFrame, strategy, costs: CostModel | None = None,
        warmup: int = 60) -> Result:
    """`strategy(view) -> target position in [-1, 1]`, called once per bar with a
    PointInTimeView clamped to that bar."""
    costs = costs if costs is not None else CostModel()
    idx = frame.index
    opens = frame["open"].to_numpy()
    closes = frame["close"].to_numpy()

    position = 0.0
    equity = 1.0
    eq_curve, ret_curve, pos_curve = [], [], []
    trades: list[dict] = []
    turnover_total = 0.0
    costs_total = 0.0

    for i in range(len(idx)):
        if i < warmup or i + 1 >= len(idx):
            eq_curve.append(equity); ret_curve.append(0.0); pos_curve.append(position)
            continue

        # --- decide at bar i, using only data up to and including bar i -----
        target = float(strategy(PointInTimeView(frame, idx[i])))
        target = max(-1.0, min(1.0, target))

        # --- fill at bar i+1's OPEN ----------------------------------------
        fill_price = opens[i + 1]
        traded = abs(target - position)
        cost = costs.cost_of(traded) if traded > 0 else 0.0
        if traded > 0:
            turnover_total += traded
            costs_total += cost
            # One row per FILL, with the price it printed at and the equity it
            # printed against. A blotter is not a nicety: without it, "turnover
            # 41.2" is a number nobody can tie back to a decision, and a cost
            # dispute has nothing to argue from.
            trades.append({
                "bar": idx[i + 1], "decided_at": idx[i],
                "side": "buy" if target > position else "sell",
                "from_position": position, "to_position": target,
                "traded": traded, "fill_price": float(fill_price),
                "cost": cost, "equity_before": equity})

        # --- P&L for bar i+1, split at the open ----------------------------
        # The OLD position is held across the overnight gap (prior close ->
        # this open); the NEW position is held from the fill to this close.
        #
        # An earlier version computed only open->close for the whole bar, which
        # silently discarded every overnight gap. On a mean-zero synthetic walk
        # that is nearly invisible; on real equities it is most of the return --
        # buy-and-hold SPY 2015-2024 came out at 47% instead of ~190%. Any
        # strategy holding overnight was being measured on a portfolio that
        # liquidates at every close and re-buys at every open.
        prev_close = closes[i]
        overnight = (fill_price / prev_close) - 1.0
        intraday = (closes[i + 1] / fill_price) - 1.0

        # The three sub-periods COMPOUND, they do not add. Writing this as
        # `position*overnight + target*intraday` drops the cross term, which is
        # invisible on one bar and compounds into several percent over a few
        # thousand of them -- it inflated buy-and-hold on the synthetic series by
        # 3.4% against the price return it must reproduce exactly.
        #
        # Order matters and is the real sequence of events: the OLD position
        # carries the overnight gap, costs are paid when the trade prints at the
        # open, and the NEW position carries the rest of the day.
        net = ((1 + position * overnight)
               * (1 - cost)
               * (1 + target * intraday)) - 1
        position = target
        equity *= (1 + net)

        eq_curve.append(equity); ret_curve.append(net); pos_curve.append(position)

    return Result(equity=pd.Series(eq_curve, index=idx),
                  returns=pd.Series(ret_curve, index=idx),
                  positions=pd.Series(pos_curve, index=idx),
                  turnover=turnover_total, costs_paid=costs_total, n_bars=len(idx),
                  initial_equity=1.0, trades=trades)


def buy_and_hold(frame: pd.DataFrame, warmup: int = 60) -> Result:
    return run(frame, lambda view: 1.0, CostModel(0, 0, 0), warmup=warmup)
