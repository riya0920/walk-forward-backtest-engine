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

    def sortino(self) -> float:
        r = self.returns.dropna()
        downside = r[r < 0]
        if len(downside) < 2 or downside.std() == 0:
            return float("nan")
        return float(r.mean() / downside.std() * np.sqrt(TRADING_DAYS))

    def max_drawdown(self) -> tuple[float, int]:
        eq = self.equity
        peak = eq.cummax()
        dd = eq / peak - 1.0
        trough = dd.idxmin()
        peak_before = eq.loc[:trough].idxmax()
        duration = int(len(eq.loc[peak_before:trough]))
        return float(dd.min()), duration

    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)

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
        position = target

        # --- hold through bar i+1: open -> close ---------------------------
        bar_ret = (closes[i + 1] / fill_price) - 1.0
        # Net return, costs included. Booking costs against equity but reporting
        # a gross return series would leave every risk-adjusted metric measuring
        # a portfolio nobody holds -- Sharpe would not move as costs rise.
        net = (1 + position * bar_ret) * (1 - cost) - 1
        equity *= (1 + net)

        eq_curve.append(equity); ret_curve.append(net); pos_curve.append(position)

    return Result(equity=pd.Series(eq_curve, index=idx),
                  returns=pd.Series(ret_curve, index=idx),
                  positions=pd.Series(pos_curve, index=idx),
                  turnover=turnover_total, costs_paid=costs_total, n_bars=len(idx))


def buy_and_hold(frame: pd.DataFrame, warmup: int = 60) -> Result:
    return run(frame, lambda view: 1.0, CostModel(0, 0, 0), warmup=warmup)
