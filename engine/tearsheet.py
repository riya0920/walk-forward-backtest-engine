"""Tearsheet and blotter -- the two artefacts a reader can actually check.

A summary table is a claim. A blotter is evidence: every fill, the bar it was
decided on, the bar it printed on, the price, the cost, and the equity it printed
against. The two must tie -- sum of `traded` equals reported turnover, sum of
`cost` equals reported costs -- and `test_blotter_ties_to_the_summary` asserts it.
That tie is the point. A backtest whose headline turnover cannot be reassembled
from its own fills has a headline nobody can audit.

Deliberately absent: plots. An equity-curve PNG is the part of a tearsheet that
carries the least information per pixel, and it cannot be diffed in a pull
request. The monthly return table below carries the same information and is
readable in a terminal, in a diff, and in CI output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def blotter(result) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame(columns=["bar", "decided_at", "side", "traded",
                                     "fill_price", "cost"])
    df = pd.DataFrame(result.trades)
    df["cost_bps"] = df["cost"] / df["traded"].replace(0, np.nan) * 1e4
    return df


def round_trips(result) -> pd.DataFrame:
    """Collapse the blotter into completed round trips.

    A "trade" in a P&L discussion is a round trip, not a fill: entering and
    later flattening is one trade with one outcome. Reporting fills as trades
    doubles the count and halves the average size, which is how a strategy with
    12 real positions gets described as having 24 trades.
    """
    rows, open_pos = [], None
    for t in result.trades:
        if open_pos is None:
            if t["to_position"] != 0:
                open_pos = t
            continue
        # A flip (long -> short) closes the old trade and opens a new one.
        rows.append({
            "entered": open_pos["bar"], "exited": t["bar"],
            "position": open_pos["to_position"],
            "entry_price": open_pos["fill_price"], "exit_price": t["fill_price"],
            "bars_held": None,
            "gross_return": (t["fill_price"] / open_pos["fill_price"] - 1)
                            * np.sign(open_pos["to_position"]),
            "costs": open_pos["cost"] + t["cost"]})
        open_pos = t if t["to_position"] != 0 else None
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bars_held"] = (df["exited"] - df["entered"]).dt.days
        df["net_return"] = df["gross_return"] - df["costs"]
    return df


def monthly_returns(result) -> pd.DataFrame:
    r = result.returns.dropna()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    m.index.names = ["year", "month"]
    return m.unstack("month")


def tearsheet(result, benchmark=None, name: str = "strategy") -> str:
    dd, dur = result.max_drawdown()
    rt = round_trips(result)
    lines = []
    add = lines.append

    add("# Tearsheet -- {}".format(name))
    add("")
    add("| metric | value |")
    add("|---|---|")
    add("| bars | {} |".format(result.n_bars))
    add("| total return | {:.2%} |".format(result.total_return()))
    add("| Sharpe (252d, daily) | {:.3f} |".format(result.sharpe()))
    add("| Sortino (LPM2, MAR 0) | {:.3f} |".format(result.sortino()))
    add("| max drawdown | {:.2%} over {} bars |".format(dd, dur))
    add("| turnover (sum of |dw|) | {:.1f} |".format(result.turnover))
    add("| costs paid | {:.4f} of starting equity |".format(result.costs_paid))
    add("| fills | {} |".format(len(result.trades)))
    add("| round trips | {} |".format(len(rt)))
    if benchmark is not None:
        add("| benchmark total return | {:.2%} |".format(benchmark.total_return()))
        add("| benchmark Sharpe | {:.3f} |".format(benchmark.sharpe()))
        add("| excess return | {:+.2%} |".format(
            result.total_return() - benchmark.total_return()))

    if not rt.empty:
        wins = rt[rt.net_return > 0]
        add("")
        add("## Round trips")
        add("")
        add("| | |")
        add("|---|---|")
        add("| hit rate | {:.1%} ({} of {}) |".format(
            len(wins) / len(rt), len(wins), len(rt)))
        add("| mean net return | {:.3%} |".format(rt.net_return.mean()))
        add("| best / worst | {:.2%} / {:.2%} |".format(
            rt.net_return.max(), rt.net_return.min()))
        add("| median bars held | {:.0f} |".format(rt.bars_held.median()))
        add("| costs as % of gross | {:.1%} |".format(
            rt.costs.sum() / max(abs(rt.gross_return).sum(), 1e-12)))
        add("")
        add("A hit rate is not an edge. A strategy right 30% of the time with a")
        add("3:1 payoff beats one right 70% of the time at 1:3, so the hit rate is")
        add("printed next to the mean and never on its own.")

    add("")
    add("## Monthly returns")
    add("")
    m = monthly_returns(result)
    add("| year | " + " | ".join("{:02d}".format(c) for c in m.columns) + " |")
    add("|---" * (len(m.columns) + 1) + "|")
    for year, row in m.iterrows():
        cells = ["" if pd.isna(v) else "{:+.2%}".format(v) for v in row]
        add("| {} | ".format(year) + " | ".join(cells) + " |")

    add("")
    add("## Reconciliation")
    add("")
    b = blotter(result)
    tie_turnover = abs(b["traded"].sum() - result.turnover) if not b.empty else 0.0
    tie_costs = abs(b["cost"].sum() - result.costs_paid) if not b.empty else 0.0
    add("| check | residual |")
    add("|---|---|")
    add("| sum(blotter.traded) vs turnover | {:.2e} |".format(tie_turnover))
    add("| sum(blotter.cost) vs costs_paid | {:.2e} |".format(tie_costs))
    add("")
    add("Those two residuals are the reason the blotter exists. If either is")
    add("non-zero the summary above is describing a different portfolio from the")
    add("one the fills built.")
    return "\n".join(lines)
