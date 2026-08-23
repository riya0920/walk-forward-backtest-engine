"""Bootstrap data-snooping tests over the FULL candidate set, not just the winner.

`run_audit.py` reports the winner and adjusts it with a closed-form benchmark
that needs only the trial count. This script runs the two bootstrap tests that
use the actual joint return matrix instead, and then demonstrates the one place
where the choice between them changes the answer.

    python run_reality_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.backtest import CostModel, buy_and_hold, run
from engine.reality_check import (hansens_spa, outperformance_matrix,
                                  whites_reality_check)
from engine.strategies import make_synthetic_prices, mean_reversion, momentum

COSTS = CostModel()
MOM_GRID = [3, 5, 8, 10, 15, 20, 30, 40, 60, 90]
REV_GRID = [5, 10, 15, 20, 30, 50, 80]


def _line(title):
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)


def _report(res: dict) -> None:
    print(res["test"])
    print("  strategies compared        : {}".format(res["n_strategies"]))
    print("  periods                    : {}".format(res["n_periods"]))
    print("  block length (mean)        : {}".format(res["block_len"]))
    print("  statistic                  : {:.4f}".format(res["statistic"]))
    print("  p-value                    : {:.4f}".format(res["p_value"]))
    if "n_treated_as_null" in res:
        print("  recentred on own mean      : {}".format(
            res["n_recentred_on_own_mean"]))
        print("  treated as exactly null    : {}".format(res["n_treated_as_null"]))
    print("  verdict                    : {}".format(res["verdict"]))


def build_candidates(prices):
    labels, series = [], []
    for p in MOM_GRID:
        labels.append("momentum({})".format(p))
        series.append(run(prices, momentum(p), COSTS).returns.to_numpy())
    for p in REV_GRID:
        labels.append("reversion({})".format(p))
        series.append(run(prices, mean_reversion(p), COSTS).returns.to_numpy())
    return labels, series


def build_discarded(prices, n=30):
    """Variants a researcher would actually have tried and thrown away.

    A fixed-period sign flip on the underlying, paying the full round-trip
    spread every bar. They are real candidate strategies with a real reason to
    lose (turnover), not random noise columns -- if the padding were noise the
    demonstration would be rigged.
    """
    under = prices["close"].pct_change().fillna(0).to_numpy()
    out = []
    for j in range(n):
        period = j + 2
        sign = np.where((np.arange(len(under)) // period) % 2 == 0, 1.0, -1.0)
        out.append(under * sign - 2 * COSTS.total_bps / 1e4)
    return out


def main() -> None:
    prices = make_synthetic_prices()
    labels, series = build_candidates(prices)
    bench = buy_and_hold(prices).returns.to_numpy()
    d = outperformance_matrix(series, bench)

    print("=" * 78)
    print("DATA-SNOOPING TESTS OVER THE WHOLE CANDIDATE SET")
    print("=" * 78)
    print("{} variants x {} bars.".format(d.shape[1], d.shape[0]))
    print("Performance measure: per-bar return differential vs a benchmark.")
    print("Resampling: stationary bootstrap, mean block length 5 bars, so serial")
    print("dependence and cross-strategy correlation both survive the resample.")

    _line("1. THE CANDIDATES (differential vs buy and hold)")
    print("{:<18}{:>14}{:>16}".format("variant", "mean d (bps)", "ann. Sharpe"))
    for k, name in enumerate(labels):
        col = d[:, k]
        sr = col.mean() / col.std() * np.sqrt(252) if col.std() else 0.0
        print("{:<18}{:>14.3f}{:>16.3f}".format(name, col.mean() * 1e4, sr))

    _line("2. VS BUY AND HOLD")
    rc = whites_reality_check(d)
    _report(rc)
    spa = hansens_spa(d)
    print()
    _report(spa)
    print("\nBoth land near p = 0.99, which is the right answer and an uninformative")
    print("one: on a drifting random walk nothing beats holding it, so the test is")
    print("pinned at the ceiling. A test at its ceiling cannot show anything about")
    print("itself, which is why the comparison below uses a benchmark the")
    print("candidates can actually get close to.")

    # ------------------------------------------------------- cash benchmark
    zero = np.zeros(len(bench))
    dz = outperformance_matrix(series, zero)

    _line("3. VS CASH -- the region where these tests discriminate")
    rc0 = whites_reality_check(dz)
    _report(rc0)
    print("\n  best variant: {}".format(labels[rc0["best_index"]]))
    spa0 = hansens_spa(dz)
    print()
    _report(spa0)
    print("\nRC p = {:.3f} is *just* inside the 5% line. Hold on to that.".format(
        rc0["p_value"]))

    # ---------------------------------------------------------------- padding
    _line("4. WHY THE TWO TESTS DISAGREE: PADDING WITH DISCARDED VARIANTS")
    print("RC takes a maximum over every column, so its null distribution is")
    print("inflated by strategies that never had a chance. SPA recentres those to")
    print("zero instead of counting them as evidence. The consequence is testable:")
    print("add variants that are known to be bad and watch only one p-value move.\n")

    junk = build_discarded(prices, 30)
    print("discarded-variant mean return: {:.2f} bps/bar -- these are not marginal"
          .format(float(np.mean([c.mean() for c in junk])) * 1e4))
    print()
    print("{:<30}{:>6}{:>10}{:>10}{:>14}".format(
        "candidate set", "K", "RC p", "SPA p", "SPA nulled"))
    for extra in (0, 5, 10, 20, 30):
        dd = outperformance_matrix(series + junk[:extra], zero)
        r = whites_reality_check(dd)
        s = hansens_spa(dd)
        print("{:<30}{:>6}{:>10.4f}{:>10.4f}{:>14}".format(
            "real + {} discarded".format(extra), dd.shape[1],
            r["p_value"], s["p_value"], s["n_treated_as_null"]))

    print("\nRC crosses its own 5% line between the first and second row. Nothing")
    print("about the strategy changed -- only how many losers stayed in the")
    print("spreadsheet. By the last row RC has drifted an order of magnitude while")
    print("SPA has not moved at all, and its `nulled` column tracks the padding")
    print("exactly: those columns are recentred to zero rather than being allowed")
    print("to raise the bar for everyone else.")
    print("\nRead that in both directions. RC punishes an honest researcher who")
    print("reports every variant tried, and rewards one who quietly drops the bad")
    print("ones before running the test. That incentive is backwards, and it is")
    print("the reason Hansen wrote SPA.")

    _line("5. WHAT NEITHER TEST COVERS")
    print("Both test the variants that were RUN. Anything abandoned before it was")
    print("recorded -- a feature dropped by eye, a date range quietly moved, the")
    print("choice to study this asset at all -- is invisible to both, and no")
    print("bootstrap can recover it. The variant counter in run_audit.py is the")
    print("only defence there, and it is a discipline rather than a statistic.")
    print("=" * 78)


if __name__ == "__main__":
    main()
