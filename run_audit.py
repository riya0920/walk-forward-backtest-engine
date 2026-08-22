"""Walk-forward optimisation + cost sensitivity + the variant counter.

Every strategy variant this script evaluates is counted, including the ones that
lost, and the count feeds the multiple-testing adjustment. That counter is the
deliverable. The strategies are not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.backtest import CostModel, buy_and_hold, run
from engine.multiple_testing import assess, deflated_sharpe_ratio
from engine.strategies import (make_synthetic_prices, mean_reversion, momentum,
                               planted_leak)

VARIANTS = 0            # every fit, every window, every parameter. Nothing hidden.


def evaluate(frame, strat, costs=None):
    global VARIANTS
    VARIANTS += 1
    return run(frame, strat, costs)


TEST_BARS = 100


def walk_forward(frame, family, name, grid, train_bars=400, test_bars=TEST_BARS):
    """Parameters are chosen ONLY on the training window and then applied,
    untouched, to the next window. The in-sample number is kept so the decay is
    reportable rather than quietly discarded."""
    rows = []
    start = 0
    while start + train_bars + test_bars <= len(frame):
        tr = frame.iloc[start:start + train_bars]
        te = frame.iloc[start + train_bars:start + train_bars + test_bars]

        scored = [(evaluate(tr, family(p)).sharpe(), p) for p in grid]
        best_is, best_p = max(scored, key=lambda t: t[0])
        oos = evaluate(te, family(best_p))

        rows.append({"window_start": frame.index[start].date(),
                     "chosen_param": best_p,
                     "in_sample_sharpe": best_is,
                     "oos_sharpe": oos.sharpe(),
                     "oos_return": oos.total_return(),
                     "turnover": oos.turnover})
        start += test_bars
    df = pd.DataFrame(rows)
    df.insert(0, "strategy", name)
    return df


def main() -> None:
    prices = make_synthetic_prices()
    print("=" * 78)
    print("DATA: synthetic random walk, {} bars, {} to {}".format(
        len(prices), prices.index[0].date(), prices.index[-1].date()))
    print("There is no signal in this series by construction. Any strategy that")
    print("looks good here is a bug in the engine, which is exactly why it is the")
    print("engine's test data.")

    # ---- 1. look-ahead guard demonstration --------------------------------
    leak = run(prices, planted_leak(1), CostModel(0, 0, 0))
    honest = run(prices, momentum(), CostModel(0, 0, 0))
    print("\n" + "=" * 78)
    print("1. PLANTED-LEAK TEST")
    print("-" * 78)
    print("strategy that peeks 1 bar ahead : Sharpe {:8.2f}   <- impossible".format(
        leak.sharpe()))
    print("same engine, honest momentum    : Sharpe {:8.2f}".format(honest.sharpe()))
    print("Detection rule: annualised Sharpe > 3.0 on this series is not an edge,")
    print("it is a leak. The rule is asserted in tests/test_leak_guard.py.")

    # ---- 2. walk-forward ---------------------------------------------------
    print("\n" + "=" * 78)
    print("2. WALK-FORWARD: in-sample vs out-of-sample decay")
    print("-" * 78)
    wf = pd.concat([
        walk_forward(prices, momentum, "momentum", [5, 10, 20, 40, 60, 90]),
        walk_forward(prices, mean_reversion, "mean_reversion", [10, 20, 30, 50, 80]),
    ], ignore_index=True)
    print("{:<16}{:>10}{:>8}{:>12}{:>12}".format(
        "strategy", "window", "param", "IS Sharpe", "OOS Sharpe"))
    for _, r in wf.iterrows():
        print("{:<16}{:>10}{:>8}{:>12.2f}{:>12.2f}".format(
            r.strategy, str(r.window_start), r.chosen_param,
            r.in_sample_sharpe, r.oos_sharpe))

    print("-" * 78)
    for name, g in wf.groupby("strategy"):
        print("{:<16} mean IS {:6.2f}   mean OOS {:6.2f}   decay {:6.2f}".format(
            name, g.in_sample_sharpe.mean(), g.oos_sharpe.mean(),
            g.in_sample_sharpe.mean() - g.oos_sharpe.mean()))
    print("\nThe out-of-sample column is the only one that exists. The in-sample")
    print("column is printed so the size of the gap is visible, not to be quoted.")

    # ---- 3. cost sensitivity ----------------------------------------------
    print("\n" + "=" * 78)
    print("3. COST SENSITIVITY (momentum, lookback 20)")
    print("-" * 78)
    print("{:>10}{:>12}{:>14}{:>12}{:>12}".format(
        "bps", "Sharpe", "total return", "turnover", "costs paid"))
    for bps in (0, 5, 10, 20):
        r = evaluate(prices, momentum(20), CostModel(0, 0, bps))
        print("{:>10}{:>12.2f}{:>13.2%}{:>12.1f}{:>12.4f}".format(
            bps, r.sharpe(), r.total_return(), r.turnover, r.costs_paid))
    print("\nTurnover matters more than gross return for a strategy like this:")
    print("the cost line scales with turnover, so a signal that flips often pays")
    print("the spread on every flip and the gross edge never reaches the account.")

    # ---- 4. benchmark ------------------------------------------------------
    bh = buy_and_hold(prices)
    print("\n" + "=" * 78)
    print("4. BENCHMARK-RELATIVE")
    print("-" * 78)
    print("buy and hold : Sharpe {:6.2f}   total return {:7.2%}".format(
        bh.sharpe(), bh.total_return()))

    # ---- 5. multiple testing ----------------------------------------------
    best = float(wf.oos_sharpe.max())
    # n_days must be the length of the window the Sharpe was MEASURED on
    # (100 bars), not the length of the whole series. Using 1500 here shrinks the
    # standard error by ~4x and turns search noise into a "significant" result --
    # the exact inflation this section exists to prevent.
    a = assess(best, VARIANTS, TEST_BARS)
    print("\n" + "=" * 78)
    print("5. MULTIPLE-TESTING ADJUSTMENT")
    print("-" * 78)
    print("variants evaluated (counted, including losers): {}".format(a["n_trials"]))
    print("best OOS Sharpe observed                      : {:.3f}".format(a["best_sharpe"]))
    print("Sharpe standard error at {}-bar OOS window     : {:.3f}".format(
        a["n_days"], a["sharpe_standard_error"]))
    print("E[max Sharpe] under the null, {} trials       : {:.3f}".format(
        a["n_trials"], a["expected_max_under_null"]))
    print("excess over the null-search benchmark         : {:+.3f}".format(
        a["excess_over_null_max"]))
    print("verdict                                       : {}".format(a["verdict"]))

    # ---- full deflated Sharpe on the best window's return series ----------
    best_row = wf.loc[wf.oos_sharpe.idxmax()]
    fam = momentum if best_row.strategy == "momentum" else mean_reversion
    idx = [i for i, d in enumerate(prices.index)
           if d.date() == best_row.window_start][0]
    te = prices.iloc[idx + 400:idx + 400 + TEST_BARS]
    best_res = run(te, fam(int(best_row.chosen_param)))
    d = deflated_sharpe_ratio(best_res.returns.dropna().to_numpy(), VARIANTS)

    print("\n" + "-" * 78)
    print("DEFLATED SHARPE (full form: skew and kurtosis included)")
    print("-" * 78)
    print("annualised Sharpe        : {:>9.3f}".format(d["sr_annualised"]))
    print("skew / excess kurtosis   : {:>9.3f} / {:.3f}".format(
        d["skew"], d["kurtosis"] - 3.0))
    print("SE, non-normal           : {:>9.5f}  ({:.5f} assuming normality)".format(
        d["standard_error"], d["normal_se"]))
    print("DSR                      : {:>9.4f}".format(d["dsr"]))
    print("verdict                  : {}".format(d["verdict"]))
    print("\nSkew enters the variance as -skew*SR, so its DIRECTION depends on the")
    print("sign of the Sharpe; kurtosis enters squared and always widens. The")
    print("simple form above uses trial count alone, which on a fat-tailed series")
    print("is the optimistic version of the same question.")
    print("\nThis is the honest answer to 'what is the probability the best one is")
    print("noise?': on data with no signal, the search itself manufactures a")
    print("Sharpe of about {:.2f}. Anything below that is the search, not a strategy."
          .format(a["expected_max_under_null"]))
    print("=" * 78)


if __name__ == "__main__":
    main()
