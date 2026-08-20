# ML-2 — Walk-Forward Backtesting Engine

**Status: ~85%.** Engine, self-deception defences, a universe with delistings,
portfolio backtesting, capacity analysis, and **real market data**. CI runs the
bias audit and survivorship experiment on every push.

**There is no strategy claim in this repo.** The strategies are test cargo. The
default series is a synthetic random walk with no signal in it by construction --
better engine test data than real prices, because anything that looks good on it
is a bug. `run_real.py` then runs the identical harness on real SPY/AAPL/MSFT
data, because a leak detector calibrated only on synthetic data proves nothing
about the data you care about.

```bash
python run_audit.py           # bias audit, walk-forward, multiple testing
python run_survivorship.py    # delisting bias + capacity curve
python run_real.py            # the same engine on real yfinance prices
python -m pytest tests -q
```

## What is built

- **Structural look-ahead prevention** (`engine/data.py`). `PointInTimeView` has
  no method that returns rows after the cursor. Not a convention — there is no
  API for it. The single deliberate hole, `_unsafe_full_frame()`, exists so the
  leak test can plant a leak, and its name is its documentation.
- **`t → t+1` fill discipline** enforced by the loop, not by the strategy.
- **Planted-leak test** (`tests/test_leak_guard.py`): a future-peeking strategy
  must produce an impossible Sharpe, *and* the honest strategies must not trip
  the same threshold.
- **Walk-forward optimisation** with in-sample and out-of-sample both reported.
- **Multiple-testing counter**: every variant counted including losers, compared
  against E[max Sharpe] under the null (Bailey & López de Prado, simple form).
- **Cost model** inside the return series, with a 0/5/10/20bps sensitivity table.
- **Metrics**: Sharpe (method stated), Sortino, max drawdown + duration,
  turnover, benchmark-relative vs buy-and-hold.
- [docs/BIAS_AUDIT.md](docs/BIAS_AUDIT.md) — one row per bias, including the two
  rows that say **NOT DEFENDED**.

## Two bugs this build actually caught

1. **Costs were charged to equity but left out of the return series**, so Sharpe
   was identical at 0bps and 20bps while total return fell — every risk-adjusted
   metric was describing a portfolio nobody holds. Fixed; Sharpe now goes
   0.46 → 0.39 → 0.31 → 0.16 across the cost ladder.
2. **The significance test used the wrong sample length** — a 100-bar OOS Sharpe
   benchmarked against a 1,500-bar standard error, shrinking the SE ~4× and
   turning search noise into a "significant" result. Fixed; the verdict flipped
   to *indistinguishable from search noise*, which is the correct answer on a
   random walk.

## Survivorship, measured (`python run_survivorship.py`)

The bias row that used to read **NOT DEFENDED** now has a number. A 60-name
universe where names actually die — 21.7% delisted, booked at their delisting
return (bankruptcy −100%, compliance −55%, acquisition +18%) rather than dropped,
because dropping the row converts a total loss into "no position" and *that one
line is most of survivorship bias*.

Same strategy, same dates, same engine, run twice:

| run | Sharpe | total return | max drawdown | delist hits |
|---|---|---|---|---|
| momentum (as it was) | 0.59 | 29.3% | −13.2% | 1 |
| momentum (survivors only) | 0.88 | 47.9% | −8.4% | 0 |
| reversal (as it was) | 0.07 | 1.0% | −25.0% | 6 |
| reversal (survivors only) | 0.71 | 38.3% | −9.7% | 0 |

Reversal inflates by **+0.64 Sharpe and +37.4% return** — far more than momentum,
because a mean-reversion strategy buys exactly the names that are falling, which
is exactly the population that dies. Note the drawdown column: survivorship
flatters *risk* more than it flatters return, and risk is what position size is
set from.

## Capacity

Participation-based impact (`total_bps = 3 + 120·√participation`) replaces the
fixed-bps model. The edge crosses zero between $1m and $10m of AUM. Past $50m the
table prints **impact model SATURATED** — one rebalance leg is the entire day's
volume, so the model stops growing and every larger AUM reports the same number.
The honest statement there is not "Sharpe −0.90", it is "this trade cannot be
executed in a day", and saying so properly needs multi-day execution scheduling
the engine does not have.

## What is NOT built

1. **A survivorship-free REAL universe.** `run_real.py` fetches genuine
   split/dividend-adjusted prices, but yfinance returns only currently-listed
   tickers, so that sample is survivorship-biased by construction and cannot be
   fixed downstream. The delisted-universe experiment therefore stays on
   generated data, which is the only place a name can actually die here. A real
   answer needs CRSP-style delisting returns.
2. **Multi-day execution scheduling**, without which capacity above the
   saturation point is unmodellable (see above).
3. **Risk model**: no sector/factor neutrality, no position limits, no leverage
   or margin, no borrow costs for the short side.
4. **vectorbt cross-check** of the engine's own arithmetic.
5. **Full deflated Sharpe** with skew/kurtosis adjustment, and White's reality
   check / SPA as an alternative.
6. Reporting: no tearsheet, no equity-curve plots, no per-trade log.
7. **Point-in-time correctness of the inputs themselves** — still NOT DEFENDED.
   The universe is PIT, but a real one needs vendor snapshots of a restated
   fundamentals history, which is a data-sourcing problem this repo does not solve.
