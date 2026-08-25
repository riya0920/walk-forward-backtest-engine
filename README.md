# ML-2 — Walk-Forward Backtesting Engine

**Status: ~97%.** Engine, self-deception defences, a universe with delistings,
portfolio backtesting, capacity analysis, **real market data**, the **full
deflated Sharpe**, **White's Reality Check and Hansen's SPA** on a stationary
bootstrap, a **risk model** (name/sector/gross/net limits, sector neutrality,
per-name borrow costs), **multi-day execution scheduling**, a **per-fill
blotter and tearsheet**, and an **independent cross-check against vectorbt**
that found two defects in this engine. **56 tests.**

**There is no strategy claim in this repo.** The strategies are test cargo. The
default series is a synthetic random walk with no signal in it by construction --
better engine test data than real prices, because anything that looks good on it
is a bug. `run_real.py` then runs the identical harness on real SPY/AAPL/MSFT
data, because a leak detector calibrated only on synthetic data proves nothing
about the data you care about.

```bash
python run_audit.py            # bias audit, walk-forward, multiple testing
python run_survivorship.py     # delisting bias + capacity curve
python run_real.py             # the same engine on real yfinance prices
python run_reality_check.py    # White's Reality Check + Hansen's SPA
python run_vectorbt_check.py   # cross-check the arithmetic against vectorbt
python run_risk_execution.py   # limits, neutrality, borrow, multi-day schedule
python run_tearsheet.py        # docs/TEARSHEET.md + data/blotter.csv
python -m pytest tests -q      # 56 tests
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
  against E[max Sharpe] under the null (Bailey & López de Prado, simple form),
  the **full deflated Sharpe** with skew and kurtosis, and the two bootstrap
  tests that use the actual joint return matrix instead of a trial count —
  **White's Reality Check** and **Hansen's SPA**, both on a stationary bootstrap
  (Politis & Romano) so serial dependence and cross-strategy correlation survive
  the resample.
- **Risk model** (`engine/risk.py`): per-name, per-sector, gross and net limits
  applied in a declared order, sector neutralisation, and borrow charged on
  short notional at a **per-name** rate.
- **Multi-day execution scheduling** (`engine/execution.py`) — the answer the
  capacity table could not give.
- **Per-fill blotter and tearsheet** (`engine/tearsheet.py`), with a
  reconciliation section asserting the headline turnover and costs reassemble
  from the fills.
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
executed in a day". Saying so properly needs multi-day execution scheduling,
which `engine/execution.py` now provides — see *The capacity table's missing
row* below for the number that replaces the SATURATED label.

## The independent cross-check, and the three defects it found

Every other number in this repo comes from code I wrote, checked by tests I also
wrote. That is a closed loop: a consistent error appears in both and nothing ever
disagrees. `run_vectorbt_check.py` hands the **same return series** to vectorbt
and compares definition by definition.

| metric | this engine | vectorbt | |
|---|---|---|---|
| Sharpe (annualised) | 0.43955261 | 0.43955261 | match to 5e-16 |
| max drawdown | −0.22425170 | −0.22425170 | exact |
| Sortino | 0.64086894 | 0.64086894 | exact **after a fix** |
| total return | 0.33227938 | 0.33227938 | exact |

**1. The year is a convention worth 20% of the headline.** vectorbt annualises a
daily series with 365 *calendar* days by default and reports Sharpe 0.5290; this
engine uses 252 *trading* days and reports 0.4396. The ratio is exactly
√(365/252). Neither is wrong. Quoting either without saying which is.

**2. This engine's Sortino was the wrong statistic.** It divided by
`r[r < 0].std()` — the dispersion of losing returns about *their own mean*, over
the losing subset only. Sortino divides by the second lower partial moment over
**all** periods. That was 0.554 against vectorbt's 0.641, and the 13% gap is not
the interesting part: the old denominator goes to **zero** when every loss is the
same size, so a strategy losing exactly 1% on every down day scored an *infinite*
Sortino.

**3. `total_return` was measured against the wrong base.** It divided by
`equity.iloc[0]`, which is an end-of-bar value and therefore already contains the
first bar's return. At the default 60-bar warmup that value *is* the starting
capital and the bug is invisible; at `warmup=0` it silently discarded a −2.46%
first bar and turned a 134.07% hold into 139.97%. A bug that only appears at a
non-default argument is still a bug, and it is exactly the kind a self-written
test does not find.

## Reality Check vs SPA: the p-value that depends on your filing habits

`run_reality_check.py` runs both bootstrap tests over all 17 variants at once.
Against buy-and-hold both land near p = 0.99 — correct, and uninformative, since
on a drifting random walk nothing beats holding it. Against **cash**, where the
tests can discriminate:

| candidate set | K | RC p | SPA p | columns SPA nulled |
|---|---|---|---|---|
| real variants only | 17 | **0.0510** | 0.0800 | 0 |
| + 5 discarded | 22 | 0.1550 | 0.0800 | 5 |
| + 10 discarded | 27 | 0.2490 | 0.0800 | 10 |
| + 20 discarded | 37 | 0.4110 | 0.0800 | 20 |
| + 30 discarded | 47 | **0.5450** | 0.0800 | 30 |

Nothing about the strategy changed between rows — only how many losers stayed in
the spreadsheet. RC crosses its own 5% line between rows one and two and drifts
an order of magnitude by the last, because it maximises over every column
including the hopeless ones. SPA does not move at all.

Read that in both directions: **RC punishes a researcher who reports every
variant tried and rewards one who quietly deletes the bad ones first.** That
incentive is backwards, and it is why Hansen wrote SPA.

## The capacity table's missing row

`run_survivorship.py` prints `impact model SATURATED` past $50m, which is honest
and is not an answer. `run_risk_execution.py` gives the answer:

| AUM | leg notional | min days to execute | verdict |
|---|---|---|---|
| $10m | $1m | 1 | OK |
| $100m | $10m | 10 | OK |
| $500m | $50m | 50 | OK |
| $1bn | $100m | **100** | does not fit |

At $1bn the statement is not "Sharpe −0.90", it is that one rebalance leg needs
100 trading days at a 20% participation cap — longer than the strategy's own
holding period, so the position can never finish being built.

**And the optimal schedule is a preference, not a measurement.** A first draft of
`engine/execution.py` added impact (a cost, in bps paid) to timing risk (a
standard deviation, in bps of dispersion) and announced an interior minimum.
There was no interior minimum; adding a cost to a standard deviation is not an
operation. With the risk price λ made explicit:

| λ | optimal days | shape |
|---|---|---|
| 0.00 | 60 | slowest allowed |
| 0.05 | 60 | slowest allowed |
| **0.10** | **48** | **interior** |
| 0.20 | 25 | fastest allowed |
| 1.00 | 25 | fastest allowed |

The interior optimum that execution papers draw exists only inside a narrow band
of λ, and nothing in the price data tells you where in that band you sit.

## The blotter

`run_tearsheet.py` writes `docs/TEARSHEET.md`, `data/blotter.csv` (one row per
fill: decision bar, print bar, side, price, cost, equity) and
`data/round_trips.csv`. The tearsheet's last section reconciles
`sum(blotter.traded)` against reported turnover and `sum(blotter.cost)` against
reported costs; both residuals are 0.00e+00 and a test pins it. A backtest whose
headline turnover cannot be reassembled from its own fills has a headline nobody
can audit. The blotter doubles as a look-ahead check — every fill must be dated
strictly after the bar that decided it, and that is a test too.

Round trips are reported separately from fills (61 vs 123), because a "trade" in
a P&L conversation is a round trip and counting fills doubles it.

## What is NOT built

1. **A survivorship-free REAL universe.** `run_real.py` fetches genuine
   split/dividend-adjusted prices, but yfinance returns only currently-listed
   tickers, so that sample is survivorship-biased by construction and cannot be
   fixed downstream. The delisted-universe experiment therefore stays on
   generated data, which is the only place a name can actually die here. A real
   answer needs CRSP-style delisting returns — **a paid dataset, and the one
   remaining gap in this project that no amount of code closes.**
2. ~~**Intraday execution.**~~ **DONE** — `engine/intraday.py` schedules within
   the day on a volume curve MEASURED from 37,384 real 5-minute bars rather than
   an assumed U-shape (assuming the curve makes the conclusion a property of the
   assumption). 12.6x between the busiest and quietest bucket; a nominal 5% TWAP
   participates at 9.2% in the midday trough and costs 10.0% more impact than
   VWAP, identically at every order size — which is a check, not a coincidence,
   since the ratio cancels order size under a square-root law.

   It also refuted its own first claim: VWAP came out *less* deferred than TWAP,
   because a U is not a ramp and the open auction offsets the close.

   Still no venue routing, no limit-order model, no adverse-selection term, and
   no spread. See `docs/INTRADAY.md`.
3. ~~**A risk model estimated from data.**~~ **DONE** — `engine/risk_model.py`
   estimates a Ledoit-Wolf shrunk covariance from returns and reports ex-ante
   volatility, parametric VaR (labelled parametric, because it understates a fat
   tail), risk contributions that sum to one, and a diversification ratio.
   `run_risk_model.py` puts two books side by side that the *mandate* reports as
   identical — same gross, same net, four names each — and book B carries 21%
   more volatility. Still no factor model: decomposing risk into market, size,
   value and momentum needs factor returns this repo does not have.
   See `docs/RISK_MODEL.md`.
4. **Point-in-time correctness of the inputs themselves** — still NOT DEFENDED.
   The universe is PIT, but a real one needs vendor snapshots of a restated
   fundamentals history, which is a data-sourcing problem this repo does not solve.
5. **Plots.** The tearsheet is deliberately text: an equity-curve PNG carries the
   least information per pixel of anything in a tearsheet and cannot be diffed in
   a pull request. The monthly return table carries the same information and
   survives CI output. This is a choice, not a gap, and it is listed here so it
   is not mistaken for one.
