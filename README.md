# ML-2 — Walk-Forward Backtesting Engine

**Status: ~20% slice.** The engine and its self-deception defences are built. The
universe, the real data, and the reporting layer are not.

**There is no strategy claim in this repo.** The two strategies are test cargo
for the engine, run on a synthetic random walk that has no signal in it by
construction — which makes it better engine test data than real prices, because
anything that looks good on it is a bug.

```bash
python run_audit.py
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

## What is NOT built (the other 80%)

1. **Survivorship**: no universe, no delisted tickers, no bound on the bias.
   Single synthetic series only.
2. **Real data.** No yfinance/vendor loader, no corporate actions, no
   point-in-time snapshots. The engine has never seen a real price.
3. **Participation-based slippage.** The slippage model is fixed-bps; a
   volume-participation model (and the market-impact discussion that goes with
   it) is missing.
4. **Portfolio-level backtesting**: one instrument, one position in [-1, 1]. No
   cross-sectional ranking, no sizing, no risk limits, no leverage or margin.
5. **vectorbt cross-check** of the engine's own arithmetic.
6. **Full deflated Sharpe** with skew/kurtosis adjustment, and White's reality
   check / SPA as an alternative.
7. Reporting: no tearsheet, no equity-curve plots, no per-trade log.
