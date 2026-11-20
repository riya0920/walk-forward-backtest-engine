# Bias audit

One row per bias, what defends against it, and, where nothing does, that.

| Bias | Defence in this repo | Status |
|---|---|---|
| Look-ahead: reading a bar you couldn't have had | `PointInTimeView` slices at the cursor; there is no method that returns future rows. `view.at()` raises `LookAheadError`. | **Structural** |
| Look-ahead: executing at the signal's own price | The event loop fills at `t+1`'s open. The strategy has no say. | **Structural** |
| Look-ahead: restated / point-in-time-incorrect inputs | Nothing. Prices here are synthetic and never restated. Real data needs vendor PIT snapshots. | **NOT DEFENDED** |
| Survivorship | Universe with delistings booked at their delisting return; same strategy run on the as-it-was universe vs survivors only. | **Measured: +0.64 Sharpe inflation on reversal** |
| Overfitting | Walk-forward: parameters chosen only on the training window, applied untouched to the next. IS and OOS both reported. | **Built** |
| Multiple testing | Every variant counted (147 in the current run), compared against E[max Sharpe] under the null. | **Built** |
| Costs | Commission + half-spread + slippage on every fill, **inside the return series**. Sensitivity at 0/5/10/20bps. | **Built** |
| Capacity | Participation-based square-root impact; total return turns negative at about $10m, Sharpe crosses zero between $10m and $25m AUM. | **Built** |

## The three doors look-ahead comes through

1. **Reading a bar you could not have had yet**: today's close used at today's
   open. Closed structurally: the view object physically cannot return it.
2. **Executing at the price that generated the signal**: signal from today's
   close, filled at today's close. Closed in the loop: fills are at `t+1` open.
   `test_fills_happen_at_the_next_open_not_at_the_signal_bar` pins it.
3. **Point-in-time incorrectness of the inputs themselves**: a universe or a
   fundamentals table assembled today and applied to 2015. This is a data
   sourcing problem, not an engine problem, and **this repo does not solve it.**

## The planted-leak test

A strategy that peeks one bar ahead is run through the same engine. On the
synthetic random walk it produces an annualised Sharpe far above anything the
honest strategies reach, and the harness asserts that gap. The threshold (3.0) is
calibrated in both directions: the leak must trip it, and momentum and
mean-reversion must *not*; a detector that flags everything is useless.

## In-sample vs out-of-sample decay (current run)

| strategy | mean IS Sharpe | mean OOS Sharpe | decay |
|---|---|---|---|
| momentum | 1.14 | 0.37 | 0.77 |
| mean_reversion | 0.18 | −0.43 | 0.61 |

The data is a random walk. Both true Sharpes are zero. The IS column is the
search finding patterns in noise, and the size of the decay is the point.

Every OOS number is scored on all 100 test bars, with the strategy allowed to
read the training prices before them (`out_of_sample()` in `run_audit.py`). An
earlier version backtested the test bars on their own, so the 60-bar warm-up
left at most 39 of 100 traded; `tests/test_walk_forward.py` pins the fix.

## Multiple testing (current run)

```
variants evaluated (counted, including losers): 147
best OOS Sharpe observed                      : 3.177
Sharpe standard error at 100-bar OOS window   : 1.587
E[max Sharpe] under the null, 147 trials      : 4.228
excess over the null-search benchmark         : -1.051
verdict                                       : indistinguishable from search noise
```

Method: expected maximum Sharpe under the null of zero skill (Bailey & López de
Prado's deflated-Sharpe construction, simple form, skew/kurtosis adjustment not
applied, and that omission makes this *optimistic*, not conservative).

The `n_days` input is the length of the window the Sharpe was measured on (100
bars), not the 1,500-bar series. An earlier version of this script used the full
series length; that shrinks the standard error roughly fourfold and flips the
verdict to "significant". It was wrong, and it is the exact error this section
exists to catch.

## Sharpe methodology

Annualised as `mean(daily returns) / std(daily returns) × √252`, computed on the
**net** return series with costs included. Stated because daily×√252 and
monthly-compounded give different numbers and "Sharpe 1.4" without the method is
not a number.
