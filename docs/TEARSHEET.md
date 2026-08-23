# Tearsheet -- momentum(20), 8bps round trip, synthetic walk

| metric | value |
|---|---|
| bars | 1500 |
| total return | 25.28% |
| Sharpe (252d, daily) | 0.359 |
| Sortino (LPM2, MAR 0) | 0.521 |
| max drawdown | -23.58% over 161 bars |
| turnover (sum of |dw|) | 123.0 |
| costs paid | 0.0615 of starting equity |
| fills | 123 |
| round trips | 61 |
| benchmark total return | 143.32% |
| benchmark Sharpe | 0.980 |
| excess return | -118.04% |

## Round trips

| | |
|---|---|
| hit rate | 26.2% (16 of 61) |
| mean net return | 0.406% |
| best / worst | 15.19% / -4.02% |
| median bars held | 6 |
| costs as % of gross | 3.6% |

A hit rate is not an edge. A strategy right 30% of the time with a
3:1 payoff beats one right 70% of the time at 1:3, so the hit rate is
printed next to the mean and never on its own.

## Monthly returns

| year | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | +0.00% | +0.00% | +0.00% | -1.61% | -4.57% | +4.35% | +1.41% | -1.50% | +4.87% | +0.83% | -0.15% | +3.83% |
| 2020 | +11.98% | -0.94% | +0.00% | -1.75% | -0.59% | +1.02% | +9.34% | +2.86% | -1.98% | -1.84% | +3.43% | +5.64% |
| 2021 | +1.41% | -2.92% | -3.35% | +2.13% | +1.36% | +0.56% | -0.44% | +9.55% | -8.07% | +0.00% | +0.00% | -2.92% |
| 2022 | +0.00% | +7.09% | -8.14% | +0.47% | +0.00% | +0.81% | -1.18% | +0.13% | +11.42% | -0.30% | -2.59% | -0.50% |
| 2023 | -6.36% | -1.74% | +0.24% | +2.27% | -9.60% | -0.64% | -1.16% | +7.94% | -3.17% | -0.76% | -1.44% | +3.23% |
| 2024 | +8.25% | -0.12% | -3.55% | +1.03% | -4.05% | +0.17% | -1.05% | -2.63% | +2.27% |  |  |  |

## Reconciliation

| check | residual |
|---|---|
| sum(blotter.traded) vs turnover | 0.00e+00 |
| sum(blotter.cost) vs costs_paid | 2.08e-17 |

Those two residuals are the reason the blotter exists. If either is
non-zero the summary above is describing a different portfolio from the
one the fills built.