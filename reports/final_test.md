# Final test: 2025-01-01 to 2026-10-10

Models fitted on fights before 2025-01-01; scored on 2025-01-01 to 2026-10-10. Settings were frozen in `config_frozen.json` before this run.

## 1. Scores

932 decided fights with usable opening and closing odds (77 event nights).

| model | accuracy | log_loss | brier | vs_opening | low | high |
|---|---|---|---|---|---|---|
| Elo only | 0.5660 | 0.6833 | 0.2449 | 0.0778 | 0.0554 | 0.0979 |
| A1 logistic, no odds | 0.6545 | 0.6288 | 0.2193 | 0.0233 | 0.0033 | 0.0411 |
| A2 boosting, no odds | 0.6588 | 0.6365 | 0.2226 | 0.0310 | 0.0112 | 0.0485 |
| A average | 0.6577 | 0.6309 | 0.2201 | 0.0254 | 0.0055 | 0.0428 |
| Opening odds | 0.6663 | 0.6055 | 0.2088 | 0.0000 | 0.0000 | 0.0000 |
| Stretched opening odds | 0.6663 | 0.6054 | 0.2089 | -0.0001 | -0.0045 | 0.0046 |
| B1 logistic + opening odds | 0.6770 | 0.5986 | 0.2056 | -0.0069 | -0.0157 | 0.0016 |
| B2 boosting + opening odds | 0.6803 | 0.6036 | 0.2077 | -0.0019 | -0.0106 | 0.0070 |
| B average (primary) | 0.6727 | 0.6000 | 0.2062 | -0.0055 | -0.0138 | 0.0029 |
| Closing odds | 0.6974 | 0.5842 | 0.1997 | -0.0213 | -0.0319 | -0.0106 |

- Primary model against the stretched opening odds: -0.0054 (95% interval -0.0118 to +0.0008). Stretch slope fitted before the test: 1.207.
- B1 weight on the opening odds: 1.009, fitted on 4788 fights.
- No-odds A average on all 951 decided fights in the window: accuracy 0.6562, log loss 0.6324.

### Calibration of the primary model

| band | fights | predicted | actual |
|---|---|---|---|
| (0.499, 0.55] | 87 | 0.527 | 0.494 |
| (0.55, 0.6] | 112 | 0.577 | 0.518 |
| (0.6, 0.65] | 123 | 0.624 | 0.561 |
| (0.65, 0.7] | 135 | 0.676 | 0.674 |
| (0.7, 0.75] | 141 | 0.723 | 0.667 |
| (0.75, 0.8] | 131 | 0.773 | 0.786 |
| (0.8, 0.85] | 111 | 0.823 | 0.820 |
| (0.85, 1.0] | 92 | 0.885 | 0.848 |

## 2. Betting at opening prices (edge above 5%, capped at 20%)

- Fights available: 944, of which 12 were draws or no contests.
- Bets: 322, 63.7% won, average price 1.79.
- Return: +5.2% per unit staked (95% interval -3.6% to +13.7%); +16.8 units.
- Closing-line value: +1.9% (95% interval +0.3% to +3.8%); 53% of bets beat the close.
- Worst peak-to-trough run: 9.4 units.

| check | bets | return | low | high | clv |
|---|---|---|---|---|---|
| Primary rule | 322 | 0.052 | -0.036 | 0.137 | 0.019 |
| Prices 2% worse than recorded | 285 | 0.038 | -0.046 | 0.128 | 0.009 |
| Prices 5% worse than recorded | 237 | 0.023 | -0.071 | 0.129 | 0.004 |
| Bet at the CLOSING price instead | 331 | -0.100 | -0.198 | -0.004 | -0.045 |
| Without the 5% of bets with the largest line moves | 305 | 0.042 | -0.048 | 0.129 | 0.002 |
| No cap on the edge | 341 | 0.059 | -0.022 | 0.143 | 0.027 |
| Favourites only | 227 | 0.027 | -0.054 | 0.104 | -0.012 |
| Underdogs only | 95 | 0.113 | -0.097 | 0.333 | 0.094 |
| Threshold 0% (not the frozen rule) | 654 | 0.036 | -0.025 | 0.096 | -0.001 |
| Threshold 2% (not the frozen rule) | 513 | 0.075 | 0.010 | 0.141 | 0.007 |
| Threshold 8% (not the frozen rule) | 183 | 0.073 | -0.054 | 0.198 | 0.034 |
| Threshold 10% (not the frozen rule) | 120 | 0.083 | -0.074 | 0.251 | 0.043 |
| B1 alone, same rule | 355 | 0.066 | -0.027 | 0.167 | 0.023 |
| B2 alone, same rule | 303 | 0.006 | -0.083 | 0.093 | 0.000 |
| Stretched opening odds, any positive edge | 487 | -0.023 | -0.075 | 0.025 | -0.055 |
| Year 2025 | 173 | 0.092 | -0.016 | 0.203 | 0.015 |
| Year 2026 | 149 | 0.006 | -0.124 | 0.142 | 0.024 |

## 3. Pass rule

1. Lower log loss than the opening odds, interval below zero: **FAIL** (-0.0055, -0.0138 to +0.0029)
2. Lower log loss than the stretched opening odds, interval below zero: **FAIL** (-0.0054, -0.0118 to +0.0008)
3. Closing-line value above zero, interval above zero: **PASS** (+0.0190, +0.0027 to +0.0380)

**Overall: NOT PASSED** (1 of 3).
