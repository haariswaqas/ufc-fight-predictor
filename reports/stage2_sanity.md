# Stage 2 sanity checks

All on 2019-2024 out-of-sample predictions of the B1 model (logistic regression on top of the opening odds), betting rule: edge above 5%, capped at 20%.

| run | log_loss | bets | return | low | high | clv |
|---|---|---|---|---|---|---|
| B1 as trained | 0.6212 | 1194 | 0.0838 | 0.0318 | 0.1424 | 0.0027 |
| Placebo: shuffled features, mean of 20 | 0.6332 | 619 | -0.0588 | -0.2087 | 0.0177 |  |
| Without any age feature | 0.6239 | 1041 | 0.0441 | -0.0096 | 0.1003 | -0.0076 |
| Age features only | 0.6261 | 593 | 0.0817 | 0.0100 | 0.1517 | 0.0309 |
| Same recipe on CLOSING odds, bets at closing prices | 0.6041 | 998 | 0.0228 | -0.0407 | 0.0873 | -0.0422 |

- Opening odds log loss on the same fights: 0.6302. In the placebo `low`/`high` are the smallest and largest of the 20 runs.
- Independent recomputation of the chosen rule from the raw odds file: 1194 bets, return +0.0838 (backtest module: 1194 bets, +0.0838).