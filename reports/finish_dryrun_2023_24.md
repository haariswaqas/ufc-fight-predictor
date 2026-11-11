# Dry run: 2023-24 standing in for the test (not the real test)

Models fitted on fights before 2023-01-01; scored on 2023-01-01 to 2024-12-31. Settings were frozen in `finish_frozen.json` before this run. Log loss is the score for probabilities; lower is better.

## 1. How and when the fight ends, given who won

1014 decided fights (85 event nights).

| model | log_loss | vs_base_rates | low | high | avg_probability_given_to_what_happened |
|---|---|---|---|---|---|
| Base rates | 1.5350 | 0.0000 | 0.0000 | 0.0000 | 0.3168 |
| Logistic | 1.4527 | -0.0823 | -0.1402 | -0.0412 | 0.3286 |
| Boosting | 1.4619 | -0.0732 | -0.1355 | -0.0265 | 0.3214 |
| Average of the two (primary) | 1.4517 | -0.0833 | -0.1433 | -0.0398 | 0.3250 |

Validation (2019-2024) for comparison: base rates 1.5472, primary 1.4609.

## 2. Combined with the winner model

1000 decided fights with usable opening odds. The chance of each fighter winning comes from the frozen winner model.

### How the fight ends: knockout, submission or decision

| model | log_loss | top_pick_correct |
|---|---|---|
| Base rates | 0.9945 | 0.5140 |
| Average of the two (primary) | 0.9624 | 0.5330 |

Difference: -0.0321 (95% interval -0.0542 to -0.0087). Actual share: knockout 29.8%, submission 18.3%, decision 51.9%.

Goes the distance, ranking skill (AUC, 0.5 is a coin flip): model 0.658, base rates 0.559.

### Calibration: predicted against actual, in five bands

| method | band | fights | predicted | actual |
|---|---|---|---|---|
| Knockout | (0.121, 0.255] | 200 | 0.208 | 0.145 |
| Knockout | (0.255, 0.318] | 200 | 0.285 | 0.195 |
| Knockout | (0.318, 0.371] | 200 | 0.343 | 0.310 |
| Knockout | (0.371, 0.444] | 200 | 0.405 | 0.360 |
| Knockout | (0.444, 0.782] | 200 | 0.526 | 0.480 |
| Submission | (0.046, 0.106] | 200 | 0.084 | 0.100 |
| Submission | (0.106, 0.14] | 200 | 0.123 | 0.130 |
| Submission | (0.14, 0.176] | 200 | 0.158 | 0.130 |
| Submission | (0.176, 0.237] | 200 | 0.202 | 0.220 |
| Submission | (0.237, 0.559] | 200 | 0.320 | 0.335 |
| Decision | (0.0706, 0.361] | 200 | 0.291 | 0.355 |
| Decision | (0.361, 0.444] | 200 | 0.405 | 0.430 |
| Decision | (0.444, 0.504] | 200 | 0.475 | 0.495 |
| Decision | (0.504, 0.568] | 200 | 0.535 | 0.580 |
| Decision | (0.568, 0.762] | 200 | 0.639 | 0.735 |

## 3. Exact outcome: who wins, how, and in which round

| model | log_loss | top_pick_correct | actual_in_top_3 | avg_probability_given_to_what_happened |
|---|---|---|---|---|
| Base rates | 2.1316 | 0.3510 | 0.6260 | 0.1840 |
| Average of the two (primary) | 2.0506 | 0.3670 | 0.6510 | 0.1861 |

Difference: -0.0810 (95% interval -0.1369 to -0.0395). Both rows use the same winner probabilities.

Among the 481 fights that ended in a finish, the model's most likely finish (fighter, method and round) was exactly right 24.3% of the time, against 22.2% for base rates.

### Predicted and actual share of fights ending in each round

| format | round | fights | predicted_finish | actual_finish |
|---|---|---|---|---|
| 3 rounds | 1 | 901 | 0.257 | 0.235 |
| 3 rounds | 2 | 901 | 0.174 | 0.164 |
| 3 rounds | 3 | 901 | 0.093 | 0.080 |
| 3 rounds | decision | 901 | 0.476 | 0.521 |
| 5 rounds | 1 | 99 | 0.191 | 0.162 |
| 5 rounds | 2 | 99 | 0.160 | 0.162 |
| 5 rounds | 3 | 99 | 0.107 | 0.081 |
| 5 rounds | 4 | 99 | 0.075 | 0.051 |
| 5 rounds | 5 | 99 | 0.062 | 0.040 |
| 5 rounds | decision | 99 | 0.406 | 0.505 |

## 4. Against the bookmakers' method-of-victory odds (not part of the pass rule)

| source | log_loss | top_pick_correct |
|---|---|---|
| Base rates with our winner model | 1.6032 | 0.3399 |
| Our model | 1.5516 | 0.3666 |
| Method odds | 1.5343 | 0.3876 |

712 fights with method odds (the source stops in March 2026); six outcomes. Our model minus the odds: +0.0173 (95% interval -0.0095 to +0.0441); positive means the odds are better.

## 5. Pass rule

1. Method and round given the winner, against base rates, interval below zero: **PASS** (-0.0833, -0.1433 to -0.0398)
2. Knockout / submission / decision with the winner model, against base rates, interval below zero: **PASS** (-0.0321, -0.0542 to -0.0087)
3. Exact outcome with the winner model, against base rates, interval below zero: **PASS** (-0.0810, -0.1369 to -0.0395)

**Overall: PASS** (3 of 3).
