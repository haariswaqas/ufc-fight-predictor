# Method and round model: validation

Every number is out-of-sample: each year from 2019 to 2024 was predicted by a model trained only on earlier fights. No fight from 2025 on was read. Log loss is the score for probabilities; lower is better.

## 1. How and when the fight ends, given who won

| model | log_loss | vs_base_rates | low | high | avg_probability_given_to_what_happened |
|---|---|---|---|---|---|
| Base rates | 1.5472 | 0.0000 | 0.0000 | 0.0000 | 0.3152 |
| Logistic | 1.4663 | -0.0808 | -0.1143 | -0.0545 | 0.3319 |
| Boosting | 1.4654 | -0.0817 | -0.1171 | -0.0543 | 0.3307 |
| Average of the two | 1.4609 | -0.0862 | -0.1204 | -0.0597 | 0.3313 |

2961 fights. The outcome is the exact method and round (for example knockout in round 2), or a decision. Base rates are the round-by-round finish rates of earlier fights with the same format (3 or 5 rounds), sex and weight group. An interval entirely below zero is a clear improvement.

**Chosen model: Average of the two.** Settings: logistic `{"kind": "logit", "start": 2013, "c": 0.003, "year": true}`; boosting `{"kind": "gbm", "start": 2010, "rounds": 100, "year": true, "params": {"num_leaves": 8, "max_depth": -1, "learning_rate": 0.04, "min_child_samples": 150, "feature_fraction": 0.6, "lambda_l2": 5.0, "min_split_gain": 0.01}}`.

## 2. Combined with the winner model

2926 decided fights from 2019-2024 that have opening odds. The chance of each fighter winning comes from the frozen winner model's own out-of-sample predictions.

### How the fight ends: knockout, submission or decision

| model | log_loss | vs_base_rates | low | high | top_pick_correct |
|---|---|---|---|---|---|
| Base rates | 0.9996 | 0.0000 | 0.0000 | 0.0000 | 0.5147 |
| Average of the two | 0.9639 | -0.0357 | -0.0475 | -0.0237 | 0.5253 |

Share of these fights that actually ended by knockout 31.7%, submission 17.5%, decision 50.7%. Always saying "decision" is right 50.7% of the time.

### Does the fight go the distance?

- Ranking skill (AUC, 0.5 is a coin flip): model 0.637, base rates 0.561.
- Brier score (lower is better): model 0.2357, base rates 0.2459.

| band | fights | predicted_decision | actual_decision |
|---|---|---|---|
| (0.067, 0.389] | 586 | 0.311 | 0.355 |
| (0.389, 0.47] | 585 | 0.433 | 0.432 |
| (0.47, 0.529] | 585 | 0.500 | 0.484 |
| (0.529, 0.597] | 585 | 0.561 | 0.590 |
| (0.597, 0.814] | 585 | 0.663 | 0.675 |

### When the model says knockout, and when it says submission

| method | band | fights | predicted | actual |
|---|---|---|---|---|
| Knockout | (0.107, 0.233] | 586 | 0.189 | 0.152 |
| Knockout | (0.233, 0.292] | 585 | 0.262 | 0.284 |
| Knockout | (0.292, 0.351] | 585 | 0.321 | 0.294 |
| Knockout | (0.351, 0.431] | 585 | 0.387 | 0.381 |
| Knockout | (0.431, 0.866] | 585 | 0.522 | 0.477 |
| Submission | (0.041999999999999996, 0.102] | 586 | 0.081 | 0.089 |
| Submission | (0.102, 0.136] | 585 | 0.119 | 0.106 |
| Submission | (0.136, 0.171] | 585 | 0.153 | 0.168 |
| Submission | (0.171, 0.225] | 585 | 0.197 | 0.202 |
| Submission | (0.225, 0.576] | 585 | 0.302 | 0.313 |

## 3. Exact outcome: who wins, how, and in which round

| model | log_loss | top_pick_correct | actual_in_top_3 | avg_probability_given_to_what_happened |
|---|---|---|---|---|
| Base rates | 2.1675 | 0.3219 | 0.6217 | 0.1774 |
| Average of the two | 2.0814 | 0.3360 | 0.6292 | 0.1849 |

Difference in log loss: -0.0860 (95% interval -0.1193 to -0.0604). Both rows use the same winner probabilities, so the difference is the method-and-round model alone. The top pick is almost always "the favourite by decision", which is why exact-outcome hit rates are low for any model.

Among the 1442 fights that ended in a finish, the model's most likely finish (fighter, method and round) was exactly right 24.4% of the time, against 22.7% for base rates.

### Predicted and actual share of fights ending in each round

| format | round | fights | predicted_finish | actual_finish |
|---|---|---|---|---|
| 3 rounds | 1 | 2636 | 0.242 | 0.248 |
| 3 rounds | 2 | 2636 | 0.163 | 0.162 |
| 3 rounds | 3 | 2636 | 0.090 | 0.079 |
| 5 rounds | 1 | 290 | 0.209 | 0.169 |
| 5 rounds | 2 | 290 | 0.157 | 0.159 |
| 5 rounds | 3 | 290 | 0.106 | 0.093 |
| 5 rounds | 4 | 290 | 0.071 | 0.059 |
| 5 rounds | 5 | 290 | 0.060 | 0.052 |

## 4. Against the bookmakers' method-of-victory odds

| source | log_loss | top_pick_correct |
|---|---|---|
| Base rates with our winner model | 1.6231 | 0.3277 |
| Our model | 1.5652 | 0.3450 |
| Method odds | 1.5388 | 0.3705 |

2429 fights with method odds; six outcomes (each fighter by knockout, submission or decision). Our model minus the odds: +0.0264 (95% interval +0.0121 to +0.0403); positive means the odds are better.

Method only, given the winner: our model 0.9452, method odds 0.9289; difference +0.0163 (95% interval +0.0031 to +0.0298). The origin and timing of these odds are not documented, and the bookmaker margin (about 22% across the six outcomes) was removed by simple scaling, so treat this as a rough benchmark.

## 5. What the model responds to

| input | share_of_importance |
|---|---|
| w_c_sub_p15 | 0.085 |
| lbs | 0.071 |
| rnd | 0.067 |
| w_r_sub_p15 | 0.047 |
| w_c_kd_p15 | 0.038 |
| w_avg_fight_min | 0.038 |
| l_avg_fight_min | 0.034 |
| w_sub_w_p15 | 0.030 |
| p_w | 0.028 |
| rounds_left | 0.027 |
| l_early_fin_l_rate | 0.027 |
| w_c_ctrl_share | 0.027 |
| l_ko_l_p15 | 0.026 |
| l_sub_l_p15 | 0.026 |
| w_c_td_p15 | 0.025 |

`w_` is the assumed winner, `l_` the loser, `rnd` the round number, `p_w` the winner's chance from the opening odds. Importance is the boosting model's, fitted on all fights before 2025.

| grouped_by | group | fights | predicted_knockout | actual_knockout | predicted_ko_in_rounds_1_2 | actual_ko_in_rounds_1_2 |
|---|---|---|---|---|---|---|
| Winner's knockout wins per fight | lowest quarter | 741 | 0.243 | 0.216 | 0.193 | 0.177 |
| Winner's knockout wins per fight | second | 740 | 0.291 | 0.300 | 0.234 | 0.258 |
| Winner's knockout wins per fight | third | 740 | 0.346 | 0.324 | 0.275 | 0.262 |
| Winner's knockout wins per fight | highest quarter | 740 | 0.458 | 0.430 | 0.362 | 0.343 |
| Loser's knockout losses per fight | lowest quarter | 742 | 0.282 | 0.268 | 0.213 | 0.209 |
| Loser's knockout losses per fight | second | 739 | 0.304 | 0.281 | 0.238 | 0.223 |
| Loser's knockout losses per fight | third | 740 | 0.333 | 0.314 | 0.267 | 0.255 |
| Loser's knockout losses per fight | highest quarter | 740 | 0.418 | 0.407 | 0.347 | 0.353 |

### Three-round against five-round fights, and the shift over time

| format | years | fights | predicted_ko_round_1 | actual_ko_round_1 | predicted_decision | actual_decision |
|---|---|---|---|---|---|---|
| 3 rounds | 2019-21 | 1300 | 0.150 | 0.168 | 0.520 | 0.523 |
| 3 rounds | 2022-24 | 1366 | 0.169 | 0.158 | 0.490 | 0.502 |
| 5 rounds | 2019-21 | 142 | 0.186 | 0.127 | 0.371 | 0.451 |
| 5 rounds | 2022-24 | 153 | 0.126 | 0.098 | 0.425 | 0.477 |

Five-round fights changed around 2019: round-1 knockouts fell from about 25% of fights in 2013-18 to about 11% in 2019-24, and decisions rose from about 31% to about 46%. Three-round fights barely moved. A model trained on the earlier years over-predicts early finishes in five-round fights until it has seen enough of the new pattern, which is why the 2019-21 row is further off than 2022-24. The year is a model input so that it can follow this.

## 6. Example: Ilia Topuria vs Max Holloway, 26 October 2024

| outcome | probability |
|---|---|
| Ilia Topuria by decision | 0.329 |
| Max Holloway by decision | 0.187 |
| Ilia Topuria by knockout, round 2 | 0.076 |
| Ilia Topuria by knockout, round 1 | 0.067 |
| Ilia Topuria by knockout, round 3 | 0.051 |
| Ilia Topuria by knockout, round 4 | 0.036 |
| Max Holloway by knockout, round 1 | 0.033 |
| Ilia Topuria by knockout, round 5 | 0.033 |

Winner model: Topuria 67.2%. Chance of any knockout 38.2%, submission 10.2%, decision 51.6%. What happened: Ilia Topuria by knockout, round 3, which the model gave 5.1%.
