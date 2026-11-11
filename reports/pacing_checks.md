# Pacing and fade checks

Descriptive checks on fights before 2025. No model was fitted. `n` is the number of fights or fighter appearances behind each figure.

## A. When fights end: 3-round against 5-round fights (2010-2024)

| scheduled | round | fights_reaching_it | ko_in_this_round | sub_in_this_round |
|---|---|---|---|---|
| 3 rounds | 1 | 6091 | 0.160 | 0.088 |
| 3 rounds | 2 | 4538 | 0.126 | 0.084 |
| 3 rounds | 3 | 3562 | 0.079 | 0.055 |
| 5 rounds | 1 | 618 | 0.175 | 0.047 |
| 5 rounds | 2 | 479 | 0.154 | 0.048 |
| 5 rounds | 3 | 380 | 0.108 | 0.055 |
| 5 rounds | 4 | 317 | 0.076 | 0.044 |
| 5 rounds | 5 | 278 | 0.079 | 0.032 |

| sched_rounds | fights | ko | submission | decision | ko_in_rounds_1_2 |
|---|---|---|---|---|---|
| 3 | 6091 | 0.300 | 0.183 | 0.495 | 0.254 |
| 5 | 618 | 0.435 | 0.155 | 0.388 | 0.294 |

**Same fighters in both formats.** 149 fighters have at least three 3-round and three 5-round fights. Averaging each fighter's own rates: a knockout in round 1 in 20.0% of their 3-round fights against 16.8% of their 5-round fights; a knockout in rounds 1-2 in 31.7% against 29.4%.

## B. Pace: do fighters start slower in 5-round fights? (2013-2024)

Significant strikes attempted per minute in round 1, for the 105 fighters with at least three completed first rounds in each format: 7.70 in 3-round fights and 7.44 in 5-round fights. Average change per fighter -0.26 (standard error 0.24); 55% of them start slower in 5-round fights.

| scheduled | fighter_appearances | round_1 | round_2 | round_3 | round_4 | round_5 |
|---|---|---|---|---|---|---|
| 3 rounds, went the distance | 5296 | 7.66 | 8.25 | 8.60 |  |  |
| 5 rounds, went the distance | 442 | 7.32 | 8.67 | 8.61 | 8.95 | 9.19 |

Strikes attempted per minute by round, in fights that went the full distance.

## C. Knockout artists: what happens when the early knockout does not come

A knockout artist here is a fighter with at least 5 UFC fights whose knockout wins per fight, before the fight in question, are in the top quarter (0.28 or more after adjustment for sample size). 1541 appearances qualify.

| group | situation | n | won | won_by_ko | lost_by_ko | pre_fight_odds_implied | won_where_odds_known |
|---|---|---|---|---|---|---|---|
| Knockout artists | All fights | 1541 | 0.520 | 0.239 | 0.207 | 0.517 | 0.519 |
| Knockout artists | Fight reached round 3 | 848 | 0.537 | 0.093 | 0.073 | 0.516 | 0.536 |
| Knockout artists | 5-round fight reached round 4 | 239 | 0.506 | 0.109 | 0.079 | 0.502 | 0.505 |
| Other experienced fighters | All fights | 4621 | 0.499 | 0.139 | 0.155 | 0.494 | 0.497 |
| Other experienced fighters | Fight reached round 3 | 2824 | 0.504 | 0.044 | 0.044 | 0.496 | 0.503 |
| Other experienced fighters | 5-round fight reached round 4 | 302 | 0.520 | 0.053 | 0.066 | 0.509 | 0.524 |

`pre_fight_odds_implied` is the closing-odds win probability before the fight, for the fights that have odds; compare it with `won_where_odds_known` on the same fights.

Their knockout wins by round: round 1 50%, round 2 28%, round 3 14%, round 4 4%, round 5 3%.

| group | n | round_1_attempts_per_min | round_3_attempts_per_min | change | round_1_absorbed_per_min | round_3_absorbed_per_min |
|---|---|---|---|---|---|---|
| Knockout artists | 615 | 7.61 | 9.27 | 1.66 | 3.46 | 4.05 |
| Other experienced fighters | 2223 | 7.80 | 8.99 | 1.19 | 3.38 | 3.95 |

Output in round 1 and round 3, for fights where both rounds were completed (2013-2024).

## D. Is pace, and fading, a stable trait of a fighter?

For 2311 appearances with at least 3 earlier fights that completed three rounds: the correlation between a fighter's earlier average round-1 pace and their round-1 pace this time is 0.35. For the change from round 1 to round 3 (the fade) it is 0.16. Near 0 means no carry-over; near 1 means a fixed trait.

## E. Do fighters who absorb more strikes get knocked out more?

| measure_before_the_fight | lowest quarter | second | third | highest quarter |
|---|---|---|---|---|
| Strikes absorbed per minute, career | 0.141 | 0.171 | 0.175 | 0.169 |
| Strikes absorbed per minute, recent | 0.138 | 0.160 | 0.180 | 0.178 |
| Knockdowns suffered per 15 min, career | 0.102 | 0.151 | 0.179 | 0.224 |
| Knockout losses per fight | 0.132 | 0.138 | 0.162 | 0.224 |
| Strikes landed per minute, career | 0.176 | 0.163 | 0.163 | 0.154 |

Share of fights lost by knockout, by quarter of each pre-fight measure (8290 appearances with at least 3 earlier fights; overall 0.164).

Most significant strikes landed in the UFC before 2025: Max Holloway 3457, Sean Strickland 2069, Angela Hill 2036, King Green 1869, Rafael Dos Anjos 1822.
Most absorbed: Max Holloway 2298, Angela Hill 1839, Donald Cerrone 1825, Marlon Vera 1738, Edson Barboza 1700.
