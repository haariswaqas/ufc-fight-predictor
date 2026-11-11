# Method and round model, final test: 2025-01-01 to 2026-10-10

Models fitted on fights before 2025-01-01; scored on 2025-01-01 to 2026-10-10. Settings were frozen in `finish_frozen.json` before this run. All numbers in the tables are unchanged from the run; the explanations and examples were added afterwards.

## The short version

- The model predicts **how** a fight ends (knockout, submission, decision) and **in which round**, for each fighter.
- It was tested once on 931 to 950 fights it had never seen.
- It was compared with **base rates**: the plain historical averages, which know nothing about the two fighters.
- The model was clearly better than base rates on all three checks set in advance. **Result: PASS, 3 of 3.**
- The improvement is real but modest. The model shifts probabilities in the right direction; it does not "call" fights. An exact result (fighter, method and round) is usually a long shot even when the model reads the fight well.

## How to read this report

### What the model produces

For every fight the model gives a probability to each possible result. In a three-round fight there are 14 possible results: each fighter can win by knockout in round 1, 2 or 3, by submission in round 1, 2 or 3, or by decision. In a five-round fight there are 22. The probabilities add up to 100%.

"Knockout" here includes technical knockouts and doctor stoppages.

### Base rates: what the model is compared with

Base rates are the "know nothing about the fighters" forecast. They are counted from earlier fights with the same length (3 or 5 rounds), the same sex, and the same weight group (205 lb and above, or below).

Example: for any men's five-round fight below 205 lb, base rates say knockout 41.1%, submission 14.3%, decision 44.6%. They say exactly this for Yan vs Dvalishvili, for Gaethje vs Topuria and for every other such fight. The model's job is to do better than that by using what is known about the two fighters.

### Log loss: the main score

Log loss measures how good a set of probabilities is. For each fight, look at the probability the forecast gave to **what actually happened**, and take the negative natural logarithm of it. Then average over all fights. **Lower is better.**

| Probability given to what happened | Log loss for that fight |
|---|---|
| 90% | 0.11 |
| 64% | 0.44 |
| 45% | 0.80 |
| 20% | 1.61 |
| 5% | 3.00 |
| 1% | 4.61 |

Two things to notice:

- Giving a high probability to what happened costs little. Giving a low probability costs a lot.
- The penalty grows fast as the probability approaches zero. A forecast that says "1%" to something that then happens is punished far more than one that says "20%". This is why log loss rewards honest probabilities and punishes overconfidence.

A real example from this test, Petr Yan beating Merab Dvalishvili by decision (6 December 2025). Given that Yan won:

- The model said a Yan win would be a decision 64.3% of the time. Log loss: 0.44.
- Base rates said 44.6%. Log loss: 0.81.

The model scored better on this fight by 0.37. It knew that both fighters' fights tend to go long; base rates did not.

**Turning an average log loss back into something readable.** An average log loss of L means the forecast typically gave a probability of about e^(-L) to what happened. For example, 1.47 corresponds to about 23%, and 1.56 to about 21%. This conversion is used in the sections below.

### The other terms in the tables

| Term | Meaning |
|---|---|
| `log_loss` | Average log loss over the fights in that table. Lower is better. |
| `vs_base_rates` | The model's log loss minus the base rates' log loss. Negative means the model is better. |
| `low`, `high` | The 95% interval around that difference. See below. |
| `avg_probability_given_to_what_happened` | The plain average of the probabilities given to the actual results. Higher is better. |
| `top_pick_correct` | How often the single most likely result according to the forecast was the actual result. |
| `actual_in_top_3` | How often the actual result was among the forecast's three most likely results. |
| `predicted`, `actual` | In calibration tables: the average probability the model gave, and how often the thing really happened. |
| `band` | The range of predicted probabilities that a group of fights falls in. |
| AUC | Ranking skill. Explained in section 2. |
| Logistic, Boosting | The two models. The primary model is the average of their probabilities. |

### The 95% interval

The test has a limited number of fights, so part of any difference is luck. To measure how much, the event nights in the test were redrawn at random 2,000 times (some nights counted twice, some left out), and the difference was recomputed each time. The interval is the range that holds 95% of those results.

- If the whole interval is below zero, the model is better than base rates beyond reasonable doubt. That is the pass condition.
- If the interval crosses zero, the difference could be luck.

Example: -0.0955 with an interval of -0.1168 to -0.0744 means the model's advantage is somewhere between 0.074 and 0.117, and in any case not zero.

## 1. How and when the fight ends, given who won

This section sets aside the question of who wins. It asks: **knowing who won, how good was the forecast of the method and round?** That isolates the new model from the winner model.

950 decided fights (78 event nights).

| model | log_loss | vs_base_rates | low | high | avg_probability_given_to_what_happened |
|---|---|---|---|---|---|
| Base rates | 1.5649 | 0.0000 | 0.0000 | 0.0000 | 0.2988 |
| Logistic | 1.4767 | -0.0881 | -0.1105 | -0.0647 | 0.3195 |
| Boosting | 1.4728 | -0.0920 | -0.1141 | -0.0712 | 0.3246 |
| Average of the two (primary) | 1.4694 | -0.0955 | -0.1168 | -0.0744 | 0.3220 |

Validation (2019-2024) for comparison: base rates 1.5472, primary 1.4609.

**What the numbers mean**

- Base rates scored 1.5649, which corresponds to typically giving about 21% to the method and round that happened.
- The model scored 1.4694, about 23%.
- For scale: a forecast that spread its probability evenly over the 7 possible endings of a three-round fight would score 1.95 (about 14%).
- The improvement of 0.0955 is roughly a 10% relative increase in the probability placed on what actually happened. It is not a jump from guessing to knowing.
- The advantage on the test (-0.0955) is about the same as on validation (1.4609 - 1.5472 = -0.0863). The model did not get worse on new fights, which is the main thing a test is for.
- Averaging the two models (1.4694) beat each one alone (1.4767 and 1.4728), which is why the average is the primary model.

**Examples from the test, given the winner**

| Fight and result | Model | Base rates | Better |
|---|---|---|---|
| Yan beat Dvalishvili by decision | 64.3% | 44.6% | Model |
| Ankalaev beat Pereira by decision (March 2025) | 42.1% | 21.6% | Model |
| Morales beat Burns by knockout in round 1 | 17.3% | 14.9% | Model |
| Pereira beat Ankalaev by knockout in round 1 (October 2025) | 14.2% | 27.5% | Base rates |
| Chimaev beat Du Plessis by decision | 34.5% | 44.6% | Base rates |

The model is not better on every fight. It is better on average, and the interval shows that the average is not luck.

## 2. Combined with the winner model

From here on the forecasts are complete: the winner model supplies each fighter's chance of winning, and the method and round model splits that chance by method and round.

931 decided fights with usable opening odds. The chance of each fighter winning comes from the frozen winner model.

### How the fight ends: knockout, submission or decision

Here the round and the winner are ignored. The only question is which of the three methods ended the fight.

| model | log_loss | top_pick_correct |
|---|---|---|
| Base rates | 1.0191 | 0.4887 |
| Average of the two (primary) | 0.9584 | 0.5392 |

Difference: -0.0607 (95% interval -0.0807 to -0.0413). Actual share: knockout 36.4%, submission 17.5%, decision 46.1%.

**What the numbers mean**

- `top_pick_correct` of 0.5392 means the model's most likely method was the actual method in 53.9% of fights. Base rates managed 48.9%.
- Always saying "decision" would have been right 46.1% of the time, so 53.9% is a real but small gain.
- Log loss 0.9584 corresponds to typically giving about 38% to the method that happened, against about 36% for base rates. Three equal guesses of 33% each would score 1.10.
- This period had unusually many knockouts (36.4%). Base rates, built on earlier years, expected fewer, which is one reason they did worse here than in validation.

Goes the distance, ranking skill (AUC, 0.5 is a coin flip): model 0.657, base rates 0.575.

**What AUC means.** Take one fight that went to a decision and one that ended early, both picked at random. AUC is how often the forecast gave the higher decision probability to the one that really went the distance. 0.5 is a coin flip and 1.0 is perfect. The model gets this right 65.7% of the time. That is useful ranking skill, far from certainty.

### Calibration: predicted against actual, in five bands

Calibration asks whether the probabilities can be taken at face value: when the model says 40%, does it happen about 40% of the time?

To check, the 931 fights were sorted by the model's probability for a method and cut into five equal groups. `band` is the range of probabilities in the group, `predicted` is the group's average probability, and `actual` is how often that method really ended those fights.

| method | band | fights | predicted | actual |
|---|---|---|---|---|
| Knockout | (0.111, 0.237] | 187 | 0.188 | 0.171 |
| Knockout | (0.237, 0.305] | 186 | 0.274 | 0.237 |
| Knockout | (0.305, 0.366] | 186 | 0.336 | 0.344 |
| Knockout | (0.366, 0.45] | 186 | 0.406 | 0.457 |
| Knockout | (0.45, 0.822] | 186 | 0.538 | 0.613 |
| Submission | (0.0422, 0.0997] | 187 | 0.081 | 0.048 |
| Submission | (0.0997, 0.133] | 186 | 0.116 | 0.161 |
| Submission | (0.133, 0.166] | 186 | 0.150 | 0.118 |
| Submission | (0.166, 0.222] | 186 | 0.190 | 0.226 |
| Submission | (0.222, 0.664] | 186 | 0.300 | 0.323 |
| Decision | (0.082, 0.371] | 187 | 0.304 | 0.246 |
| Decision | (0.371, 0.449] | 186 | 0.412 | 0.387 |
| Decision | (0.449, 0.522] | 186 | 0.485 | 0.484 |
| Decision | (0.522, 0.599] | 186 | 0.558 | 0.543 |
| Decision | (0.599, 0.8] | 186 | 0.664 | 0.645 |

**How to read a row.** The first row: in the 187 fights where the model gave a knockout between 11.1% and 23.7% (18.8% on average), 17.1% ended by knockout. Predicted and actual are close, so the model was honest there.

**What the table shows**

- `actual` rises steadily with `predicted` for all three methods. The model separates knockout-prone fights from decision-prone ones.
- In the fights it rated most knockout-prone it said 53.8% and the result was 61.3%. It was too cautious about knockouts at the top, in a period with many knockouts.
- In the fights it rated least likely to go the distance it said 30.4% and the result was 24.6%. Same story from the other side.
- With about 186 fights in a group, a gap of 3 to 4 percentage points is normal noise. The 7-point knockout gap at the top is larger than noise.

**Practical reading:** when the model shows a knockout chance above about 45%, the true chance in this period was somewhat higher than shown.

## 3. Exact outcome: who wins, how, and in which round

This is the hardest question: the exact result, for example "Topuria by knockout in round 2". There are 14 possible results in a three-round fight and 22 in a five-round fight.

| model | log_loss | top_pick_correct | actual_in_top_3 | avg_probability_given_to_what_happened |
|---|---|---|---|---|
| Base rates | 2.1625 | 0.3040 | 0.6144 | 0.1769 |
| Average of the two (primary) | 2.0724 | 0.3201 | 0.6380 | 0.1892 |

Difference: -0.0900 (95% interval -0.1110 to -0.0687). Both rows use the same winner probabilities.

**What the numbers mean**

- Both rows use the same winner probabilities, so the difference between them is purely the method and round model.
- Log loss 2.0724 corresponds to typically giving about 12.6% to the exact result that happened, against about 11.5% for base rates. Spreading evenly over 14 results would give 7.1% each (log loss 2.64).
- `top_pick_correct` of 0.3201: the model's single most likely exact result happened in 32.0% of fights. That top pick is almost always "the favourite by decision", so this number mostly reflects how often favourites win decisions.
- `actual_in_top_3` of 0.6380: in 63.8% of fights the actual result was among the model's three most likely results.

**Examples from the test**

| Fight and result | Winner's chance | Method and round, given the winner | Exact result: model | Exact result: base rates |
|---|---|---|---|---|
| Yan beat Dvalishvili by decision | 22.8% | 64.3% | 14.7% | 10.2% |
| Morales beat Burns by knockout in round 1 | 86.3% | 17.3% | 14.9% | 12.8% |
| Gaethje beat Topuria by knockout in round 4 | 11.0% | 6.3% | 0.7% | 0.4% |

The exact-result probability is the winner's chance multiplied by the method-and-round chance given that winner: 22.8% x 64.3% = 14.7%.

The Gaethje row shows why exact results are hard. The model gave Gaethje an 11% chance of winning at all, and a round-4 knockout was a small slice of that. A result at 0.7% costs a log loss of 4.97 on that one fight. Upsets like this are part of the sport, and no model built on past records will price them high.

Among the 502 fights that ended in a finish, the model's most likely finish (fighter, method and round) was exactly right 30.7% of the time, against 28.1% for base rates.

This sets decisions aside. In fights that ended early, the model's best guess at the exact finish was right about 3 times in 10. That guess is usually "the favourite by knockout in round 1", the most common single finish.

### Predicted and actual share of fights ending in each round

This checks timing. `predicted_finish` is the average probability the model gave to a finish (knockout or submission, by either fighter) in that round. `actual_finish` is the share of fights that really ended in that round. The "decision" row is the share that went the full distance.

| format | round | fights | predicted_finish | actual_finish |
|---|---|---|---|---|
| 3 rounds | 1 | 843 | 0.254 | 0.305 |
| 3 rounds | 2 | 843 | 0.170 | 0.157 |
| 3 rounds | 3 | 843 | 0.089 | 0.075 |
| 3 rounds | decision | 843 | 0.487 | 0.464 |
| 5 rounds | 1 | 88 | 0.159 | 0.170 |
| 5 rounds | 2 | 88 | 0.151 | 0.182 |
| 5 rounds | 3 | 88 | 0.099 | 0.114 |
| 5 rounds | 4 | 88 | 0.069 | 0.057 |
| 5 rounds | 5 | 88 | 0.060 | 0.045 |
| 5 rounds | decision | 88 | 0.462 | 0.432 |

**What the table shows**

- **Three-round fights:** 30.5% ended in round 1, against 25.4% predicted. This is the model's clearest miss. Round-1 finishes were more common in 2025-26 than in the years it learned from. Rounds 2 and 3 are close.
- **Five-round fights:** predictions and results are close in every round, and the shape is right: fewer round-1 finishes than in three-round fights (17.0% against 30.5%). This is the pacing effect in championship and main-event fights.
- There are only 88 five-round fights, so each round's figure can move by 4 points or so through luck alone.

**Practical reading:** for three-round fights, the true chance of a round-1 finish in this period was a few points higher than the model showed.

## 4. Against the bookmakers' method-of-victory odds (not part of the pass rule)

Here the fight has six possible results: each fighter by knockout, by submission or by decision. The round is ignored. The model is compared with probabilities worked out from method-of-victory odds in one of the raw data files.

| source | log_loss | top_pick_correct |
|---|---|---|
| Base rates with our winner model | 1.5888 | 0.3475 |
| Our model | 1.5075 | 0.4051 |
| Method odds | 1.5664 | 0.3780 |

590 fights with method odds (the source stops in March 2026); six outcomes. Our model minus the odds: -0.0588 (95% interval -0.1039 to -0.0167); positive means the odds are better.

On its face this says the model beat the bookmakers. **Do not read it that way.** See section 6: the odds in this window are not genuine bookmaker prices.

## 5. Pass rule

The three conditions were written into `finish_frozen.json` before the test was run, so the result could not influence what counted as success. Each one requires the whole 95% interval to be below zero.

1. Method and round given the winner, against base rates, interval below zero: **PASS** (-0.0955, -0.1168 to -0.0744)
2. Knockout / submission / decision with the winner model, against base rates, interval below zero: **PASS** (-0.0607, -0.0807 to -0.0413)
3. Exact outcome with the winner model, against base rates, interval below zero: **PASS** (-0.0900, -0.1110 to -0.0687)

**Overall: PASS** (3 of 3).

**What passing means and does not mean**

- It means the model knows something real about how and when fights end, beyond historical averages, and that this held on fights it had never seen.
- It does not mean the model beats bookmakers' method or round prices. Bookmakers also know more than base rates.

## 6. Caveat on section 4, added after the run

The method odds in the source file change character in 2025. Through 2024 the six prices carry a bookmaker margin of about 22%; in 2025 the margin is about 0% and in 2026 it is negative (the implied probabilities sum to 0.94). Their predicted share of decisions also jumps from about 43% to about 54%. These are not the same kind of prices as before, so the model "beating" them in this window is not evidence that it beats real method markets. On 2019-2024, where the prices look like real bookmaker lines, the odds were better than the model by 0.026. The pass rule does not depend on this section.

**What "margin" means here.** A bookmaker's prices, turned into probabilities, add up to more than 100%; the excess is the bookmaker's cut. Six real method prices add up to about 122%. Prices that add up to 100% or less cannot be real quotes from one bookmaker, because anyone could bet every outcome and not lose. That is why the 2025-26 figures are not trusted.

## 7. Using these numbers in practice

- **Treat the output as probabilities, not predictions.** "Makhachev by decision 34%" means it is the most likely single result and still fails to happen two times in three.
- **Use the fair price.** A probability of 5% corresponds to a fair decimal price of 1 / 0.05 = 20.0. The app shows this for any combination of winner, method and round. A bookmaker's price below the fair price is poor value according to the model.
- **Do not bet method or round markets on this model alone.** On the years with genuine method odds, the bookmakers were slightly better, and their margin on these markets is large (about 22% across six outcomes, far more than on the winner market).
- **Know the two lean directions seen in this test:** round-1 finishes in three-round fights and knockouts in the most knockout-prone fights were both more common than the model said.
