# Paper trading

A record of what the frozen model would have bet, at prices you could actually get.
No money is staked. The aim is to find out two things the backtest could not:
whether the edge is real, and whether the prices are reachable.

Run every command from the project folder.

## Before each card

1. Open `paper/upcoming.csv` and list the fights, one per row.

   | Column | What to enter |
   |---|---|
   | `date` | Fight date, `YYYY-MM-DD` |
   | `event` | Event name |
   | `fighter_1`, `fighter_2` | Names spelled as on ufcstats.com |
   | `bout_type` | For example `Lightweight Bout`, `Women's Flyweight Bout`, `UFC Heavyweight Title Bout` |
   | `rounds` | `3` or `5` |
   | `f1_decimal`, `f2_decimal` | Decimal odds you can get right now, for each fighter |
   | `bookmaker` | Where you saw those odds |

2. Check it without logging anything:

   ```
   .venv\Scripts\python scripts\08_paper_trade.py preview
   ```

   Read the notes. `NOT IN DATA` means the name did not match anyone; fix the spelling
   unless the fighter really is making a UFC debut.

3. Log it:

   ```
   .venv\Scripts\python scripts\08_paper_trade.py score
   ```

   Every priced fight is added to `paper/ledger.csv`, with `bet` set to `True` where the
   model's edge is above 5% and not above 20%.

**Enter the odds as early as you can, and enter them once.** The backtest edge existed only
at opening prices. The first logged price for a fight stands; scoring it again changes nothing.

## After each card

Open `paper/ledger.csv` and fill in three columns for each fight:

| Column | What to enter |
|---|---|
| `close_f1_decimal`, `close_f2_decimal` | The odds just before the fight started |
| `winner` | The winner's name exactly as logged, or `draw`, or `nc` |

Then:

```
.venv\Scripts\python scripts\08_paper_trade.py report
```

## Reading the report

- **Closing-line value** is the measure to watch first. It becomes meaningful after roughly
  100 bets. Positive means your prices were better than the closing prices.
- **Return** is noise until about 200 settled bets, which is around a year of cards.
- **Model log loss versus your prices** uses every logged fight, not only the bets.

## Keeping the model current

The model reads fighter histories from `Datasets/ufc_fights_1993_2026.csv`, which currently
ends on 2026-11-07. Fights after that date are not known to it until that file is updated
and the pipeline scripts (`build_enriched_dataset.py`, then `01` to `03`) are re-run.
The model settings themselves are frozen in `config_frozen.json` and should not be changed.
