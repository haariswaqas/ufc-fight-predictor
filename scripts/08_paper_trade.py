"""Paper trading with the frozen model. No money is staked; this only keeps a record.

    python scripts/08_paper_trade.py score     score paper/upcoming.csv and log it to paper/ledger.csv
    python scripts/08_paper_trade.py preview   same, but print only (nothing is logged)
    python scripts/08_paper_trade.py report    results so far, from the rows you have settled

The first logged price for a fight stands. Re-scoring the same fight does not overwrite it.
"""
import sys
from datetime import datetime

import _path  # noqa: F401
import numpy as np
import pandas as pd

from ufc import ROOT, load_config
from ufc.metrics import log_loss
from ufc.paper import LEDGER_COLS, UPCOMING_COLS, score

PAPER = ROOT / "paper"
UPCOMING, LEDGER = PAPER / "upcoming.csv", PAPER / "ledger.csv"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def load_ledger():
    if LEDGER.exists():
        return pd.read_csv(LEDGER)
    return pd.DataFrame(columns=LEDGER_COLS)


def cmd_score(log):
    up = pd.read_csv(UPCOMING)
    missing = [c for c in UPCOMING_COLS if c not in up.columns]
    if missing:
        sys.exit(f"paper/upcoming.csv is missing columns: {missing}")
    up = up[up.fighter_1.notna() & up.fighter_2.notna()].reset_index(drop=True)
    res, n_fit, last = score(up, load_config())
    show = res[["fighter_1", "fighter_2", "f1_decimal", "f2_decimal", "market_p_f1", "model_p_f1", "pick",
                "pick_price", "edge", "bet"]]
    print(f"Model fitted on {n_fit} fights with odds, through {last:%Y-%m-%d}.\n")
    print(show.to_string(index=False))
    for r in res.itertuples():
        if isinstance(r.notes, str) and r.notes:
            print(f"  note: {r.fighter_1} vs {r.fighter_2}: {r.notes}")
    bets = res[res.bet]
    print(f"\n{len(bets)} paper bet(s) out of {int(res.model_p_f1.notna().sum())} priced fights:")
    for r in bets.itertuples():
        print(f"  {r.pick} at {r.pick_price:.2f} (model {r.model_p_f1 if r.pick == r.fighter_1 else 1 - r.model_p_f1:.1%}, "
              f"edge {r.edge:+.1%})")
    if not log:
        print("\nPreview only: nothing logged.")
        return
    led = load_ledger()
    key = lambda d: d.date.astype(str) + "|" + d.fighter_1 + "|" + d.fighter_2
    new = res[res.model_p_f1.notna() & ~key(res).isin(set(key(led)))].copy()
    new["logged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    skipped = int(res.model_p_f1.notna().sum()) - len(new)
    pd.concat([led, new], ignore_index=True)[LEDGER_COLS].to_csv(LEDGER, index=False)
    print(f"\nLogged {len(new)} fight(s) to {LEDGER}" + (f"; {skipped} already logged and left unchanged." if skipped else "."))


def cmd_report():
    led = load_ledger()
    if led.empty:
        sys.exit("The ledger is empty. Run `score` first.")
    done = led[led.winner.notna()].copy()
    print(f"Logged fights: {len(led)}; settled: {len(done)}; paper bets: {int(led.bet.sum())}; "
          f"settled bets: {int(done.bet.sum())}.")
    if done.empty:
        print("Nothing settled yet. Fill in `winner` (and the closing prices) in paper/ledger.csv.")
        return
    w = done.winner.str.strip()
    decided = done[(w == done.fighter_1) | (w == done.fighter_2)]
    odd = done[~done.index.isin(decided.index) & ~w.str.lower().isin(["draw", "nc", "no contest", "void"])]
    if len(odd):
        print("Unrecognised `winner` values (use a fighter name exactly as logged, or draw / nc):")
        print(odd[["fighter_1", "fighter_2", "winner"]].to_string(index=False))
    if len(decided):
        y = (decided.winner.str.strip() == decided.fighter_1).astype(float)
        print(f"\nAll {len(decided)} decided fights: model log loss {log_loss(y, decided.model_p_f1):.4f}, "
              f"your prices {log_loss(y, decided.market_p_f1):.4f} (lower is better).")
    b = done[done.bet.astype(bool)].copy()
    if b.empty:
        return
    is_decided = b.index.isin(decided.index)
    won = b.winner.str.strip() == b.pick
    b["profit"] = np.where(~is_decided, 0.0, np.where(won, b.pick_price - 1, -1.0))
    print(f"\nPaper bets settled: {len(b)} ({int((~is_decided).sum())} void). Won {int((won & is_decided).sum())}.")
    print(f"Return: {b.profit.mean():+.1%} per unit staked; {b.profit.sum():+.2f} units.")
    c = b[b.close_f1_decimal.notna() & b.close_f2_decimal.notna()]
    if len(c):
        p1 = (1 / c.close_f1_decimal) / (1 / c.close_f1_decimal + 1 / c.close_f2_decimal)
        p_pick = np.where(c.pick == c.fighter_1, p1, 1 - p1)
        clv = c.pick_price * p_pick - 1
        print(f"Closing-line value on {len(c)} bets: {clv.mean():+.1%}; {int((clv > 0).sum())} beat the close.")
    se = b.profit.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else np.nan
    print(f"Rough 95% range on the return: {b.profit.mean() - 1.96 * se:+.1%} to {b.profit.mean() + 1.96 * se:+.1%}. "
          "Treat anything under about 200 settled bets as noise.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "score":
        cmd_score(log=True)
    elif cmd == "preview":
        cmd_score(log=False)
    elif cmd == "report":
        cmd_report()
    else:
        sys.exit(__doc__)
