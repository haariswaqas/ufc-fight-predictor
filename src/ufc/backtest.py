"""Betting simulation on out-of-sample predictions."""
import numpy as np
import pandas as pd

from .market import sigmoid
from .metrics import event_bootstrap


def haircut(dec, h):
    """Worsen a decimal price: the profit part shrinks by the fraction h."""
    return 1 + (dec - 1) * (1 - h)


def ledger(o, z, threshold, cap=np.inf, h=0.0, price="open"):
    """One row per bet. Bets one unit on the side whose model edge at the given price is
    above `threshold` (and not above `cap`). Draws and no contests are voided at zero.

    o: prediction table (needs y, result, p_close, a_/b_ open and close decimals, date)
    z: model log-odds for fighter A, aligned with o."""
    p = sigmoid(np.asarray(z, float))
    da, db = haircut(o[f"a_{price}_decimal"].values, h), haircut(o[f"b_{price}_decimal"].values, h)
    ea, eb = p * da - 1, (1 - p) * db - 1
    on_a = ea >= eb
    edge = np.where(on_a, ea, eb)
    bet = (edge > threshold) & (edge <= cap) & ~np.isnan(edge)
    dec = np.where(on_a, da, db)
    y = o.y.values
    void = ~o.is_target.values
    won = np.where(on_a, y == 1, y == 0)
    profit = np.where(void, 0.0, np.where(won, dec - 1, -1.0))
    p_close_side = np.where(on_a, o.p_close.values, 1 - o.p_close.values)
    out = pd.DataFrame({
        "fight_id": o.fight_id.values, "date": o.date.values,
        "pick": np.where(on_a, o.a_name.values, o.b_name.values),
        "opponent": np.where(on_a, o.b_name.values, o.a_name.values),
        "model_prob": np.where(on_a, p, 1 - p), "price": dec, "edge": edge,
        "is_favourite": np.where(on_a, o.p_open.values, 1 - o.p_open.values) > 0.5,
        "void": void, "won": won & ~void, "profit": profit,
        # value of the price taken, judged by the closing line with the margin removed
        "clv": dec * p_close_side - 1,
        "move": np.abs(o.p_close.values - o.p_open.values),
    })
    return out[bet].reset_index(drop=True)


def summarise(L, reps=2000, seed=0):
    if len(L) == 0:
        return {"bets": 0}
    lo, hi = event_bootstrap(L.date.values, lambda i: L.profit.values[i].mean(), reps, seed)
    clo, chi = event_bootstrap(L.date.values, lambda i: L.clv.values[i].mean(), reps, seed)
    cum = L.sort_values("date").profit.cumsum().values
    drawdown = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
    return {"bets": len(L), "win_rate": float(L.won.mean()), "avg_price": float(L.price.mean()),
            "return": float(L.profit.mean()), "return_low": lo, "return_high": hi,
            "units_won": float(L.profit.sum()), "clv": float(L.clv.mean()), "clv_low": clo,
            "clv_high": chi, "beat_close_share": float((L.clv > 0).mean()), "max_drawdown": drawdown}
