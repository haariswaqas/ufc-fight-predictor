"""Matchup rows: one row per fight, fighter A minus fighter B, in a neutral order."""
import hashlib

import numpy as np
import pandas as pd

from .history import fighter_bouts, prefight

RATES = ["slpm", "sapm", "sig_acc", "sig_def", "kd_p15", "kda_p15", "td_p15", "tda_p15", "td_acc",
         "td_def", "sub_p15", "ctrl_share", "ctrla_share"]
# per-fighter values that become A-minus-B differences
BASE = (["debut", "n_fights", "log_n_fights", "ufc_minutes", "win_rate", "recent_win_rate", "streak",
         "finish_rate", "ko_win_rate", "sub_win_rate", "ko_loss_rate", "sub_loss_rate", "ko_losses",
         "days_since", "fights_2y", "elo", "avg_opp_elo", "age", "height_cm", "reach_cm", "southpaw"]
        + [f"{p}_{r}" for p in ("c", "r") for r in RATES])
# attributes that come from a present-day profile snapshot: never flag their absence
STATIC = ["age", "height_cm", "reach_cm", "southpaw"]
INTERACTIONS = ["d_reach_x_heavy", "d_elo_x_five_round", "d_age_x_heavy"]
FEATURES = ["d_" + c for c in BASE] + INTERACTIONS
MARKET = ["p_open", "p_close", "a_open_decimal", "b_open_decimal", "a_close_decimal",
          "b_close_decimal"]


def swap_flag(fight_id, seed):
    """Deterministic per-fight coin flip, independent of every other row."""
    return int(hashlib.md5(f"{seed}:{fight_id}".encode()).hexdigest(), 16) % 2 == 1


def split_label(date, cfg):
    s = cfg["splits"]
    d = pd.Timestamp(date)
    if d > pd.Timestamp(s["test_end"]):
        return "after_test"
    if d >= pd.Timestamp(s["test_start"]):
        return "test"
    if d >= pd.Timestamp(s["val_start"]):
        return "val"
    return "train" if d >= pd.Timestamp(s["train_start"]) else "history"


def matchups(clean, pre, cfg, flip=False):
    """Build the feature table. `flip=True` reverses every orientation (used by tests)."""
    p1 = pre[pre.side == 1].set_index("fight_id")
    p2 = pre[pre.side == 2].set_index("fight_id")
    c = clean.set_index("fight_id")
    p1, p2 = p1.loc[c.index], p2.loc[c.index]
    swap = np.array([swap_flag(i, cfg["seed"]) for i in c.index]) ^ flip
    sgn = np.where(swap, -1.0, 1.0)

    out = pd.DataFrame(index=c.index)
    out["date"] = c.date
    out["split"] = [split_label(d, cfg) for d in c.date]
    out["a_name"] = np.where(swap, c.fighter_2, c.fighter_1)
    out["b_name"] = np.where(swap, c.fighter_1, c.fighter_2)
    out["a_key"] = np.where(swap, c.f2_key, c.f1_key)
    out["b_key"] = np.where(swap, c.f1_key, c.f2_key)
    out["swapped"] = swap
    out["result"] = c.result
    out["is_target"] = c.is_target
    out["y"] = np.where(swap, 1 - c.f1_won, c.f1_won)          # 1 when fighter A won
    for col in ("lbs", "female", "heavy", "five_round", "ufc_title"):
        out[col] = c[col]

    for col in BASE:
        diff = (p1[col].values - p2[col].values) * sgn
        # a value missing on either side gives a difference of exactly zero
        out["d_" + col] = np.nan_to_num(diff, nan=0.0)
    out["d_reach_x_heavy"] = out.d_reach_cm * c.heavy.values
    out["d_age_x_heavy"] = out.d_age * c.heavy.values
    out["d_elo_x_five_round"] = out.d_elo * c.five_round.values
    # Elo's own win probability for fighter A (a baseline, not a fitted model)
    out["p_elo"] = 1 / (1 + 10 ** (-(p1.elo.values - p2.elo.values) * sgn / 400))

    # ---- market columns, kept apart from the fight features
    out["has_market"] = c.has_market
    out["odds_conflict"] = c.odds_conflict
    out["open_suspect"] = c.open_suspect
    out["market_ok"] = c.market_ok
    out["p_open"] = np.where(swap, 1 - c.f1_p_open, c.f1_p_open)
    out["p_close"] = np.where(swap, 1 - c.f1_p_close, c.f1_p_close)
    for kind in ("open", "close"):
        f1, f2 = c[f"f1_{kind}_decimal"], c[f"f2_{kind}_decimal"]
        out[f"a_{kind}_decimal"] = np.where(swap, f2, f1)
        out[f"b_{kind}_decimal"] = np.where(swap, f1, f2)
    return out.reset_index()


def build_all(clean, cfg, flip=False):
    """clean fights -> (fighter bouts, pre-fight snapshots, matchup features)."""
    bouts = fighter_bouts(clean)
    pre = prefight(bouts, cfg)
    return bouts, pre, matchups(clean, pre, cfg, flip=flip)
