"""Paper trading: score upcoming fights with the frozen recipe and keep a ledger.

The upcoming fights are appended to the clean fight table as rows without a result, so
they go through exactly the same history engine and feature code as every past fight.
"""
import difflib
import hashlib
import json

import numpy as np
import pandas as pd

from . import ROOT
from .clean import STAT_COLS, bout_lbs, ident, load_clean, norm
from .features import build_all
from .finish import X_COLS, X_YEAR, FinishGBM, FinishLogit, distribution, scenarios
from .market import fit_stretch, sigmoid
from .models import FEATURE_SETS, SymGBM, SymLogit, design, recency_weights

UPCOMING_COLS = ["date", "event", "fighter_1", "fighter_2", "bout_type", "rounds", "f1_decimal",
                 "f2_decimal", "bookmaker"]
LEDGER_COLS = ["logged_at", "date", "event", "fighter_1", "fighter_2", "bookmaker", "f1_decimal",
               "f2_decimal", "market_p_f1", "model_p_f1", "pick", "pick_price", "edge", "bet", "notes",
               # filled in by hand after the fight
               "close_f1_decimal", "close_f2_decimal", "winner"]


def _profiles(clean):
    cols = ["key", "name", "height_cm", "reach_cm", "stance", "dob", "lbs", "female"]
    parts = [clean[[f"f{s}_key", f"fighter_{s}", f"f{s}_height_cm", f"f{s}_reach_cm", f"f{s}_stance",
                    f"f{s}_dob", "lbs", "female"]].set_axis(cols, axis=1) for s in "12"]
    return pd.concat(parts, ignore_index=True)


def resolve(name, lbs, prof, tott):
    """(fighter key, profile dict, note). A name not in the data is treated as a UFC debutant."""
    hits = prof[prof.name.map(ident) == ident(name)]
    if len(hits):
        keys = hits.groupby("key").lbs.median()
        key = keys.index[0] if len(keys) == 1 else (keys - lbs).abs().idxmin()
        row = hits[hits.key == key].iloc[-1]
        return key, {c: row[c] for c in ("height_cm", "reach_cm", "stance", "dob")}, ""
    known = sorted(set(prof.name))
    near = difflib.get_close_matches(name, known, n=3, cutoff=0.8)
    note = "NOT IN DATA: treated as UFC debutant" + (f" (did you mean {', '.join(near)}?)" if near else "")
    t = tott[tott.n == norm(name)]
    profile = {"height_cm": np.nan, "reach_cm": np.nan, "stance": np.nan, "dob": np.nan}
    if len(t) == 1:
        profile = {c: t.iloc[0][c] for c in profile}
    return "n:" + ident(name), profile, note


def _tott():
    t = pd.read_csv(ROOT / "Datasets/raw/greco1899/ufc_fighter_tott.csv")
    t["n"] = t.FIGHTER.map(norm)
    h = t.HEIGHT.str.extract(r"(\d+)\D+(\d+)").astype(float)
    t["height_cm"] = ((h[0] * 12 + h[1]) * 2.54).round(1)
    t["reach_cm"] = (pd.to_numeric(t.REACH.str.extract(r"(\d+)")[0], errors="coerce") * 2.54).round(1)
    t["dob"] = pd.to_datetime(t.DOB, format="%b %d, %Y", errors="coerce").dt.strftime("%Y-%m-%d")
    t["stance"] = t.STANCE.where(t.STANCE.notna() & (t.STANCE != ""))
    return t


def upcoming_rows(up, clean):
    """Turn the upcoming card into rows shaped like the clean fight table."""
    prof, tott = _profiles(clean), _tott()
    women = set(prof.key[prof.female])
    rows, notes = [], []
    for r in up.itertuples(index=False):
        lbs = bout_lbs(r.bout_type)
        k1, pr1, n1 = resolve(r.fighter_1, lbs, prof, tott)
        k2, pr2, n2 = resolve(r.fighter_2, lbs, prof, tott)
        date = pd.Timestamp(r.date)
        fid = "up" + hashlib.md5(f"{date:%Y-%m-%d}|{r.fighter_1}|{r.fighter_2}".encode()).hexdigest()[:14]
        female = ("women" in str(r.bout_type).lower()) or (k1 in women and k2 in women)
        five = int(r.rounds) == 5
        priced = pd.notna(r.f1_decimal) and pd.notna(r.f2_decimal)
        row = {c: np.nan for c in clean.columns}
        row.update(fight_id=fid, date=date, event=r.event, fighter_1=r.fighter_1, fighter_2=r.fighter_2,
                   f1_key=k1, f2_key=k2, result="upcoming", is_target=False, method_group="OTHER",
                   sched_rounds=float(r.rounds), five_round=five, bout_type=r.bout_type, lbs=lbs,
                   female=female, ufc_title="title" in str(r.bout_type).lower(),
                   heavy=bool(lbs >= 205) and not female, has_market=priced, market_ok=priced,
                   odds_conflict=False, open_suspect=False,
                   f1_open_decimal=r.f1_decimal, f2_open_decimal=r.f2_decimal)
        if priced:
            a, b = 1 / r.f1_decimal, 1 / r.f2_decimal
            row["f1_p_open"] = a / (a + b)
        for s, pr in (("1", pr1), ("2", pr2)):
            for c, v in pr.items():
                row[f"f{s}_{c}"] = v
        rows.append(row)
        notes.append("; ".join(x for x in (f"{r.fighter_1}: {n1}" if n1 else "",
                                           f"{r.fighter_2}: {n2}" if n2 else "") if x))
    out = pd.DataFrame(rows, columns=clean.columns)
    for c in ("is_target", "five_round", "female", "ufc_title", "heavy", "has_market", "odds_conflict",
              "open_suspect", "market_ok"):
        out[c] = out[c].astype(bool)
    return out, notes


def _upcoming_build(up, clean, cfg):
    new, notes = upcoming_rows(up, clean)
    assert new.date.min() > clean.date.max(), "upcoming fights must be after the last fight in the data"
    assert (new.f1_key != new.f2_key).all()
    both = pd.concat([clean, new], ignore_index=True)
    both["date"] = pd.to_datetime(both.date)
    for s in "12":
        both[f"f{s}_dob"] = pd.to_datetime(both[f"f{s}_dob"])
    _, pre, f = build_all(both, cfg)
    f = design(f, pre)
    rows = f[f.fight_id.isin(new.fight_id)].set_index("fight_id").loc[new.fight_id].reset_index()
    new_clean = both[both.fight_id.isin(new.fight_id)].set_index("fight_id").loc[new.fight_id].reset_index()
    return f, rows, notes, new_clean, pre[pre.fight_id.isin(new.fight_id)]


def upcoming_features(up, clean, cfg):
    """Model-ready feature rows for the upcoming fights, plus resolution notes."""
    return _upcoming_build(up, clean, cfg)[:3]


# ---------------------------------------------------------------- method and round
FINISH_OUT = [f"{m}_{r}" for m in ("ko", "sub") for r in range(1, 6)] + ["dec"]
FINISH_COLS = [f"f{s}_{c}" for s in "12" for c in FINISH_OUT]


def load_finish_frozen():
    return json.load(open(ROOT / "finish_frozen.json", encoding="utf-8"))


def fit_finish(clean, pre, fz):
    """Fit the frozen method/round recipe on every decided 3- or 5-round fight."""
    sc = scenarios(clean, pre)
    tr = sc[sc.actual & sc.usable]
    cols = lambda spec: X_YEAR if spec["year"] else X_COLS
    lg, gb = fz["logit"], fz["gbm"]
    t = tr[tr.date >= str(gb["start"])]
    return {"logit": FinishLogit(lg["c"], cols=cols(lg)).fit(tr[tr.date >= str(lg["start"])]),
            "gbm": [FinishGBM(gb["params"], gb["rounds"], seed=sd, cols=cols(gb)).fit(t) for sd in fz["gbm_seeds"]],
            "n": len(tr), "last": tr.date.max()}


def fit_finish_history(cfg):
    clean = load_clean(cfg)
    pre = pd.read_csv(ROOT / cfg["paths"]["data"] / "fighter_prefight.csv", parse_dates=["date"])
    return fit_finish(clean, pre, load_finish_frozen())


def finish_distribution(sc, fit):
    """Outcome probabilities given each scenario's winner: mean of the logistic and boosting models."""
    g = sum(distribution(sc, m.hazards) for m in fit["gbm"]) / len(fit["gbm"])
    return (distribution(sc, fit["logit"].hazards) + g) / 2


def fit_frozen(f, frozen):
    """Fit the frozen B1 and B2 recipes on every decided fight with usable odds."""
    end = f.date[f.is_target & f.market_ok].max() + pd.Timedelta(days=1)
    models = {}
    for name in ("B1", "B2"):
        s = frozen["models"][name]
        tr = f[f.is_target & f.market_ok & (f.date >= str(s["start"]))]
        w = recency_weights(tr.date, end, s["half_life"])
        if name == "B1":
            models[name] = [SymLogit(FEATURE_SETS[s["feature_set"]], s["lam"], market=True).fit(tr, tr.y, w)]
        else:
            slope = fit_stretch(tr.p_open, tr.y)
            models[name] = [SymGBM(s["params"], s["rounds"], market=True, seed=sd)
                            .fit(tr, tr.y, w, market_slope=slope) for sd in frozen["gbm_seeds"]]
    return models, len(tr), end - pd.Timedelta(days=1)


def load_frozen():
    return json.load(open(ROOT / "config_frozen.json", encoding="utf-8"))


def fit_history(cfg):
    """Fit the frozen recipe on the historical fights only. The result can be reused for
    any upcoming card, because upcoming fights are never training rows."""
    clean = load_clean(cfg)
    _, pre, f = build_all(clean, cfg)
    return fit_frozen(design(f, pre), load_frozen())


def score(up, cfg, fitted=None):
    """Score an upcoming card. Returns one ledger-shaped row per fight.
    `fitted` is an optional result of fit_history(), to avoid refitting."""
    return score_full(up, cfg, fitted)[:3]


def score_full(up, cfg, fitted=None, finish=None):
    """score() plus method and round probabilities.
    `finish` is a result of fit_finish_history(); without it the fourth value is None.
    The fourth value has one row per fight and columns f1_ko_1 .. f2_dec: the chance that
    fighter 1 (or 2) wins by that method in that round. Each priced row sums to one."""
    frozen = load_frozen()
    rule = frozen["betting_rule"]
    clean = load_clean(cfg)
    f, rows, notes, new_clean, new_pre = _upcoming_build(up, clean, cfg)
    models, n_fit, last = fitted if fitted is not None else fit_frozen(f, frozen)
    out, p1s = [], []
    for i, r in rows.iterrows():
        u = up.iloc[i]
        rec = {"date": pd.Timestamp(u.date).strftime("%Y-%m-%d"), "event": u.event, "fighter_1": u.fighter_1,
               "fighter_2": u.fighter_2, "bookmaker": u.bookmaker, "f1_decimal": u.f1_decimal,
               "f2_decimal": u.f2_decimal, "notes": notes[i], "bet": False}
        if not r.has_market:
            rec["notes"] = (rec["notes"] + "; " if rec["notes"] else "") + "no price entered: not scored"
            out.append(rec)
            p1s.append(np.nan)
            continue
        one = rows.iloc[[i]]
        z = np.mean([np.mean([m.decision(one)[0] for m in ms]) for ms in models.values()])
        p_a = float(sigmoid(z))
        p1 = 1 - p_a if r.swapped else p_a                 # back to the order in the input file
        p1s.append(p1)
        m1 = (1 / u.f1_decimal) / (1 / u.f1_decimal + 1 / u.f2_decimal)
        e1, e2 = p1 * u.f1_decimal - 1, (1 - p1) * u.f2_decimal - 1
        on1 = e1 >= e2
        edge = e1 if on1 else e2
        rec.update(market_p_f1=round(m1, 4), model_p_f1=round(p1, 4),
                   pick=u.fighter_1 if on1 else u.fighter_2,
                   pick_price=u.f1_decimal if on1 else u.f2_decimal, edge=round(edge, 4),
                   bet=bool(rule["edge_threshold"] < edge <= rule["edge_cap"]))
        if edge > rule["edge_cap"]:
            rec["notes"] = (rec["notes"] + "; " if rec["notes"] else "") + "edge above the cap: skipped as likely model error"
        out.append(rec)
    fin = None
    if finish is not None:
        sc = scenarios(new_clean, new_pre)                 # first all "fighter 1 wins" rows, then "fighter 2 wins"
        d, n, p1 = finish_distribution(sc, finish).to_numpy(), len(new_clean), np.array(p1s)[:, None]
        fin = pd.DataFrame(np.hstack([p1 * d[:n], (1 - p1) * d[n:]]), columns=FINISH_COLS)
    return pd.DataFrame(out).reindex(columns=LEDGER_COLS), n_fit, last, fin


# ---------------------------------------------------------------- ledger
LEDGER = ROOT / "paper" / "ledger.csv"
UPCOMING = ROOT / "paper" / "upcoming.csv"
VOID_WORDS = ("draw", "nc", "no contest", "void")


def load_ledger():
    if LEDGER.exists():
        return pd.read_csv(LEDGER).reindex(columns=LEDGER_COLS)
    return pd.DataFrame(columns=LEDGER_COLS)


def _fight_key(d):
    return d.date.astype(str) + "|" + d.fighter_1.astype(str) + "|" + d.fighter_2.astype(str)


def append_ledger(res, logged_at):
    """Add newly scored fights. A fight already in the ledger keeps its first logged price.
    Returns (rows added, rows skipped)."""
    led = load_ledger()
    priced = res[res.model_p_f1.notna()]
    new = priced[~_fight_key(priced).isin(set(_fight_key(led)))].copy()
    new["logged_at"] = logged_at
    LEDGER.parent.mkdir(exist_ok=True)
    frames = [d for d in (led, new) if len(d)]
    (pd.concat(frames, ignore_index=True) if frames else led)[LEDGER_COLS].to_csv(LEDGER, index=False)
    return len(new), len(priced) - len(new)


def settle(led):
    """Per-fight outcomes for the settled rows of a ledger.
    Adds: decided, f1_won, bet_profit, clv (bets with closing prices only)."""
    d = led[led.winner.notna() & (led.winner.astype(str).str.strip() != "")].copy()
    w = d.winner.astype(str).str.strip()
    d["decided"] = (w == d.fighter_1) | (w == d.fighter_2)
    d["unrecognised"] = ~d.decided & ~w.str.lower().isin(VOID_WORDS)
    d["f1_won"] = np.where(d.decided, (w == d.fighter_1).astype(float), np.nan)
    bet = d.bet.astype(str).str.lower().isin(["true", "1"])
    d["is_bet"] = bet
    won = w == d.pick
    d["bet_profit"] = np.where(~bet, np.nan, np.where(~d.decided, 0.0, np.where(won, d.pick_price - 1, -1.0)))
    p1 = (1 / d.close_f1_decimal) / (1 / d.close_f1_decimal + 1 / d.close_f2_decimal)
    p_pick = np.where(d.pick == d.fighter_1, p1, 1 - p1)
    d["clv"] = np.where(bet, d.pick_price * p_pick - 1, np.nan)
    return d
