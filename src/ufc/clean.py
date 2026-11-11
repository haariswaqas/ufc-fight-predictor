"""Turn the enriched fight file into a clean, modelling-ready fight table.

One row per fight, oldest first. Nothing here looks at who won except to record
the target; all pre-fight features are built later in history.py.
"""
import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from . import ROOT

STAT_COLS = ["kd", "sub_att", "rev", "sig_str_landed", "sig_str_att", "total_str_landed",
             "total_str_att", "td_landed", "td_att", "ctrl_seconds"]

# per-fighter round-level striking kept from the nlbair file
ROUND_COLS = ["r1_att", "r1_landed", "late_att", "late_landed"]

CLASS_LBS = {
    "strawweight": 115, "flyweight": 125, "bantamweight": 135, "featherweight": 145,
    "lightweight": 155, "welterweight": 170, "middleweight": 185,
    "light heavyweight": 205, "heavyweight": 265,
}


def norm(name):
    """Normalised fighter name (same rule as scripts/build_enriched_dataset.py)."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace("-", " ").replace("'", "").replace(".", ""))
    s = re.sub(r"\b(jr|sr|ii|iii)\b", " ", s)
    return " ".join(s.split())


def ident(name):
    """Name key for fighter identity. Unlike norm() it keeps Jr/Sr/II/III, so a father
    and son with the same name stay two people."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace("-", " ").replace("'", "").replace(".", ""))
    return " ".join(s.split())


def bout_lbs(bout_type):
    b = str(bout_type).lower()
    for k in sorted(CLASS_LBS, key=len, reverse=True):
        if k in b:
            return float(CLASS_LBS[k])
    return np.nan


def _mmss(t):
    m, s = str(t).split(":")
    return int(m) * 60 + int(s)


def _round_minutes(fmt):
    m = re.search(r"\(([\d-]+)\)", str(fmt))
    return [int(x) for x in m.group(1).split("-")] if m else None


def fight_seconds(round_, time_, fmt):
    """Seconds actually fought. Overtime periods are counted as rounds by ufcstats."""
    t = _mmss(time_)
    mins = _round_minutes(fmt)
    if mins is None:  # "No Time Limit" or blank: always a single round
        return float(t)
    return float(sum(mins[: int(round_) - 1]) * 60 + t)


def scheduled_rounds(fmt):
    m = re.match(r"\s*(\d+) Rnd", str(fmt))
    return float(m.group(1)) if m else np.nan


def method_group(method):
    m = str(method)
    if m.startswith("KO/TKO") or m.startswith("TKO"):
        return "KO"
    if m.startswith("Submission"):
        return "SUB"
    if m.startswith("Decision"):
        return "DEC"
    return "OTHER"


def _fight_id(row):
    m = re.search(r"fight-details/(\w+)", str(row.fight_url))
    if m:
        return m.group(1)
    raw = f"{row.date:%Y-%m-%d}|{row.fighter_1}|{row.fighter_2}|{row['round']}|{row.time}"
    return "x" + hashlib.md5(raw.encode()).hexdigest()[:15]


def _novig(dec_a, dec_b):
    pa, pb = 1 / dec_a, 1 / dec_b
    return pa / (pa + pb)


def build_clean(cfg):
    d = pd.read_csv(ROOT / cfg["paths"]["enriched"], parse_dates=["date"])
    d["file_row"] = np.arange(len(d))
    d["fight_id"] = d.apply(_fight_id, axis=1)
    assert d.fight_id.is_unique

    # ---- fighter identity: ufcstats fighter ids via the nlbair file, joined on fight id
    round_cols = [f"f{s}_r{r}_sig_str_{k}" for s in (1, 2) for r in (1, 2, 3, 4, 5) for k in ("att", "landed")]
    n = pd.read_csv(ROOT / cfg["paths"]["nlbair"], low_memory=False,
                    usecols=["fight_id", "f1_name", "f2_name", "f1_id", "f2_id"] + round_cols)
    d = d.merge(n.rename(columns={"f1_id": "nl_f1_id", "f2_id": "nl_f2_id",
                                  "f1_name": "nl_f1", "f2_name": "nl_f2"}),
                on="fight_id", how="left")
    d["lbs"] = d.bout_type.map(bout_lbs)
    name_ids = {}   # normalised name -> {fighter id: [weights fought at]}
    for side in ("1", "2"):
        j = d[d[f"nl_f{side}_id"].notna()]
        for nm, fid, lbs in zip(j[f"fighter_{side}"].map(ident), j[f"nl_f{side}_id"], j.lbs):
            name_ids.setdefault(nm, {}).setdefault(fid, []).append(lbs)

    def key(name, known_id, lbs):
        if isinstance(known_id, str):
            return known_id
        nm = ident(name)
        ids = name_ids.get(nm)
        if not ids:
            return "n:" + nm
        if len(ids) == 1:
            return next(iter(ids))
        # same name, several fighters: pick the one who fights nearest this weight
        med = {i: np.nanmedian(w) for i, w in ids.items()}
        return min(med, key=lambda i: abs(med[i] - lbs) if lbs == lbs else np.inf)

    d["f1_key"] = [key(a, b, c) for a, b, c in zip(d.fighter_1, d.nl_f1_id, d.lbs)]
    d["f2_key"] = [key(a, b, c) for a, b, c in zip(d.fighter_2, d.nl_f2_id, d.lbs)]
    assert (d.f1_key != d.f2_key).all()

    # ---- target and result
    d["is_target"] = d.result.eq("win")
    d["f1_won"] = np.where(d.is_target, (d.winner == d.fighter_1).astype(float), np.nan)
    assert ((d.winner == d.fighter_1) | (d.winner == d.fighter_2))[d.is_target].all()
    d["method_group"] = d.method.map(method_group)

    # ---- duration and context
    d["fight_secs"] = [fight_seconds(r, t, f) for r, t, f in zip(d["round"], d.time, d.time_format)]
    d["sched_rounds"] = d.time_format.map(scheduled_rounds)
    d["five_round"] = d.sched_rounds.eq(5)
    d["female"] = d.bout_type.str.contains("Women", case=False)
    # women's catch weight bouts are labelled plain "Catch Weight Bout"
    women = set(d.f1_key[d.female]) | set(d.f2_key[d.female])
    d["female"] = d.female | (d.f1_key.isin(women) & d.f2_key.isin(women))
    d["ufc_title"] = d.title_bout & ~d.bout_type.str.contains("Tournament", case=False)
    d["heavy"] = d.lbs.ge(205) & ~d.female

    # ---- round-by-round striking (2013 on, from the nlbair file): first round and rounds 3+
    has_rounds = d.f1_r1_sig_str_att.notna() & d.f2_r1_sig_str_att.notna()
    end_secs = d.time.map(_mmss)

    def round_secs(r):
        return np.where(d["round"] > r, 300.0, np.where(d["round"] == r, end_secs, 0.0))

    d["r1_secs"] = np.where(has_rounds, round_secs(1), np.nan)
    d["late_secs"] = np.where(has_rounds, sum(round_secs(r) for r in (3, 4, 5)), np.nan)
    for s in "12":
        d[f"f{s}_r1_att"] = d[f"f{s}_r1_sig_str_att"]
        d[f"f{s}_r1_landed"] = d[f"f{s}_r1_sig_str_landed"]
        for k in ("att", "landed"):
            late = sum(d[f"f{s}_r{r}_sig_str_{k}"].fillna(0) for r in (3, 4, 5))
            d[f"f{s}_late_{k}"] = np.where(has_rounds, late, np.nan)

    # ---- market: labelled opening and closing lines only
    d["has_market"] = d.f1_open_decimal.notna() & d.f1_close_decimal.notna()
    d["f1_p_open"] = _novig(d.f1_open_decimal, d.f2_open_decimal)
    d["f1_p_close"] = _novig(d.f1_close_decimal, d.f2_close_decimal)
    # an opening line that is the mirror image of the close after an extreme move is a
    # side-swapped entry in the source, not a real price; treated like odds_conflict
    d["open_suspect"] = (d.has_market & ((d.f1_p_open > 0.5) != (d.f1_p_close > 0.5))
                         & ((d.f1_p_close - d.f1_p_open).abs() > 0.40)
                         & (((1 - d.f1_p_open) - d.f1_p_close).abs() < 0.05))
    d["market_ok"] = d.has_market & ~d.odds_conflict & ~d.open_suspect

    # ---- one profile per fighter: the enriched file attached profiles by name spelling,
    # so a few rows of one fighter carry a namesake's profile or none at all
    for col in ("height_cm", "reach_cm", "stance", "dob"):
        long = pd.concat([d[["f1_key", f"f1_{col}"]].set_axis(["k", "v"], axis=1),
                          d[["f2_key", f"f2_{col}"]].set_axis(["k", "v"], axis=1)]).dropna()
        best = long.groupby("k").v.agg(lambda s: s.value_counts().index[0])
        for s in "12":
            d[f"f{s}_{col}"] = d[f"f{s}_key"].map(best)

    # oldest first; within an event the file lists the main event first
    d = d.sort_values(["date", "file_row"], ascending=[True, False]).reset_index(drop=True)
    keep = (["fight_id", "date", "event", "fighter_1", "fighter_2", "f1_key", "f2_key", "result",
             "is_target", "f1_won", "method", "method_group", "round", "time", "time_format",
             "fight_secs", "sched_rounds", "five_round", "bout_type", "lbs", "female", "ufc_title",
             "heavy"]
            + [f"f{s}_{c}" for s in "12" for c in STAT_COLS]
            + [f"f{s}_{c}" for s in "12" for c in ("height_cm", "reach_cm", "stance", "dob")]
            + ["r1_secs", "late_secs"] + [f"f{s}_{c}" for s in "12" for c in ROUND_COLS]
            + ["has_market", "f1_p_open", "f1_p_close", "f1_open_decimal", "f2_open_decimal",
               "f1_close_decimal", "f2_close_decimal", "odds_conflict", "open_suspect",
               "market_ok"])
    return d[keep]


def load_clean(cfg):
    return pd.read_csv(ROOT / cfg["paths"]["data"] / "fights_clean.csv",
                       parse_dates=["date", "f1_dob", "f2_dob"])
