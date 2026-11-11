"""Enrich Datasets/ufc_fights_1993_2026.csv with fighter attributes and betting odds.

Inputs (already downloaded into Datasets/raw/):
  greco1899/ufc_fighter_tott.csv   height, reach, stance, DOB (ufcstats.com)
  nlbair/ufc_all_fights_wide.csv   BestFightOdds opening + closing odds, 2013-02 .. 2026-10 (CC BY 4.0)
  shortlikeafox/ufc-master.csv     moneyline odds, American format, 2010-03 .. 2026-03
  jansen88/odds_raw.csv            moneyline odds, decimal, betmma.tips, 2014-11 .. 2023-12

Output: Datasets/ufc_fights_enriched.csv  (original file is left untouched)

Run:  python scripts/build_enriched_dataset.py
"""
import difflib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "Datasets"
RAW = DS / "raw"

CLASS_LBS = {
    "strawweight": 115, "flyweight": 125, "bantamweight": 135, "featherweight": 145,
    "lightweight": 155, "welterweight": 170, "middleweight": 185,
    "light heavyweight": 205, "heavyweight": 265,
}


def norm(name):
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s.replace("-", " ").replace("'", "").replace(".", ""))
    s = re.sub(r"\b(jr|sr|ii|iii)\b", " ", s)
    return " ".join(s.split())


def bout_lbs(bout_type):
    b = str(bout_type).lower()
    for k in sorted(CLASS_LBS, key=len, reverse=True):
        if k in b:
            return CLASS_LBS[k]
    return np.nan


def american_to_decimal(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a > 0, 1 + a / 100, 1 + 100 / a.abs())


# ---------------------------------------------------------------- fights
fights = pd.read_csv(DS / "ufc_fights_1993_2026.csv", parse_dates=["date"])
fights["n1"] = fights.fighter_1.map(norm)
fights["n2"] = fights.fighter_2.map(norm)
fights["lbs"] = fights.bout_type.map(bout_lbs)

# ---------------------------------------------------------------- attributes
tott = pd.read_csv(RAW / "greco1899/ufc_fighter_tott.csv")
tott["n"] = tott.FIGHTER.map(norm)
h = tott.HEIGHT.str.extract(r"(\d+)\D+(\d+)").astype(float)
tott["height_cm"] = ((h[0] * 12 + h[1]) * 2.54).round(1)
tott["reach_cm"] = (pd.to_numeric(tott.REACH.str.extract(r"(\d+)")[0], errors="coerce") * 2.54).round(1)
tott["weight_lbs"] = pd.to_numeric(tott.WEIGHT.str.extract(r"(\d+)")[0], errors="coerce")
tott["dob"] = pd.to_datetime(tott.DOB, format="%b %d, %Y", errors="coerce")
tott["stance"] = tott.STANCE.where(tott.STANCE.notna() & (tott.STANCE != ""))
tott = tott[tott.n != ""]
by_name = {n: g for n, g in tott.groupby("n")}
ambiguous = set()


def lookup(n, lbs):
    g = by_name.get(n)
    if g is None:
        return None
    if len(g) == 1:
        return g.iloc[0]
    # several fighters share this name: pick the one whose listed weight fits the bout
    ambiguous.add(n)
    if np.isnan(lbs):
        return None
    d = (g.weight_lbs - lbs).abs()
    best = d.nsmallest(2).values
    if np.isnan(best[0]) or (len(best) > 1 and best[0] == best[1]) or best[0] > 25:
        return None
    return g.loc[d.idxmin()]


for side in ("1", "2"):
    rows = [lookup(n, l) for n, l in zip(fights["n" + side], fights.lbs)]
    for col in ("height_cm", "reach_cm", "stance", "dob"):
        fights[f"f{side}_{col}"] = [r[col] if r is not None else np.nan for r in rows]
    fights[f"f{side}_dob"] = pd.to_datetime(fights[f"f{side}_dob"])
    fights[f"f{side}_age"] = ((fights.date - fights[f"f{side}_dob"]).dt.days / 365.25).round(2)

# ---------------------------------------------------------------- odds sources
def implied(american):
    a = pd.to_numeric(american, errors="coerce")
    return pd.Series(np.where(a > 0, 100 / (a + 100), a.abs() / (a.abs() + 100)), index=american.index)


# N: BestFightOdds opening line and closing range, 2013-02 .. 2026-10 (labelled, so preferred)
n = pd.read_csv(RAW / "nlbair/ufc_all_fights_wide.csv", low_memory=False)
n = n[n.has_odds.astype(str).str.lower() == "true"]
N = pd.DataFrame({
    "date": pd.to_datetime(n.event_date), "a": n.f1_name.map(norm), "b": n.f2_name.map(norm),
    "open_a": american_to_decimal(n.f1_open), "open_b": american_to_decimal(n.f2_open),
    # closing line = midpoint of the range across books, in probability space
    "oa": 2 / (implied(n.f1_close_low) + implied(n.f1_close_high)),
    "ob": 2 / (implied(n.f2_close_low) + implied(n.f2_close_high)),
}).dropna(subset=["oa", "ob"]).reset_index(drop=True)

# A: moneyline of unspecified timing, 2010-03 .. 2026-03
m = pd.read_csv(RAW / "shortlikeafox/ufc-master.csv", low_memory=False)
A = pd.DataFrame({
    "date": pd.to_datetime(m.date), "a": m.R_fighter.map(norm), "b": m.B_fighter.map(norm),
    "oa": american_to_decimal(m.R_odds), "ob": american_to_decimal(m.B_odds),
}).dropna(subset=["oa", "ob"]).reset_index(drop=True)

# B: betmma.tips decimal odds of unspecified timing, 2014-11 .. 2023-12
j = pd.read_csv(RAW / "jansen88/odds_raw.csv")
jd = pd.to_datetime(j.date.str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True), format="%d %b %y")
B = pd.DataFrame({
    "date": jd, "a": j.fighter1.map(norm), "b": j.fighter2.map(norm),
    "oa": pd.to_numeric(j.fighter1_odds, errors="coerce"),
    "ob": pd.to_numeric(j.fighter2_odds, errors="coerce"),
}).dropna(subset=["oa", "ob"])
B = B[(B.oa > 1) & (B.ob > 1)].reset_index(drop=True)

SRC_N = "nlbair/ufc-data (BestFightOdds)"
SRC_A = "shortlikeafox/ultimate_ufc_dataset"
SRC_B = "jansen88/ufc-data (betmma.tips)"


def sim(x, y):
    if x == y:
        return 1.0
    xs, ys = set(x.split()), set(y.split())
    if xs and ys and (xs <= ys or ys <= xs):
        return 0.93
    return difflib.SequenceMatcher(None, x, y).ratio()


def match(src, cols=("oa", "ob")):
    """Match on date (+-1 day) and both names. Returns (f1 values, f2 values) for a column pair."""
    idx = {}
    for r in src.itertuples():
        idx.setdefault(r.date.normalize(), []).append(r)
    row = np.full(len(fights), -1)
    swp = np.zeros(len(fights), dtype=bool)
    for i, (d, n1, n2) in enumerate(zip(fights.date, fights.n1, fights.n2)):
        best, best_s = None, 0.0
        for off in (0, -1, 1):
            for r in idx.get(d + pd.Timedelta(days=off), ()):
                s_direct = min(sim(n1, r.a), sim(n2, r.b))
                s_swap = min(sim(n1, r.b), sim(n2, r.a))
                s, swap = (s_direct, False) if s_direct >= s_swap else (s_swap, True)
                if s > best_s:
                    best, best_s = (r.Index, swap), s
            if best_s == 1.0:
                break
        if best and best_s >= 0.80:
            row[i], swp[i] = best

    def pull(ca, cb):
        va = np.where(row >= 0, src[ca].values[np.clip(row, 0, None)], np.nan)
        vb = np.where(row >= 0, src[cb].values[np.clip(row, 0, None)], np.nan)
        return np.where(swp, vb, va), np.where(swp, va, vb)

    return pull


pullN, pullA, pullB = match(N), match(A), match(B)
n1, n2 = pullN("oa", "ob")
no1, no2 = pullN("open_a", "open_b")
a1, a2 = pullA("oa", "ob")
b1, b2 = pullB("oa", "ob")

fights["f1_open_decimal"] = np.round(no1, 3)
fights["f2_open_decimal"] = np.round(no2, 3)
fights["f1_close_decimal"] = np.round(n1, 3)
fights["f2_close_decimal"] = np.round(n2, 3)

# best available line per fight: labelled closing line first, then the unlabelled sources
use_n = ~np.isnan(n1)
use_a = ~use_n & ~np.isnan(a1)
use_b = ~use_n & ~use_a & ~np.isnan(b1)
pick = lambda x, y, z: np.where(use_n, x, np.where(use_a, y, np.where(use_b, z, np.nan)))
fights["f1_odds_decimal"] = np.round(pick(n1, a1, b1), 3)
fights["f2_odds_decimal"] = np.round(pick(n2, a2, b2), 3)
fights["odds_source"] = np.where(use_n, SRC_N, np.where(use_a, SRC_A, np.where(use_b, SRC_B, None)))
fights["odds_type"] = np.where(use_n, "closing", np.where(use_a | use_b, "unspecified", None))
# bookmaker margin removed, so the two probabilities sum to 1
p1, p2 = 1 / fights.f1_odds_decimal, 1 / fights.f2_odds_decimal
fights["f1_implied_prob"] = (p1 / (p1 + p2)).round(4)
fights["f2_implied_prob"] = (p2 / (p1 + p2)).round(4)

# flag fights where another source implies a very different probability than the
# line we kept (likely a flipped or mis-entered line somewhere); treat as unreliable
with np.errstate(invalid="ignore"):
    novig = lambda x, y: (1 / x) / (1 / x + 1 / y)
    pn, pa, pb = novig(n1, n2), novig(a1, a2), novig(b1, b2)
    pbest = fights.f1_implied_prob.values
    gaps = np.abs(np.vstack([pn, pa, pb]) - pbest)
fights["odds_conflict"] = np.nan_to_num(gaps, nan=0).max(axis=0) > 0.15

# ---------------------------------------------------------------- report
yr = fights.date.dt.year
rep = pd.DataFrame({
    "fights": fights.groupby(yr).size(),
    "odds": fights.groupby(yr).f1_odds_decimal.count(),
    "dob_both": (fights.f1_dob.notna() & fights.f2_dob.notna()).groupby(yr).sum(),
    "reach_both": (fights.f1_reach_cm.notna() & fights.f2_reach_cm.notna()).groupby(yr).sum(),
})
print(rep.T.to_string())
print(f"\nodds matched: {fights.f1_odds_decimal.notna().sum()} of {len(fights)}"
      f"  (closing {use_n.sum()}, unspecified {use_a.sum()} + {use_b.sum()})")
print(f"flagged odds_conflict: {int(fights.odds_conflict.sum())}")
with np.errstate(invalid="ignore"):
    po = novig(no1, no2)
    ok = ~np.isnan(pa) & ~np.isnan(pn)
    print(f"shortlikeafox line vs BestFightOdds over {ok.sum()} fights, mean abs probability gap: "
          f"opening {np.abs(pa - po)[ok].mean():.4f}, closing {np.abs(pa - pn)[ok].mean():.4f}")
last = fights[fights.f1_odds_decimal.isna() & (fights.date >= "2010-03-21")]
print("fights since 2010-03-21 without odds, by year:",
      last.groupby(last.date.dt.year).size().to_dict())
for label, c1, c2 in (("opening", "f1_open_decimal", "f2_open_decimal"),
                      ("closing", "f1_close_decimal", "f2_close_decimal")):
    v = fights[(fights.result == "win") & fights[c1].notna() & (fights[c1] != fights[c2])]
    fw = np.where(v[c1] < v[c2], v.winner == v.fighter_1, v.winner == v.fighter_2)
    print(f"{label} favourite won: {fw.mean():.4f} over {len(v)} fights")
w = fights[(fights.result == "win") & fights.f1_odds_decimal.notna()
           & (fights.f1_odds_decimal != fights.f2_odds_decimal)]
fav_won = np.where(w.f1_odds_decimal < w.f2_odds_decimal, w.winner == w.fighter_1, w.winner == w.fighter_2)
print(f"favourite won: {fav_won.mean():.4f} over {len(w)} decided fights with odds")
no_dob = sorted(set(fights.n1[fights.f1_dob.isna()]) | set(fights.n2[fights.f2_dob.isna()]))
print(f"fighters without DOB: {len(no_dob)};  names shared by several fighters: {len(ambiguous)}")

out = DS / "ufc_fights_enriched.csv"
fights.drop(columns=["n1", "n2", "lbs"]).to_csv(out, index=False, date_format="%Y-%m-%d")
print("wrote", out, len(fights), "rows")
