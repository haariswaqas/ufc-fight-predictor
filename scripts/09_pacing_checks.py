"""Step 1 of the method/round work: test the pacing and fade ideas against the data.

Descriptive only: no model is fitted. Uses fights before 2025-01-01, so the 2025-26
window stays untouched for the method and round models.
Writes reports/pacing_checks.md.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd

from ufc import ROOT, load_config
from ufc.clean import load_clean

cfg = load_config()
CUT = cfg["splits"]["test_start"]
clean = load_clean(cfg)
clean = clean[clean.date < CUT].reset_index(drop=True)
pre = pd.read_csv(ROOT / "data/fighter_prefight.csv", parse_dates=["date"])
pre = pre[pre.date < CUT]
L = ["# Pacing and fade checks\n",
     "Descriptive checks on fights before 2025. No model was fitted. `n` is the number of fights or "
     "fighter appearances behind each figure.\n"]


def table(df, nd=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    L.extend(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|",
              *["| " + " | ".join(map(str, r)) + " |" for r in df.values], ""])


def h(t, level=2):
    L.extend(["#" * level + " " + t, ""])


modern = clean[(clean.date >= "2010-01-01") & clean.sched_rounds.isin([3, 5])].copy()
modern["finish"] = np.where(modern.is_target & modern.method_group.isin(["KO", "SUB"]), modern.method_group, "")

# ------------------------------------------------------------------ A. when fights end
h("A. When fights end: 3-round against 5-round fights (2010-2024)")
rows = []
for sr, g in modern.groupby("sched_rounds"):
    for r in range(1, int(sr) + 1):
        reached = g[g["round"] >= r]
        here = reached[reached["round"] == r]
        rows.append({"scheduled": f"{int(sr)} rounds", "round": r, "fights_reaching_it": len(reached),
                     "ko_in_this_round": (here.finish == "KO").sum() / len(reached),
                     "sub_in_this_round": (here.finish == "SUB").sum() / len(reached)})
table(pd.DataFrame(rows))
out = modern.groupby("sched_rounds").apply(lambda g: pd.Series({
    "fights": len(g), "ko": (g.finish == "KO").mean(), "submission": (g.finish == "SUB").mean(),
    "decision": (g.is_target & (g.method_group == "DEC")).mean(),
    "ko_in_rounds_1_2": ((g.finish == "KO") & (g["round"] <= 2)).mean()}), include_groups=False).reset_index()
out["sched_rounds"] = out.sched_rounds.astype(int)
out["fights"] = out.fights.astype(int)
table(out)

# same fighters in both formats
long = pd.concat([modern.assign(k=modern.f1_key), modern.assign(k=modern.f2_key)])
long["early_ko"] = (long.finish == "KO") & (long["round"] <= 2)
long["r1_ko"] = (long.finish == "KO") & (long["round"] == 1)
cnt = long.groupby(["k", "sched_rounds"]).size().unstack(fill_value=0)
both = cnt[(cnt.get(3, 0) >= 3) & (cnt.get(5, 0) >= 3)].index
w = long[long.k.isin(both)].groupby(["k", "sched_rounds"])[["early_ko", "r1_ko"]].mean().unstack()
L.append(f"**Same fighters in both formats.** {len(both)} fighters have at least three 3-round and three "
         f"5-round fights. Averaging each fighter's own rates: a knockout in round 1 in "
         f"{w[('r1_ko', 3)].mean():.1%} of their 3-round fights against {w[('r1_ko', 5)].mean():.1%} of their "
         f"5-round fights; a knockout in rounds 1-2 in {w[('early_ko', 3)].mean():.1%} against "
         f"{w[('early_ko', 5)].mean():.1%}.\n")

# ------------------------------------------------------------------ round-by-round output
nl_cols = ["fight_id"] + [f"f{s}_r{r}_{c}" for s in (1, 2) for r in (1, 2, 3, 4, 5)
                          for c in ("sig_str_att", "sig_str_landed")]
nl = pd.read_csv(ROOT / cfg["paths"]["nlbair"], usecols=nl_cols, low_memory=False)
m = modern.merge(nl, on="fight_id")
secs_last = m.time.str.split(":").map(lambda t: int(t[0]) * 60 + int(t[1]))
parts = []
for s, o in ((1, 2), (2, 1)):
    d = pd.DataFrame({"fight_id": m.fight_id, "date": m.date, "k": m[f"f{s}_key"], "side": s,
                      "sched_rounds": m.sched_rounds, "end_round": m["round"], "end_secs": secs_last})
    for r in (1, 2, 3, 4, 5):
        full = (m["round"] > r) | ((m["round"] == r) & (secs_last == 300))      # round completed
        d[f"att{r}"] = np.where(full, m[f"f{s}_r{r}_sig_str_att"], np.nan) / 5
        d[f"abs{r}"] = np.where(full, m[f"f{o}_r{r}_sig_str_landed"], np.nan) / 5
    parts.append(d)
rd = pd.concat(parts, ignore_index=True)

h("B. Pace: do fighters start slower in 5-round fights? (2013-2024)")
c2 = rd.dropna(subset=["att1"]).groupby(["k", "sched_rounds"]).att1.agg(["mean", "size"]).unstack()
ok = c2[(c2[("size", 3)] >= 3) & (c2[("size", 5)] >= 3)]
diff = ok[("mean", 5)] - ok[("mean", 3)]
L.append(f"Significant strikes attempted per minute in round 1, for the {len(ok)} fighters with at least "
         f"three completed first rounds in each format: {ok[('mean', 3)].mean():.2f} in 3-round fights and "
         f"{ok[('mean', 5)].mean():.2f} in 5-round fights. Average change per fighter {diff.mean():+.2f} "
         f"(standard error {diff.std() / np.sqrt(len(diff)):.2f}); {(diff < 0).mean():.0%} of them start "
         "slower in 5-round fights.\n")
rows = []
for sr in (3, 5):
    g = rd[(rd.sched_rounds == sr) & (rd.end_round == sr) & (rd.end_secs == 300)]     # went the distance
    r = {"scheduled": f"{sr} rounds, went the distance", "fighter_appearances": len(g)}
    for i in range(1, sr + 1):
        r[f"round_{i}"] = g[f"att{i}"].mean()
    rows.append(r)
table(pd.DataFrame(rows), 2)
L.append("Strikes attempted per minute by round, in fights that went the full distance.\n")

# ------------------------------------------------------------------ C. knockout artists
h("C. Knockout artists: what happens when the early knockout does not come")
p = pre.merge(clean[["fight_id", "sched_rounds", "round", "is_target", "method_group", "f1_won", "f1_p_close",
                     "market_ok"]], on="fight_id")
p = p[(p.date >= "2010-01-01") & p.is_target & p.sched_rounds.isin([3, 5])].copy()
p["won"] = np.where(p.side == 1, p.f1_won, 1 - p.f1_won)
p["implied"] = np.where(p.side == 1, p.f1_p_close, 1 - p.f1_p_close)
exp = p[p.n_fights >= 5]
cut = exp.ko_win_rate.quantile(0.75)
p["ko_artist"] = (p.n_fights >= 5) & (p.ko_win_rate >= cut)
p["other_exp"] = (p.n_fights >= 5) & (p.ko_win_rate < cut)
opp = p[["fight_id", "side", "ko_artist"]].assign(side=lambda d: 3 - d.side).rename(columns={"ko_artist": "opp_ko_artist"})
p = p.merge(opp, on=["fight_id", "side"], how="left")
L.append(f"A knockout artist here is a fighter with at least 5 UFC fights whose knockout wins per fight, "
         f"before the fight in question, are in the top quarter ({cut:.2f} or more after adjustment for "
         f"sample size). {int(p.ko_artist.sum())} appearances qualify.\n")
rows = []
for label, g in (("Knockout artists", p[p.ko_artist]), ("Other experienced fighters", p[p.other_exp])):
    for cond, s in (("All fights", g), ("Fight reached round 3", g[g["round"] >= 3]),
                    ("5-round fight reached round 4", g[(g.sched_rounds == 5) & (g["round"] >= 4)])):
        mk = s[s.market_ok & s.implied.notna()]
        rows.append({"group": label, "situation": cond, "n": len(s), "won": s.won.mean(),
                     "won_by_ko": ((s.won == 1) & (s.method_group == "KO")).mean(),
                     "lost_by_ko": ((s.won == 0) & (s.method_group == "KO")).mean(),
                     "pre_fight_odds_implied": mk.implied.mean(), "won_where_odds_known": mk.won.mean()})
table(pd.DataFrame(rows))
L.append("`pre_fight_odds_implied` is the closing-odds win probability before the fight, for the fights "
         "that have odds; compare it with `won_where_odds_known` on the same fights.\n")
ka = p[p.ko_artist & (p.won == 1) & (p.method_group == "KO")]
dist = ka["round"].value_counts(normalize=True).sort_index()
L.append("Their knockout wins by round: " + ", ".join(f"round {int(r)} {v:.0%}" for r, v in dist.items()) + ".\n")

# fade in output
rdp = rd.merge(p[["fight_id", "side", "ko_artist", "other_exp"]], on=["fight_id", "side"])
g3 = rdp.dropna(subset=["att1", "att3"])
g3 = g3[g3.att1 > 0]
rows = []
for label, col in (("Knockout artists", "ko_artist"), ("Other experienced fighters", "other_exp")):
    s = g3[g3[col]]
    rows.append({"group": label, "n": len(s), "round_1_attempts_per_min": s.att1.mean(),
                 "round_3_attempts_per_min": s.att3.mean(), "change": (s.att3 - s.att1).mean(),
                 "round_1_absorbed_per_min": s.abs1.mean(), "round_3_absorbed_per_min": s.abs3.mean()})
table(pd.DataFrame(rows), 2)
L.append("Output in round 1 and round 3, for fights where both rounds were completed (2013-2024).\n")

# ------------------------------------------------------------------ D. is fading a stable trait
h("D. Is pace, and fading, a stable trait of a fighter?")
g = rd.dropna(subset=["att1", "att3"]).sort_values(["k", "date"]).copy()
g["fade"] = g.att3 - g.att1
for col in ("att1", "fade"):
    grp = g.groupby("k")[col]
    g["prior_" + col] = (grp.cumsum() - g[col]) / grp.cumcount().replace(0, np.nan)
g["n_prior"] = g.groupby("k").cumcount()
s = g[g.n_prior >= 3]
L.append(f"For {len(s)} appearances with at least 3 earlier fights that completed three rounds: the "
         f"correlation between a fighter's earlier average round-1 pace and their round-1 pace this time is "
         f"{np.corrcoef(s.prior_att1, s.att1)[0, 1]:.2f}. For the change from round 1 to round 3 (the fade) "
         f"it is {np.corrcoef(s.prior_fade, s.fade)[0, 1]:.2f}. Near 0 means no carry-over; near 1 means a "
         "fixed trait.\n")

# ------------------------------------------------------------------ E. absorption and getting stopped
h("E. Do fighters who absorb more strikes get knocked out more?")
q = p[p.n_fights >= 3].copy()
rows = []
for name, col in (("Strikes absorbed per minute, career", "c_sapm"), ("Strikes absorbed per minute, recent", "r_sapm"),
                  ("Knockdowns suffered per 15 min, career", "c_kda_p15"), ("Knockout losses per fight", "ko_loss_rate"),
                  ("Strikes landed per minute, career", "c_slpm")):
    band = pd.qcut(q[col], 4, labels=["lowest quarter", "second", "third", "highest quarter"], duplicates="drop")
    r = {"measure_before_the_fight": name}
    for b, s in q.groupby(band, observed=True):
        r[str(b)] = ((s.won == 0) & (s.method_group == "KO")).mean()
    rows.append(r)
table(pd.DataFrame(rows))
L.append(f"Share of fights lost by knockout, by quarter of each pre-fight measure ({len(q)} appearances with "
         f"at least 3 earlier fights; overall {((q.won == 0) & (q.method_group == 'KO')).mean():.3f}).\n")

b = pd.read_csv(ROOT / "data/fighter_bouts.csv", parse_dates=["date"])
b = b[b.date < CUT]
tot = b.groupby("name").agg(landed=("sig_str_landed", "sum"), absorbed=("opp_sig_str_landed", "sum"),
                            fights=("fight_id", "size"))
L.append("Most significant strikes landed in the UFC before 2025: "
         + ", ".join(f"{n} {int(v)}" for n, v in tot.landed.nlargest(5).items()) + ".")
L.append("Most absorbed: " + ", ".join(f"{n} {int(v)}" for n, v in tot.absorbed.nlargest(5).items()) + ".\n")

(ROOT / "reports/pacing_checks.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
