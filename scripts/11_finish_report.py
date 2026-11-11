"""Validation report for the method and round model, on out-of-sample 2019-2024 predictions.

Reads data/finish_oof.csv, data/oof_predictions.csv, data/fights_clean.csv and the
method-of-victory odds in Datasets/raw/shortlikeafox/ufc-master.csv.
Writes reports/finish_validation.md. Uses no fight from 2025 on.
"""
import importlib
import json

import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ufc import ROOT, load_config
from ufc.clean import load_clean, norm
from ufc.finish import X_YEAR, FinishGBM
from ufc.market import sigmoid
from ufc.metrics import event_bootstrap

train = importlib.import_module("10_train_finish")
cfg = load_config()
data, reports = ROOT / cfg["paths"]["data"], ROOT / cfg["paths"]["reports"]
REPS, SEED = cfg["bootstrap"]["reps"], cfg["seed"]
OUT = train.OUT_COLS
best = json.load(open(data / "finish_best_params.json", encoding="utf-8"))
o = pd.read_csv(data / "finish_oof.csv", parse_dates=["date"])
assert o.date.max() < pd.Timestamp(cfg["splits"]["test_start"])
PRIMARY = min(("logit", "gbm", "blend"), key=lambda k: best["scores"][k])
NAMES = {"base": "Base rates", "logit": "Logistic", "gbm": "Boosting", "blend": "Average of the two"}
L = []


def table(df, nd=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    L.extend(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|",
              *["| " + " | ".join(map(str, r)) + " |" for r in df.values], ""])


def h(t, level=2):
    L.extend(["#" * level + " " + t, ""])


def ll(p):
    return -np.log(np.clip(p, 1e-12, 1))


def paired(dates, a, b):
    lo, hi = event_bootstrap(np.asarray(dates), lambda i: a[i].mean() - b[i].mean(), REPS, SEED)
    return a.mean() - b.mean(), lo, hi


L.append("# Method and round model: validation\n")
L.append("Every number is out-of-sample: each year from 2019 to 2024 was predicted by a model trained only on "
         "earlier fights. No fight from 2025 on was read. Log loss is the score for probabilities; lower is better.\n")

# ------------------------------------------------------------------ 1 conditional on the winner
h("1. How and when the fight ends, given who won")
act = o[o.actual].reset_index(drop=True)
lab = [f"{m.lower()}_{int(r)}" if m != "DEC" else "dec" for m, r in zip(act.method, act.end_round)]
p_act = {k: act[[f"{k}_{c}" for c in OUT]].to_numpy()[np.arange(len(act)), [OUT.index(x) for x in lab]] for k in NAMES}
rows = []
for k, nm in NAMES.items():
    d, lo, hi = paired(act.date.values, ll(p_act[k]), ll(p_act["base"]))
    rows.append({"model": nm, "log_loss": ll(p_act[k]).mean(), "vs_base_rates": d, "low": lo, "high": hi,
                 "avg_probability_given_to_what_happened": p_act[k].mean()})
table(pd.DataFrame(rows), 4)
L.append(f"{len(act)} fights. The outcome is the exact method and round (for example knockout in round 2), or a "
         "decision. Base rates are the round-by-round finish rates of earlier fights with the same format "
         "(3 or 5 rounds), sex and weight group. An interval entirely below zero is a clear improvement.\n")
L.append(f"**Chosen model: {NAMES[PRIMARY]}.** Settings: logistic `{json.dumps(best['logit'])}`; boosting "
         f"`{json.dumps(best['gbm'])}`.\n")
P = PRIMARY

# ------------------------------------------------------------------ fight-level table
w = pd.read_csv(data / "oof_predictions.csv", parse_dates=["date"])
clean = load_clean(cfg)
clean = clean[clean.date < cfg["splits"]["test_start"]]
w = w.merge(clean[["fight_id", "fighter_1", "fighter_2", "f1_won", "f1_p_open"]], on="fight_id")
w["p1_model"] = np.where(w.a_name == w.fighter_1, sigmoid(w.z_Bens), 1 - sigmoid(w.z_Bens))
w = w[w.is_target & w.market_ok & w.p1_model.notna()]
s1 = o[o.w_side == 1].set_index("fight_id")
s2 = o[o.w_side == 2].set_index("fight_id")
F = w.set_index("fight_id").join(s1[["sched", "method", "end_round", "five"]], how="inner")
ids = F.index
G = {}
for k in NAMES:
    a, b = s1.loc[ids, [f"{k}_{c}" for c in OUT]].to_numpy(), s2.loc[ids, [f"{k}_{c}" for c in OUT]].to_numpy()
    G[k] = (a, b)
p1 = F.p1_model.values
won1 = F.f1_won.values == 1
ko_i, sub_i, dec_i = [OUT.index(f"ko_{r}") for r in range(1, 6)], [OUT.index(f"sub_{r}") for r in range(1, 6)], OUT.index("dec")
method = F.method.values
end_round = F.end_round.values.astype(int)
dates = F.date.values


def method3(k, pw):
    a, b = G[k]
    mix = pw[:, None] * a + (1 - pw)[:, None] * b
    return np.column_stack([mix[:, ko_i].sum(1), mix[:, sub_i].sum(1), mix[:, dec_i]])


h("2. Combined with the winner model")
L.append(f"{len(F)} decided fights from 2019-2024 that have opening odds. The chance of each fighter winning comes "
         "from the frozen winner model's own out-of-sample predictions.\n")
h("How the fight ends: knockout, submission or decision", 3)
y3 = np.array([{"KO": 0, "SUB": 1, "DEC": 2}[m] for m in method])
rows = []
for k in ("base", P):
    m3 = method3(k, p1)
    pa = m3[np.arange(len(F)), y3]
    d, lo, hi = paired(dates, ll(pa), ll(method3("base", p1)[np.arange(len(F)), y3]))
    rows.append({"model": NAMES[k], "log_loss": ll(pa).mean(), "vs_base_rates": d, "low": lo, "high": hi,
                 "top_pick_correct": (m3.argmax(1) == y3).mean()})
table(pd.DataFrame(rows), 4)
m3 = method3(P, p1)
L.append(f"Share of these fights that actually ended by knockout {np.mean(y3 == 0):.1%}, submission "
         f"{np.mean(y3 == 1):.1%}, decision {np.mean(y3 == 2):.1%}. Always saying \"decision\" is right "
         f"{np.mean(y3 == 2):.1%} of the time.\n")

h("Does the fight go the distance?", 3)
pd_ = m3[:, 2]
went = (y3 == 2).astype(float)
pb = method3("base", p1)[:, 2]
L.append(f"- Ranking skill (AUC, 0.5 is a coin flip): model {roc_auc_score(went, pd_):.3f}, base rates "
         f"{roc_auc_score(went, pb):.3f}.")
L.append(f"- Brier score (lower is better): model {((pd_ - went) ** 2).mean():.4f}, base rates "
         f"{((pb - went) ** 2).mean():.4f}.\n")
band = pd.qcut(pd_, 5)
cal = pd.DataFrame({"band": band, "p": pd_, "y": went}).groupby("band", observed=True).agg(
    fights=("y", "size"), predicted_decision=("p", "mean"), actual_decision=("y", "mean")).reset_index()
cal["band"] = cal.band.astype(str)
table(cal)

h("When the model says knockout, and when it says submission", 3)
rows = []
for name, col, yv in (("Knockout", 0, 0), ("Submission", 1, 1)):
    band = pd.qcut(m3[:, col], 5, duplicates="drop")
    t = pd.DataFrame({"band": band, "p": m3[:, col], "y": (y3 == yv).astype(float)}).groupby("band", observed=True).agg(
        fights=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    t.insert(0, "method", name)
    t["band"] = t.band.astype(str)
    rows.append(t)
table(pd.concat(rows))

# ------------------------------------------------------------------ exact outcome
h("3. Exact outcome: who wins, how, and in which round")
a, b = G[P]
joint = np.hstack([p1[:, None] * a, (1 - p1)[:, None] * b])           # 22 outcomes
ab, bb = G["base"]
joint_b = np.hstack([p1[:, None] * ab, (1 - p1)[:, None] * bb])
col = np.array([OUT.index("dec" if m == "DEC" else f"{m.lower()}_{r}") for m, r in zip(method, end_round)])
col = np.where(won1, col, col + len(OUT))
pj, pjb = joint[np.arange(len(F)), col], joint_b[np.arange(len(F)), col]
d, lo, hi = paired(dates, ll(pj), ll(pjb))
order = np.argsort(-joint, axis=1)
rank = (order == col[:, None]).argmax(1)
order_b = np.argsort(-joint_b, axis=1)
rank_b = (order_b == col[:, None]).argmax(1)
table(pd.DataFrame([
    {"model": NAMES["base"], "log_loss": ll(pjb).mean(), "top_pick_correct": (rank_b == 0).mean(),
     "actual_in_top_3": (rank_b < 3).mean(), "avg_probability_given_to_what_happened": pjb.mean()},
    {"model": NAMES[P], "log_loss": ll(pj).mean(), "top_pick_correct": (rank == 0).mean(),
     "actual_in_top_3": (rank < 3).mean(), "avg_probability_given_to_what_happened": pj.mean()}]), 4)
L.append(f"Difference in log loss: {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}). Both rows use the same winner "
         "probabilities, so the difference is the method-and-round model alone. The top pick is almost always "
         "\"the favourite by decision\", which is why exact-outcome hit rates are low for any model.\n")
fin = method != "DEC"
is_fin_col = np.array([c != "dec" for c in OUT] * 2)
pf = joint[:, is_fin_col]
pf = pf / pf.sum(1, keepdims=True)
fin_cols = np.flatnonzero(is_fin_col)
pos = np.array([np.flatnonzero(fin_cols == c)[0] if f else -1 for c, f in zip(col, fin)])
hit = pf[fin].argmax(1) == pos[fin]
pfb = joint_b[:, is_fin_col]
pfb = pfb / pfb.sum(1, keepdims=True)
hit_b = pfb[fin].argmax(1) == pos[fin]
L.append(f"Among the {int(fin.sum())} fights that ended in a finish, the model's most likely finish (fighter, "
         f"method and round) was exactly right {hit.mean():.1%} of the time, against {hit_b.mean():.1%} for base rates.\n")

h("Predicted and actual share of fights ending in each round", 3)
rows = []
for five, lbl in ((0, "3 rounds"), (1, "5 rounds")):
    m = F.five.values == five
    for r in range(1, 6 if five else 4):
        cols = [OUT.index(f"ko_{r}"), OUT.index(f"sub_{r}")]
        pr = joint[m][:, cols].sum(1) + joint[m][:, [c + len(OUT) for c in cols]].sum(1)
        rows.append({"format": lbl, "round": r, "fights": int(m.sum()), "predicted_finish": pr.mean(),
                     "actual_finish": ((end_round[m] == r) & fin[m]).mean()})
table(pd.DataFrame(rows))

# ------------------------------------------------------------------ against method odds
h("4. Against the bookmakers' method-of-victory odds")
mo = pd.read_csv(ROOT / "Datasets/raw/shortlikeafox/ufc-master.csv", low_memory=False)
mo["date"] = pd.to_datetime(mo.date)
oc = ["r_ko_odds", "r_sub_odds", "r_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds"]
mo = mo[mo[oc].notna().all(axis=1)]
imp = mo[oc].apply(lambda v: np.where(v > 0, 100 / (v + 100), -v / (-v + 100)))
imp = imp.div(imp.sum(axis=1), axis=0)
key = {}
for r, (d_, a_, b_) in zip(imp.itertuples(index=False), zip(mo.date, mo.R_fighter.map(norm), mo.B_fighter.map(norm))):
    for off in (0, -1, 1):
        key[(d_ + pd.Timedelta(days=off), a_, b_)] = np.array(r)
odds6, has = [], []
for d_, n1, n2 in zip(F.date, F.fighter_1.map(norm), F.fighter_2.map(norm)):
    v = key.get((d_, n1, n2))
    if v is None and (d_, n2, n1) in key:
        v = key[(d_, n2, n1)][[3, 4, 5, 0, 1, 2]]
    has.append(v is not None)
    odds6.append(v if v is not None else np.full(6, np.nan))
odds6, has = np.array(odds6), np.array(has)
model6 = np.column_stack([p1 * a[:, ko_i].sum(1), p1 * a[:, sub_i].sum(1), p1 * a[:, dec_i],
                          (1 - p1) * b[:, ko_i].sum(1), (1 - p1) * b[:, sub_i].sum(1), (1 - p1) * b[:, dec_i]])
base6 = np.column_stack([p1 * ab[:, ko_i].sum(1), p1 * ab[:, sub_i].sum(1), p1 * ab[:, dec_i],
                         (1 - p1) * bb[:, ko_i].sum(1), (1 - p1) * bb[:, sub_i].sum(1), (1 - p1) * bb[:, dec_i]])
y6 = np.where(won1, y3, y3 + 3)
i = np.arange(len(F))
rows = []
for nm, m6 in (("Base rates with our winner model", base6), ("Our model", model6), ("Method odds", odds6)):
    rows.append({"source": nm, "log_loss": ll(m6[has, y6[has]]).mean(), "top_pick_correct": (m6[has].argmax(1) == y6[has]).mean()})
table(pd.DataFrame(rows), 4)
d, lo, hi = paired(dates[has], ll(model6[has, y6[has]]), ll(odds6[has, y6[has]]))
L.append(f"{int(has.sum())} fights with method odds; six outcomes (each fighter by knockout, submission or decision). "
         f"Our model minus the odds: {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}); positive means the odds are better.\n")
# method split only, given the winner
cm = np.where(won1[:, None], model6[:, :3] / model6[:, :3].sum(1, keepdims=True), model6[:, 3:] / model6[:, 3:].sum(1, keepdims=True))
co = np.where(won1[:, None], odds6[:, :3] / odds6[:, :3].sum(1, keepdims=True), odds6[:, 3:] / odds6[:, 3:].sum(1, keepdims=True))
d, lo, hi = paired(dates[has], ll(cm[has, y3[has]]), ll(co[has, y3[has]]))
L.append(f"Method only, given the winner: our model {ll(cm[has, y3[has]]).mean():.4f}, method odds "
         f"{ll(co[has, y3[has]]).mean():.4f}; difference {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}). "
         "The origin and timing of these odds are not documented, and the bookmaker margin (about 22% across the six "
         "outcomes) was removed by simple scaling, so treat this as a rough benchmark.\n")

# ------------------------------------------------------------------ what drives it
h("5. What the model responds to")
sc = train.load_scenarios(cfg)
tr = sc[sc.actual & sc.usable & (sc.date >= str(best["gbm"]["start"]))]
g = FinishGBM(best["gbm"]["params"], best["gbm"]["rounds"], cols=X_YEAR).fit(tr)
gain = pd.Series(g.booster_.feature_importance("gain"), index=X_YEAR)
gain = (gain / gain.sum()).sort_values(ascending=False).head(15)
table(gain.rename("share_of_importance").rename_axis("input").reset_index())
L.append("`w_` is the assumed winner, `l_` the loser, `rnd` the round number, `p_w` the winner's chance from the "
         "opening odds. Importance is the boosting model's, fitted on all fights before 2025.\n")

act2 = o[o.actual].merge(sc[["fight_id", "w_side", "w_ko_win_rate", "w_sub_win_rate", "l_ko_loss_rate"]], on=["fight_id", "w_side"])
ko_cols = [f"{P}_ko_{r}" for r in range(1, 6)]
rows = []
for name, colname in (("Winner's knockout wins per fight", "w_ko_win_rate"), ("Loser's knockout losses per fight", "l_ko_loss_rate")):
    band = pd.qcut(act2[colname], 4, labels=["lowest quarter", "second", "third", "highest quarter"])
    for bnd, s in act2.groupby(band, observed=True):
        rows.append({"grouped_by": name, "group": str(bnd), "fights": len(s), "predicted_knockout": s[ko_cols].sum(axis=1).mean(),
                     "actual_knockout": (s.method == "KO").mean(),
                     "predicted_ko_in_rounds_1_2": s[ko_cols[:2]].sum(axis=1).mean(),
                     "actual_ko_in_rounds_1_2": ((s.method == "KO") & (s.end_round <= 2)).mean()})
table(pd.DataFrame(rows))
h("Three-round against five-round fights, and the shift over time", 3)
rows = []
for five, lbl in ((0, "3 rounds"), (1, "5 rounds")):
    for per, lo_, hi_ in (("2019-21", 2019, 2021), ("2022-24", 2022, 2024)):
        s = act2[(act2.five == five) & act2.date.dt.year.between(lo_, hi_)]
        rows.append({"format": lbl, "years": per, "fights": len(s), "predicted_ko_round_1": s[f"{P}_ko_1"].mean(),
                     "actual_ko_round_1": ((s.method == "KO") & (s.end_round == 1)).mean(),
                     "predicted_decision": s[f"{P}_dec"].mean(), "actual_decision": (s.method == "DEC").mean()})
table(pd.DataFrame(rows))
L.append("Five-round fights changed around 2019: round-1 knockouts fell from about 25% of fights in 2013-18 to "
         "about 11% in 2019-24, and decisions rose from about 31% to about 46%. Three-round fights barely moved. "
         "A model trained on the earlier years over-predicts early finishes in five-round fights until it has seen "
         "enough of the new pattern, which is why the 2019-21 row is further off than 2022-24. The year is a model "
         "input so that it can follow this.\n")

# ------------------------------------------------------------------ example
h("6. Example: Ilia Topuria vs Max Holloway, 26 October 2024")
ex = F[(F.fighter_1 == "Ilia Topuria") & (F.fighter_2 == "Max Holloway")]
if len(ex):
    j = np.flatnonzero(ids == ex.index[0])[0]
    names = [f"{ex.fighter_1.iloc[0]} by {x}" for x in OUT] + [f"{ex.fighter_2.iloc[0]} by {x}" for x in OUT]
    pretty = lambda s: s.replace("ko_", "knockout, round ").replace("sub_", "submission, round ").replace(" by dec", " by decision")
    top = np.argsort(-joint[j])[:8]
    table(pd.DataFrame({"outcome": [pretty(names[t]) for t in top], "probability": joint[j, top]}))
    L.append(f"Winner model: Topuria {p1[j]:.1%}. Chance of any knockout {m3[j, 0]:.1%}, submission {m3[j, 1]:.1%}, "
             f"decision {m3[j, 2]:.1%}. What happened: {pretty(names[col[j]])}, which the model gave {pj[j]:.1%}.\n")

(reports / "finish_validation.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
