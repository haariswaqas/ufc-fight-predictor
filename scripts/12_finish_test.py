"""Step 4 of the method/round work: the single test run of the method and round model.

Refits the frozen finish models (and the frozen winner model) on fights before the test
period, scores 2025-01-01 to 2026-10-10 once, and writes reports/finish_test.md plus
data/finish_test_predictions.csv. It refuses to run again once the report exists.

    python scripts/12_finish_test.py --freeze     write finish_frozen.json (no test data read)
    python scripts/12_finish_test.py --dry-run    same code with 2023-24 standing in for the test
    python scripts/12_finish_test.py              the real run
"""
import hashlib
import json
import sys

import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ufc import ROOT, load_config
from ufc.clean import load_clean, norm
from ufc.finish import X_COLS, X_YEAR, FinishGBM, FinishLogit, base_rate_distribution, distribution, scenarios
from ufc.market import sigmoid
from ufc.metrics import event_bootstrap
from ufc.models import design
from ufc.paper import fit_frozen, load_frozen

cfg = load_config()
data, reports = ROOT / cfg["paths"]["data"], ROOT / cfg["paths"]["reports"]
FROZEN = ROOT / "finish_frozen.json"
REPS, SEED = cfg["bootstrap"]["reps"], cfg["seed"]
OUT = [f"{m}_{r}" for m in ("ko", "sub") for r in range(1, 6)] + ["dec"]
KO_I, SUB_I, DEC_I = [OUT.index(f"ko_{r}") for r in range(1, 6)], [OUT.index(f"sub_{r}") for r in range(1, 6)], OUT.index("dec")
NAMES = {"base": "Base rates", "logit": "Logistic", "gbm": "Boosting", "blend": "Average of the two (primary)"}


def sha(name):
    return hashlib.sha256((data / name).read_bytes()).hexdigest()


def freeze():
    best = json.load(open(data / "finish_best_params.json", encoding="utf-8"))
    frozen = {
        "frozen_on": "2026-11-11",
        "primary_model": "blend: mean of the logistic and boosting outcome probabilities",
        "logit": best["logit"], "gbm": best["gbm"], "gbm_seeds": [0, 1, 2],
        "base_rates": {"start": 2010, "by": ["five", "female", "heavy_grp"]},
        "winner_probabilities": "frozen winner model (config_frozen.json, B average), refitted on fights before the test",
        "test_window": [cfg["splits"]["test_start"], cfg["splits"]["test_end"]],
        "pass_rule": {
            "1": "method and round given the winner: log loss lower than base rates, 95% event-bootstrap interval "
                 "of the difference below zero (all decided 3- and 5-round fights)",
            "2": "knockout / submission / decision with the winner model: log loss lower than base rates, interval "
                 "below zero (fights with usable opening odds)",
            "3": "exact outcome (winner, method, round) with the winner model: log loss lower than base rates with "
                 "the same winner probabilities, interval below zero",
            "note": "the comparison with bookmakers' method odds is reported but is not part of the pass rule",
        },
        "fighter_prefight_sha256": sha("fighter_prefight.csv"),
        "validation_scores": best["scores"],
    }
    FROZEN.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print("wrote", FROZEN)


def ll(p):
    return -np.log(np.clip(p, 1e-12, 1))


def run(train_end, test_start, test_end, report_path, pred_path, check_ledger, title):
    fz = json.load(open(FROZEN, encoding="utf-8"))
    clean = load_clean(cfg)
    clean = clean[clean.date <= test_end].reset_index(drop=True)
    pre_all = pd.read_csv(data / "fighter_prefight.csv", parse_dates=["date"])
    pre = pre_all[pre_all.date <= test_end]
    sc = scenarios(clean, pre)
    sc["heavy_grp"] = (sc.lbs >= 205).astype(float)
    hist = sc[sc.date < train_end]
    te = sc[sc.usable & (sc.date >= test_start) & (sc.date <= test_end)]
    assert hist.date.max() < te.date.min()

    def train_rows(start):
        return hist[hist.actual & hist.usable & (hist.date >= str(start))]

    D = {}
    s = fz["logit"]
    D["logit"] = distribution(te, FinishLogit(s["c"], cols=X_YEAR if s["year"] else X_COLS).fit(train_rows(s["start"])).hazards)
    s = fz["gbm"]
    tr = train_rows(s["start"])
    D["gbm"] = sum(distribution(te, FinishGBM(s["params"], s["rounds"], seed=sd, cols=X_YEAR if s["year"] else X_COLS)
                                .fit(tr).hazards) for sd in fz["gbm_seeds"]) / len(fz["gbm_seeds"])
    D["blend"] = (D["logit"] + D["gbm"]) / 2
    b = fz["base_rates"]
    D["base"] = base_rate_distribution(hist[hist.date >= str(b["start"])], te, by=tuple(b["by"]))
    for d in D.values():
        assert np.allclose(d.sum(axis=1), 1)

    # winner probabilities from the frozen winner model, refitted on fights before the test
    f = design(pd.read_csv(data / "features.csv", parse_dates=["date"]), pre_all)
    f = f[f.date <= test_end].reset_index(drop=True)
    models, _, _ = fit_frozen(f[f.date < train_end].reset_index(drop=True), load_frozen())
    wt = f[(f.date >= test_start) & (f.date <= test_end) & f.has_market].reset_index(drop=True)
    z = np.mean([np.mean([m.decision(wt) for m in ms], axis=0) for ms in models.values()], axis=0)
    wt["p_a"] = sigmoid(z)
    wt["p1_model"] = np.where(wt.swapped, 1 - wt.p_a, wt.p_a)
    led = pd.read_csv(check_ledger)                      # same predictions as the winner model's own run
    chk = led.merge(wt[["fight_id", "a_name", "p_a"]], on="fight_id")
    mine = np.where(chk.pick == chk.a_name, chk.p_a, 1 - chk.p_a)
    assert len(chk) == len(led) and np.allclose(mine, chk.model_prob, atol=1e-6), "winner model not reproduced"

    o = te[["fight_id", "date", "w_side", "w_name", "l_name", "sched", "actual", "method", "end_round", "five"]].copy()
    for k, d in D.items():
        for c in OUT:
            o[f"{k}_{c}"] = d[c].values
    o = o.merge(wt[["fight_id", "p1_model"]], on="fight_id", how="left")
    o.to_csv(pred_path, index=False, date_format="%Y-%m-%d")

    L = [f"# {title}\n",
         f"Models fitted on fights before {train_end}; scored on {test_start} to {test_end}. Settings were frozen in "
         "`finish_frozen.json` before this run. Log loss is the score for probabilities; lower is better.\n"]

    def table(df, nd=4):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
        L.extend(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|",
                  *["| " + " | ".join(map(str, r)) + " |" for r in df.values], ""])

    def paired(dates, a, b_):
        lo, hi = event_bootstrap(np.asarray(dates), lambda i: a[i].mean() - b_[i].mean(), REPS, SEED)
        return a.mean() - b_.mean(), lo, hi

    # ---------------------------------------------------------------- 1 given the winner
    act = o[o.actual].reset_index(drop=True)
    lab = [OUT.index("dec" if m == "DEC" else f"{m.lower()}_{int(r)}") for m, r in zip(act.method, act.end_round)]
    pa = {k: act[[f"{k}_{c}" for c in OUT]].to_numpy()[np.arange(len(act)), lab] for k in NAMES}
    L.append(f"## 1. How and when the fight ends, given who won\n\n{len(act)} decided fights "
             f"({act.date.nunique()} event nights).\n")
    rows, res = [], {}
    for k, nm in NAMES.items():
        res[k] = paired(act.date.values, ll(pa[k]), ll(pa["base"]))
        rows.append({"model": nm, "log_loss": ll(pa[k]).mean(), "vs_base_rates": res[k][0], "low": res[k][1],
                     "high": res[k][2], "avg_probability_given_to_what_happened": pa[k].mean()})
    table(pd.DataFrame(rows))
    r1 = res["blend"]
    L.append(f"Validation (2019-2024) for comparison: base rates {fz['validation_scores']['base']:.4f}, primary "
             f"{fz['validation_scores']['blend']:.4f}.\n")

    # ---------------------------------------------------------------- fight level
    c = clean[clean.is_target & clean.market_ok].set_index("fight_id")
    s1, s2 = o[o.w_side == 1].set_index("fight_id"), o[o.w_side == 2].set_index("fight_id")
    F = s1[s1.p1_model.notna()].join(c[["fighter_1", "fighter_2", "f1_won"]], how="inner")
    ids = F.index
    G = {k: (s1.loc[ids, [f"{k}_{x}" for x in OUT]].to_numpy(), s2.loc[ids, [f"{k}_{x}" for x in OUT]].to_numpy())
         for k in NAMES}
    p1, won1, dates = F.p1_model.values, F.f1_won.values == 1, F.date.values
    method, end_round = F.method.values, F.end_round.values.astype(int)
    y3 = np.array([{"KO": 0, "SUB": 1, "DEC": 2}[m] for m in method])
    n, i = len(F), np.arange(len(F))

    def method3(k):
        a, b_ = G[k]
        mix = p1[:, None] * a + (1 - p1)[:, None] * b_
        return np.column_stack([mix[:, KO_I].sum(1), mix[:, SUB_I].sum(1), mix[:, DEC_I]])

    L.append(f"## 2. Combined with the winner model\n\n{n} decided fights with usable opening odds. The chance of "
             "each fighter winning comes from the frozen winner model.\n")
    L.append("### How the fight ends: knockout, submission or decision\n")
    mb, m3 = method3("base"), method3("blend")
    r2 = paired(dates, ll(m3[i, y3]), ll(mb[i, y3]))
    table(pd.DataFrame([
        {"model": NAMES["base"], "log_loss": ll(mb[i, y3]).mean(), "top_pick_correct": (mb.argmax(1) == y3).mean()},
        {"model": NAMES["blend"], "log_loss": ll(m3[i, y3]).mean(), "top_pick_correct": (m3.argmax(1) == y3).mean()}]))
    L.append(f"Difference: {r2[0]:+.4f} (95% interval {r2[1]:+.4f} to {r2[2]:+.4f}). Actual share: knockout "
             f"{np.mean(y3 == 0):.1%}, submission {np.mean(y3 == 1):.1%}, decision {np.mean(y3 == 2):.1%}.\n")
    went = (y3 == 2).astype(float)
    L.append(f"Goes the distance, ranking skill (AUC, 0.5 is a coin flip): model {roc_auc_score(went, m3[:, 2]):.3f}, "
             f"base rates {roc_auc_score(went, mb[:, 2]):.3f}.\n")
    rows = []
    for name, col in (("Knockout", 0), ("Submission", 1), ("Decision", 2)):
        band = pd.qcut(m3[:, col], 5, duplicates="drop")
        t = pd.DataFrame({"band": band, "p": m3[:, col], "y": (y3 == col).astype(float)}).groupby("band", observed=True).agg(
            fights=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
        t.insert(0, "method", name)
        t["band"] = t.band.astype(str)
        rows.append(t)
    L.append("### Calibration: predicted against actual, in five bands\n")
    table(pd.concat(rows), 3)

    # ---------------------------------------------------------------- exact outcome
    a, b_ = G["blend"]
    ab, bb = G["base"]
    joint = np.hstack([p1[:, None] * a, (1 - p1)[:, None] * b_])
    joint_b = np.hstack([p1[:, None] * ab, (1 - p1)[:, None] * bb])
    col = np.array([OUT.index("dec" if m == "DEC" else f"{m.lower()}_{r}") for m, r in zip(method, end_round)])
    col = np.where(won1, col, col + len(OUT))
    pj, pjb = joint[i, col], joint_b[i, col]
    r3 = paired(dates, ll(pj), ll(pjb))
    rank = (np.argsort(-joint, axis=1) == col[:, None]).argmax(1)
    rank_b = (np.argsort(-joint_b, axis=1) == col[:, None]).argmax(1)
    L.append("## 3. Exact outcome: who wins, how, and in which round\n")
    table(pd.DataFrame([
        {"model": NAMES["base"], "log_loss": ll(pjb).mean(), "top_pick_correct": (rank_b == 0).mean(),
         "actual_in_top_3": (rank_b < 3).mean(), "avg_probability_given_to_what_happened": pjb.mean()},
        {"model": NAMES["blend"], "log_loss": ll(pj).mean(), "top_pick_correct": (rank == 0).mean(),
         "actual_in_top_3": (rank < 3).mean(), "avg_probability_given_to_what_happened": pj.mean()}]))
    L.append(f"Difference: {r3[0]:+.4f} (95% interval {r3[1]:+.4f} to {r3[2]:+.4f}). Both rows use the same winner "
             "probabilities.\n")
    fin = method != "DEC"
    is_fin = np.array([x != "dec" for x in OUT] * 2)
    fin_cols = np.flatnonzero(is_fin)
    pos = np.array([np.flatnonzero(fin_cols == x)[0] if f_ else -1 for x, f_ in zip(col, fin)])
    hit = joint[:, is_fin][fin].argmax(1) == pos[fin]
    hit_b = joint_b[:, is_fin][fin].argmax(1) == pos[fin]
    L.append(f"Among the {int(fin.sum())} fights that ended in a finish, the model's most likely finish (fighter, "
             f"method and round) was exactly right {hit.mean():.1%} of the time, against {hit_b.mean():.1%} for base rates.\n")
    L.append("### Predicted and actual share of fights ending in each round\n")
    rows = []
    for five, lbl in ((0, "3 rounds"), (1, "5 rounds")):
        m = F.five.values == five
        for r in range(1, 6 if five else 4):
            cols = [OUT.index(f"ko_{r}"), OUT.index(f"sub_{r}")]
            pr = joint[m][:, cols].sum(1) + joint[m][:, [x + len(OUT) for x in cols]].sum(1)
            rows.append({"format": lbl, "round": r, "fights": int(m.sum()), "predicted_finish": pr.mean(),
                         "actual_finish": ((end_round[m] == r) & fin[m]).mean()})
        rows.append({"format": lbl, "round": "decision", "fights": int(m.sum()), "predicted_finish": m3[m, 2].mean(),
                     "actual_finish": went[m].mean()})
    table(pd.DataFrame(rows), 3)

    # ---------------------------------------------------------------- method odds
    mo = pd.read_csv(ROOT / "Datasets/raw/shortlikeafox/ufc-master.csv", low_memory=False)
    mo["date"] = pd.to_datetime(mo.date)
    oc = ["r_ko_odds", "r_sub_odds", "r_dec_odds", "b_ko_odds", "b_sub_odds", "b_dec_odds"]
    mo = mo[mo[oc].notna().all(axis=1)]
    imp = mo[oc].apply(lambda v: np.where(v > 0, 100 / (v + 100), -v / (-v + 100)))
    imp = imp.div(imp.sum(axis=1), axis=0)
    key = {}
    for r, (d_, a_, b2) in zip(imp.itertuples(index=False), zip(mo.date, mo.R_fighter.map(norm), mo.B_fighter.map(norm))):
        for off in (0, -1, 1):
            key[(d_ + pd.Timedelta(days=off), a_, b2)] = np.array(r)
    odds6, has = [], []
    for d_, n1, n2 in zip(F.date, F.fighter_1.map(norm), F.fighter_2.map(norm)):
        v = key.get((d_, n1, n2))
        if v is None and (d_, n2, n1) in key:
            v = key[(d_, n2, n1)][[3, 4, 5, 0, 1, 2]]
        has.append(v is not None)
        odds6.append(v if v is not None else np.full(6, np.nan))
    odds6, has = np.array(odds6), np.array(has)

    def six(x, y):
        return np.column_stack([p1 * x[:, KO_I].sum(1), p1 * x[:, SUB_I].sum(1), p1 * x[:, DEC_I],
                                (1 - p1) * y[:, KO_I].sum(1), (1 - p1) * y[:, SUB_I].sum(1), (1 - p1) * y[:, DEC_I]])
    model6, base6 = six(a, b_), six(ab, bb)
    y6 = np.where(won1, y3, y3 + 3)
    L.append("## 4. Against the bookmakers' method-of-victory odds (not part of the pass rule)\n")
    if has.sum() >= 50:
        rows = [{"source": nm, "log_loss": ll(m6[has, y6[has]]).mean(), "top_pick_correct": (m6[has].argmax(1) == y6[has]).mean()}
                for nm, m6 in (("Base rates with our winner model", base6), ("Our model", model6), ("Method odds", odds6))]
        table(pd.DataFrame(rows))
        d, lo, hi = paired(dates[has], ll(model6[has, y6[has]]), ll(odds6[has, y6[has]]))
        L.append(f"{int(has.sum())} fights with method odds (the source stops in March 2026); six outcomes. Our model "
                 f"minus the odds: {d:+.4f} (95% interval {lo:+.4f} to {hi:+.4f}); positive means the odds are better.\n")
    else:
        L.append(f"Only {int(has.sum())} fights in the window have method odds; not scored.\n")

    # ---------------------------------------------------------------- pass rule
    ok = [r1[2] < 0, r2[2] < 0, r3[2] < 0]
    L.append("## 5. Pass rule\n")
    for j, (txt, r) in enumerate((("Method and round given the winner, against base rates", r1),
                                  ("Knockout / submission / decision with the winner model, against base rates", r2),
                                  ("Exact outcome with the winner model, against base rates", r3))):
        L.append(f"{j + 1}. {txt}, interval below zero: **{'PASS' if ok[j] else 'FAIL'}** "
                 f"({r[0]:+.4f}, {r[1]:+.4f} to {r[2]:+.4f})")
    L.append(f"\n**Overall: {'PASS' if all(ok) else 'NOT PASSED'}** ({sum(ok)} of 3).\n")
    report_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
    elif "--dry-run" in sys.argv:
        run("2023-01-01", "2023-01-01", "2024-12-31", reports / "finish_dryrun_2023_24.md",
            data / "finish_dryrun_predictions.csv", data / "dryrun_ledger.csv",
            "Dry run: 2023-24 standing in for the test (not the real test)")
    else:
        out = reports / "finish_test.md"
        if out.exists():
            sys.exit(f"{out} already exists. The test is run once; refusing to run again.")
        fz = json.load(open(FROZEN, encoding="utf-8"))
        assert sha("fighter_prefight.csv") == fz["fighter_prefight_sha256"], "fighter_prefight.csv changed after the freeze"
        s = cfg["splits"]
        run(s["test_start"], s["test_start"], s["test_end"], out, data / "finish_test_predictions.csv",
            data / "test_ledger.csv", "Method and round model, final test: 2025-01-01 to 2026-10-10")
