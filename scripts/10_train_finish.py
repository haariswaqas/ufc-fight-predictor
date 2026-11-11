"""Step 3 of the method/round work: tune and train the round-by-round finish model.

Walk-forward on 2019-2024: each year is predicted by a model trained on earlier fights.
Fights from 2025 on are cut off before anything is fitted.

Writes data/finish_tuning.csv, data/finish_best_params.json, data/finish_oof.csv.
"""
import json

import _path  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ufc import ROOT, load_config
from ufc.clean import load_clean
from ufc.finish import (X_COLS, X_YEAR, FinishGBM, FinishLogit, base_rate_distribution, distribution,
                        outcome_label, recency, round_rows, scenarios)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
CHECKPOINTS = [100, 200, 400, 700, 1000]
N_GBM = 36
SEEDS = [0, 1, 2]
OUT_COLS = [f"{m}_{r}" for m in ("ko", "sub") for r in range(1, 6)] + ["dec"]


def load_scenarios(cfg):
    cut = cfg["splits"]["test_start"]
    clean = load_clean(cfg)
    clean = clean[clean.date < cut].reset_index(drop=True)
    pre = pd.read_csv(ROOT / cfg["paths"]["data"] / "fighter_prefight.csv", parse_dates=["date"])
    pre = pre[pre.date < cut]
    sc = scenarios(clean, pre)
    assert sc.date.max() < pd.Timestamp(cut)
    return sc


def fold(sc, year, start):
    tr = sc[sc.actual & sc.usable & (sc.date >= str(start)) & (sc.date < f"{year}-01-01")]
    te = sc[sc.usable & (sc.date.dt.year == year)]
    return tr, te


def cond_log_loss(sc, dist):
    """-log P(actual method and round | actual winner), averaged over fights."""
    a = sc.loc[dist.index]
    a = a[a.actual]
    lab = [outcome_label(m, r) for m, r in zip(a.method, a.end_round)]
    p = dist.loc[a.index].to_numpy()[np.arange(len(a)), [OUT_COLS.index(x) for x in lab]]
    return float(-np.log(np.clip(p, 1e-12, 1)).mean())


def oof(sc, spec):
    parts = {}
    for year in YEARS:
        tr, te = fold(sc, year, spec["start"])
        cols = X_YEAR if spec.get("year") else X_COLS
        w = recency(tr.date, f"{year}-01-01", spec.get("half_life"))
        if spec["kind"] == "gbm":
            ck = spec["rounds"] if isinstance(spec["rounds"], list) else [spec["rounds"]]
            m = FinishGBM(spec["params"], max(ck), seed=spec.get("seed", 0), cols=cols).fit(tr, w)
            for r in ck:
                parts.setdefault(r if len(ck) > 1 else "p", []).append(distribution(te, lambda rr, r=r: m.hazards(rr, r)))
        elif spec["kind"] == "logit":
            m = FinishLogit(spec["c"], cols=cols).fit(tr, w)
            parts.setdefault("p", []).append(distribution(te, m.hazards))
        else:
            parts.setdefault("p", []).append(base_rate_distribution(tr, te, by=tuple(spec["by"])))
    return {k: pd.concat(v) for k, v in parts.items()}


def gbm_grid(n, seed):
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    for _ in range(n):
        yield {"kind": "gbm", "start": pick([2010, 2013]), "rounds": CHECKPOINTS, "year": True,
               "params": {"num_leaves": pick([4, 6, 8, 12, 16]), "max_depth": pick([-1, 3, 4, 5]),
                          "learning_rate": pick([0.01, 0.02, 0.04]), "min_child_samples": pick([40, 80, 150, 300]),
                          "feature_fraction": pick([0.4, 0.6, 0.8]), "lambda_l2": pick([0.0, 5.0, 20.0]),
                          "min_split_gain": pick([0.0, 0.01])}}


def eval_gbm(sc, spec):
    d = oof(sc, spec)
    return [{"kind": "gbm", "start": spec["start"], "year": True, "rounds": r, **spec["params"],
             "log_loss": cond_log_loss(sc, d[r])} for r in CHECKPOINTS]


def eval_logit(sc, spec):
    return {"kind": "logit", "start": spec["start"], "c": spec["c"], "year": spec["year"],
            "log_loss": cond_log_loss(sc, oof(sc, spec)["p"])}


if __name__ == "__main__":
    cfg = load_config()
    data = ROOT / cfg["paths"]["data"]
    sc = load_scenarios(cfg)
    n_fights = int((sc.actual & sc.usable).sum())
    print(f"{n_fights} usable fights before 2025; {len(round_rows(sc[sc.actual & sc.usable], True))} round rows")

    base = {"format only": oof(sc, {"kind": "base", "start": 2010, "by": ["five"]})["p"],
            "format, sex and heavy divisions": None}
    sc["heavy_grp"] = (sc.lbs >= 205).astype(float)
    base["format, sex and heavy divisions"] = oof(sc, {"kind": "base", "start": 2010,
                                                       "by": ["five", "female", "heavy_grp"]})["p"]
    for k, d in base.items():
        print(f"baseline ({k}): {cond_log_loss(sc, d):.5f}")

    # year=True lets the model follow shifts over time: five-round fights have had far fewer
    # early finishes since about 2019 (see reports/finish_validation.md)
    lspecs = [{"kind": "logit", "start": s, "c": c, "year": y} for s in (2010, 2013)
              for c in (0.001, 0.003, 0.01, 0.03, 0.1) for y in (False, True)]
    rows = Parallel(n_jobs=6)(delayed(eval_logit)(sc, s) for s in lspecs)
    grows = Parallel(n_jobs=8, verbose=2)(delayed(eval_gbm)(sc, s) for s in gbm_grid(N_GBM, 31))
    tune = pd.concat([pd.DataFrame(rows), pd.DataFrame([r for rs in grows for r in rs])]).sort_values("log_loss")
    tune.to_csv(data / "finish_tuning.csv", index=False)
    bl = tune[tune.kind == "logit"].iloc[0]
    bg = tune[tune.kind == "gbm"].iloc[0]
    pcols = ["num_leaves", "max_depth", "learning_rate", "min_child_samples", "feature_fraction", "lambda_l2",
             "min_split_gain"]
    best = {"logit": {"kind": "logit", "start": int(bl.start), "c": float(bl.c), "year": bool(bl.year)},
            "gbm": {"kind": "gbm", "start": int(bg.start), "rounds": int(bg.rounds), "year": True,
                    "params": {k: (int(bg[k]) if k in ("num_leaves", "max_depth", "min_child_samples") else float(bg[k]))
                               for k in pcols}}}
    print("best logit:", best["logit"], round(bl.log_loss, 5))
    print("best gbm:", best["gbm"], round(bg.log_loss, 5))
    print(tune[tune.kind == "gbm"].head(5).to_string(index=False))

    D = {"logit": oof(sc, best["logit"])["p"]}
    gs = Parallel(n_jobs=3)(delayed(oof)(sc, dict(best["gbm"], seed=s)) for s in SEEDS)
    D["gbm"] = sum(g["p"] for g in gs) / len(SEEDS)
    D["blend"] = (D["logit"] + D["gbm"]) / 2
    D["base"] = base["format, sex and heavy divisions"]
    best["scores"] = {k: cond_log_loss(sc, d) for k, d in D.items()}
    best["scores"]["base_format_only"] = cond_log_loss(sc, base["format only"])
    print(json.dumps(best["scores"], indent=1))
    with open(data / "finish_best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=1)

    keep = D["gbm"].index
    out = sc.loc[keep, ["fight_id", "date", "w_side", "w_name", "l_name", "sched", "actual", "method", "end_round",
                        "five", "female", "lbs", "p_w"]].copy()
    out["logit_no_year_ko_1"] = oof(sc, dict(best["logit"], year=False))["p"].loc[keep, "ko_1"].values
    for k, d in D.items():
        for c in OUT_COLS:
            out[f"{k}_{c}"] = d.loc[keep, c].values
    out.to_csv(data / "finish_oof.csv", index=False, date_format="%Y-%m-%d")
    print("wrote", data / "finish_oof.csv", len(out), "scenario rows")
