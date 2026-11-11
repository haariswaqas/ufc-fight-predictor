"""Stage 2: tune and train the four models by walk-forward on 2019-2024.

A1 logistic regression, fight features only      A2 gradient boosting, fight features only
B1 logistic regression on top of opening odds    B2 gradient boosting on top of opening odds

Every candidate is trained on fights before year Y and scored on year Y, for Y = 2019..2024.
The test period (2025 on) is cut off in load_dev() and never read here.

Writes data/tuning_logit.csv, data/tuning_gbm.csv, data/stage2_best_params.json,
data/oof_predictions.csv, data/b1_coefficients.csv.
"""
import itertools
import json

import _path  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ufc import ROOT, load_config
from ufc.market import fit_stretch, sigmoid
from ufc.models import FEATURE_SETS, SymLogit, recency_weights
from ufc.walk import OOF_YEARS, load_dev, oof_logit, score, stretch_walk_forward

CHECKPOINTS = [50, 100, 200, 400, 800, 1200]
N_GBM = 60
SEEDS = [0, 1, 2, 3, 4]


def logit_grid(market):
    starts = [2013, 2015] if market else [2005, 2010, 2013]
    for lam, start, hl, fs in itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3], starts,
                                                [None, 8.0, 4.0], list(FEATURE_SETS)):
        yield {"kind": "logit", "market": market, "start": start, "half_life": hl, "lam": lam,
               "feature_set": fs, "cols": FEATURE_SETS[fs]}


def gbm_grid(market, n, seed):
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    for _ in range(n):
        yield {"kind": "gbm", "market": market, "start": 2013 if market else pick([2010, 2013]),
               "half_life": pick([None, 8.0]), "rounds": CHECKPOINTS,
               "params": {"num_leaves": pick([3, 4, 6, 8, 12, 16]), "max_depth": pick([-1, 3, 4]),
                          "learning_rate": pick([0.005, 0.01, 0.02, 0.04]),
                          "min_child_samples": pick([40, 80, 150, 300]),
                          "feature_fraction": pick([0.4, 0.6, 0.8, 1.0]),
                          "lambda_l2": pick([0.0, 5.0, 20.0, 60.0]),
                          "min_split_gain": pick([0.0, 0.01])}}


def eval_logit(dev, spec):
    z = oof_logit(dev, spec)["z"]
    row = {k: spec[k] for k in ("market", "start", "half_life", "lam", "feature_set")}
    row["log_loss"] = score(dev, z, spec["market"])
    return row


def eval_gbm(dev, spec):
    z = oof_logit(dev, spec)
    rows = []
    for r in CHECKPOINTS:
        rows.append({"market": spec["market"], "start": spec["start"], "half_life": spec["half_life"],
                     "rounds": r, **spec["params"], "log_loss": score(dev, z[r], spec["market"])})
    return rows


def calibrated_walk_forward(dev, z, market):
    """Rescale log-odds by a slope fitted only on out-of-sample predictions from earlier years."""
    rows = dev.loc[z.index]
    out = z.copy()
    slopes = {}
    for year in OOF_YEARS[1:]:
        past = (rows.date.dt.year < year) & rows.is_target & (rows.market_ok if market else True)
        s = fit_stretch(sigmoid(z[past]), rows.y[past])
        slopes[year] = s
        now = rows.date.dt.year == year
        out[now] = z[now] * s
    return out, slopes


if __name__ == "__main__":
    cfg = load_config()
    data = ROOT / cfg["paths"]["data"]
    dev = load_dev(cfg)
    print(f"development data: {len(dev)} fights, {dev.date.min():%Y-%m-%d} to {dev.date.max():%Y-%m-%d}")

    # ------------------------------------------------------------ logistic models
    specs = list(logit_grid(False)) + list(logit_grid(True))
    rows = Parallel(n_jobs=8)(delayed(eval_logit)(dev, s) for s in specs)
    tl = pd.DataFrame(rows).sort_values("log_loss")
    tl.to_csv(data / "tuning_logit.csv", index=False)
    best = {}
    for market, name in ((False, "A1"), (True, "B1")):
        b = tl[tl.market == market].iloc[0]
        best[name] = {"kind": "logit", "market": market, "start": int(b.start),
                      "half_life": None if pd.isna(b.half_life) else float(b.half_life),
                      "lam": float(b.lam), "feature_set": b.feature_set}
        print(name, "best:", best[name], "log loss", round(b.log_loss, 5))
        print(tl[tl.market == market].head(5).to_string(index=False))

    # ------------------------------------------------------------ boosted models
    gspecs = list(gbm_grid(False, N_GBM, 11)) + list(gbm_grid(True, N_GBM, 12))
    rows = Parallel(n_jobs=8, verbose=2)(delayed(eval_gbm)(dev, s) for s in gspecs)
    tg = pd.DataFrame([r for rs in rows for r in rs]).sort_values("log_loss")
    tg.to_csv(data / "tuning_gbm.csv", index=False)
    pcols = ["num_leaves", "max_depth", "learning_rate", "min_child_samples", "feature_fraction",
             "lambda_l2", "min_split_gain"]
    for market, name in ((False, "A2"), (True, "B2")):
        b = tg[tg.market == market].iloc[0]
        best[name] = {"kind": "gbm", "market": market, "start": int(b.start),
                      "half_life": None if pd.isna(b.half_life) else float(b.half_life),
                      "rounds": int(b.rounds),
                      "params": {k: (int(b[k]) if k in ("num_leaves", "max_depth", "min_child_samples")
                                     else float(b[k])) for k in pcols}}
        print(name, "best:", best[name], "log loss", round(b.log_loss, 5))
        print(tg[tg.market == market].head(5).to_string(index=False))

    # ------------------------------------------------------------ final out-of-sample predictions
    Z = {}
    for name in ("A1", "B1"):
        s = dict(best[name], cols=FEATURE_SETS[best[name]["feature_set"]])
        Z[name] = oof_logit(dev, s)["z"]
    for name in ("A2", "B2"):       # average five seeds to steady the tree models
        zs = Parallel(n_jobs=5)(delayed(oof_logit)(dev, dict(best[name], seed=sd)) for sd in SEEDS)
        Z[name] = sum(z["z"] for z in zs) / len(SEEDS)
    Z["Aens"] = (Z["A1"] + Z["A2"]) / 2
    Z["Bens"] = (Z["B1"] + Z["B2"]) / 2
    Z["stretch"] = stretch_walk_forward(dev)
    # diagnostic only: the B1 recipe on top of the CLOSING odds. If fight features improved
    # much on the close, that would point to a leak rather than to skill.
    close_dev = dev.assign(p_open=dev.p_close)
    Z["close_plus"] = oof_logit(close_dev, dict(best["B1"], cols=FEATURE_SETS[best["B1"]["feature_set"]]))["z"]

    slopes = {}
    for name in ("A1", "A2", "Aens", "B1", "B2", "Bens"):
        Z[name + "_cal"], slopes[name] = calibrated_walk_forward(dev, Z[name], name.startswith("B"))

    keep = dev.date.dt.year.isin(OOF_YEARS)
    cols = ["fight_id", "date", "a_name", "b_name", "result", "is_target", "y", "has_market", "market_ok",
            "odds_conflict", "open_suspect", "p_open", "p_close", "a_open_decimal", "b_open_decimal",
            "a_close_decimal", "b_close_decimal", "p_elo"]
    oof = dev.loc[keep, cols].copy()
    for name, z in Z.items():
        oof["z_" + name] = z.reindex(oof.index)
    oof.to_csv(data / "oof_predictions.csv", index=False, date_format="%Y-%m-%d")

    # ------------------------------------------------------------ what B1 learned
    s = best["B1"]
    tr = dev[dev.is_target & dev.market_ok & (dev.date >= str(s["start"]))]
    m = SymLogit(FEATURE_SETS[s["feature_set"]], s["lam"], market=True).fit(
        tr, tr.y, recency_weights(tr.date, "2025-01-01", s["half_life"]))
    coef = pd.DataFrame({"feature": m.cols, "coefficient_per_sd": m.coef_}).sort_values(
        "coefficient_per_sd", key=np.abs, ascending=False)
    coef.to_csv(data / "b1_coefficients.csv", index=False)
    best["B1"]["market_slope_full_fit"] = m.market_slope_

    best["calibration_slopes"] = slopes
    best["scores"] = {n: score(dev, Z[n], n.startswith("B") or n in ("stretch", "close_plus"))
                      for n in Z}
    with open(data / "stage2_best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=1)
    print(json.dumps(best["scores"], indent=1))
    print("B1 market slope:", round(m.market_slope_, 3))
    print(coef.head(12).to_string(index=False))
