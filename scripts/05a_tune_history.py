"""Stage 2a: tune the history settings (Elo K, recency half-life, shrinkage strength).

Each candidate rebuilds the feature table and is scored by a fixed logistic model in
walk-forward on 2019-2024. The test period is cut off before anything is fitted.
Writes data/tuning_history.csv. It does not change config.yaml.
"""
import copy
import itertools

import _path  # noqa: F401
import pandas as pd
from joblib import Parallel, delayed

from ufc import ROOT, load_config
from ufc.clean import load_clean
from ufc.features import build_all
from ufc.models import FEATURE_SETS, design
from ufc.walk import oof_logit, score

cfg = load_config()
clean = load_clean(cfg)
clean = clean[clean.date < cfg["splits"]["test_start"]].reset_index(drop=True)   # no test rows at all

GRID = list(itertools.product([20, 28, 40, 56], [0.0, 1.0, 2.0], [2.0, 3.0, 5.0], [450.0, 900.0, 1800.0]))


def run(k, boost, half_life, k_seconds):
    c = copy.deepcopy(cfg)
    c["elo"].update(k=k, new_boost=boost)
    c["history"].update(recent_half_life_fights=half_life, k_seconds=k_seconds)
    _, pre, f = build_all(clean, c)
    dev = design(f, pre)
    res = {"elo_k": k, "elo_new_boost": boost, "recent_half_life_fights": half_life, "k_seconds": k_seconds}
    for name, lam in (("lam_0.003", 0.003), ("lam_0.01", 0.01)):
        z = oof_logit(dev, {"kind": "logit", "market": False, "start": 2010, "half_life": None,
                            "cols": FEATURE_SETS["all"], "lam": lam})["z"]
        res[name] = score(dev, z, market=False)
    return res


if __name__ == "__main__":
    rows = Parallel(n_jobs=6, verbose=5)(delayed(run)(*g) for g in GRID)
    t = pd.DataFrame(rows)
    t["log_loss"] = t[["lam_0.003", "lam_0.01"]].min(axis=1)
    t = t.sort_values("log_loss")
    t.to_csv(ROOT / cfg["paths"]["data"] / "tuning_history.csv", index=False)
    print(t.head(12).to_string(index=False))
    cur = t[(t.elo_k == 28) & (t.elo_new_boost == 1) & (t.recent_half_life_fights == 3) & (t.k_seconds == 900)]
    print("current config:", cur.log_loss.round(5).tolist(), " spread:", round(t.log_loss.min(), 5), "to",
          round(t.log_loss.max(), 5))
    for col in ("elo_k", "elo_new_boost", "recent_half_life_fights", "k_seconds"):
        print(t.groupby(col).log_loss.mean().round(5).to_dict())
