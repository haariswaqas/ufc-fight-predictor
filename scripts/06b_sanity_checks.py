"""Stage 2 sanity checks on the betting result. Uses 2019-2024 out-of-sample data only.

1. Placebo: give every fight the features of a random other fight from the same year.
   The pipeline must then earn nothing; a profit here would mean the code manufactures it.
2. Independent ledger: recompute the chosen rule's return from the raw odds file in its
   original fighter order, without the backtest module.
3. Ablation: how much of the result depends on the age terms.
4. Does a model built on the closing odds make money at closing prices?

Writes reports/stage2_sanity.md.
"""
import _path  # noqa: F401
import numpy as np
import pandas as pd

from ufc import ROOT, load_config
from ufc.backtest import ledger, summarise
from ufc.market import sigmoid
from ufc.metrics import log_loss
from ufc.models import FEATURE_SETS
from ufc.walk import load_dev, oof_logit, score

cfg = load_config()
data, reports = ROOT / cfg["paths"]["data"], ROOT / cfg["paths"]["reports"]
dev = load_dev(cfg)
SPEC = {"kind": "logit", "market": True, "start": 2015, "half_life": None, "lam": 0.03,
        "cols": FEATURE_SETS["all"]}
THR, CAP = 0.05, 0.20
out = ["# Stage 2 sanity checks\n",
       "All on 2019-2024 out-of-sample predictions of the B1 model (logistic regression on top of the "
       f"opening odds), betting rule: edge above {THR:.0%}, capped at {CAP:.0%}.\n"]


def run(d, spec=SPEC, price="open"):
    z = oof_logit(d, spec)["z"]
    m = d.loc[z.index]
    m = m[m.market_ok]
    s = summarise(ledger(m, z[m.index], THR, CAP, price=price), 1000, 1)
    return score(d, z, True), s


base_ll, base = run(dev)
rows = [{"run": "B1 as trained", "log_loss": base_ll, "bets": base["bets"], "return": base["return"],
         "low": base["return_low"], "high": base["return_high"], "clv": base["clv"]}]

# ---- 1 placebo: features shuffled across fights within a year (odds and results stay put)
rets, lls = [], []
for seed in range(20):
    rng = np.random.default_rng(seed)
    p = dev.copy()
    for yr, idx in p.groupby(p.date.dt.year).groups.items():
        idx = np.array(idx)
        sign = rng.choice([-1.0, 1.0], len(idx))[:, None]      # also flip sides at random
        p.loc[idx, FEATURE_SETS["all"]] = p.loc[rng.permutation(idx), FEATURE_SETS["all"]].values * sign
    ll, s = run(p)
    lls.append(ll)
    rets.append((s["bets"], s.get("return", np.nan)))
open_ll = log_loss(*(lambda m: (m.y, m.p_open))(dev[dev.is_target & dev.market_ok & (dev.date.dt.year >= 2019)]))
r = np.array(rets, float)
rows.append({"run": "Placebo: shuffled features, mean of 20", "log_loss": float(np.mean(lls)),
             "bets": int(np.nanmean(r[:, 0])), "return": float(np.nanmean(r[:, 1])),
             "low": float(np.nanmin(r[:, 1])), "high": float(np.nanmax(r[:, 1])), "clv": np.nan})

# ---- 3 ablation
no_age = [c for c in FEATURE_SETS["all"] if "age" not in c]
ll, s = run(dev, dict(SPEC, cols=no_age))
rows.append({"run": "Without any age feature", "log_loss": ll, "bets": s["bets"], "return": s["return"],
             "low": s["return_low"], "high": s["return_high"], "clv": s["clv"]})
only_age = [c for c in FEATURE_SETS["all"] if "age" in c]
ll, s = run(dev, dict(SPEC, cols=only_age))
rows.append({"run": "Age features only", "log_loss": ll, "bets": s["bets"], "return": s["return"],
             "low": s["return_low"], "high": s["return_high"], "clv": s["clv"]})

# ---- 4 closing odds as the base, bets at closing prices
cd = dev.assign(p_open=dev.p_close)
ll, s = run(cd, price="close")
rows.append({"run": "Same recipe on CLOSING odds, bets at closing prices", "log_loss": ll, "bets": s["bets"],
             "return": s["return"], "low": s["return_low"], "high": s["return_high"], "clv": s["clv"]})
t = pd.DataFrame(rows)

# ---- 2 independent ledger from the raw file, original fighter order
z = oof_logit(dev, SPEC)["z"]
m = dev.loc[z.index]
m = m[m.market_ok].assign(z=z)
raw = pd.read_csv(ROOT / cfg["paths"]["enriched"])
raw["fight_id"] = raw.fight_url.str.extract(r"fight-details/(\w+)")[0]
j = m.merge(raw[["fight_id", "fighter_1", "fighter_2", "winner", "result", "f1_open_decimal",
                 "f2_open_decimal"]], on="fight_id", suffixes=("", "_raw"))
p1 = np.where(j.a_name == j.fighter_1, sigmoid(j.z), 1 - sigmoid(j.z))      # P(raw fighter_1 wins)
e1, e2 = p1 * j.f1_open_decimal - 1, (1 - p1) * j.f2_open_decimal - 1
pick1 = e1 >= e2
edge = np.where(pick1, e1, e2)
bet = (edge > THR) & (edge <= CAP)
price = np.where(pick1, j.f1_open_decimal, j.f2_open_decimal)
picked = np.where(pick1, j.fighter_1, j.fighter_2)
profit = np.where(j.result_raw != "win", 0.0, np.where(picked == j.winner, price - 1, -1.0))
indep = (int(bet.sum()), float(profit[bet].mean()))

for c in t.columns:
    if t[c].dtype.kind == "f":
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
out += ["| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|",
        *["| " + " | ".join(map(str, r)) + " |" for r in t.values], ""]
out.append(f"- Opening odds log loss on the same fights: {open_ll:.4f}. In the placebo `low`/`high` are the "
           "smallest and largest of the 20 runs.")
out.append(f"- Independent recomputation of the chosen rule from the raw odds file: {indep[0]} bets, "
           f"return {indep[1]:+.4f} (backtest module: {base['bets']} bets, {base['return']:+.4f}).")
(reports / "stage2_sanity.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
