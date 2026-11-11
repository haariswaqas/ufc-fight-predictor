"""Walk-forward evaluation: train on everything before year Y, predict year Y.

Nothing in the test period is ever loaded: load_dev() cuts the table at test_start.
"""
import numpy as np
import pandas as pd

from . import ROOT
from .market import fit_stretch, sigmoid
from .metrics import log_loss
from .models import SymGBM, SymLogit, design, recency_weights

OOF_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def load_dev(cfg):
    """Feature table with model inputs, restricted to dates before the test period."""
    data = ROOT / cfg["paths"]["data"]
    f = pd.read_csv(data / "features.csv", parse_dates=["date"])
    pre = pd.read_csv(data / "fighter_prefight.csv", parse_dates=["date"])
    f = design(f, pre)
    dev = f[f.date < cfg["splits"]["test_start"]].reset_index(drop=True)
    assert dev.date.max() < pd.Timestamp(cfg["splits"]["test_start"])
    assert not dev.split.isin(["test", "after_test"]).any()
    return dev


def fold_rows(dev, year, start, market):
    """(training rows, rows to predict) for one walk-forward year."""
    tr = dev.is_target & (dev.date >= str(start)) & (dev.date < f"{year}-01-01")
    te = dev.date.dt.year == year
    if market:
        tr, te = tr & dev.market_ok, te & dev.has_market
    return dev[tr], dev[te]


def oof_logit(dev, spec, years=OOF_YEARS):
    """Out-of-sample log-odds for every fight in `years` under a model spec (dict).

    spec keys: kind ("logit" | "gbm"), market (bool), start (year), half_life (years or None),
    then for logit: cols, lam;  for gbm: params, rounds (int or list of checkpoints).
    Returns a DataFrame indexed like dev with one column per checkpoint (or "z")."""
    out = {}
    for year in years:
        tr, te = fold_rows(dev, year, spec["start"], spec["market"])
        w = recency_weights(tr.date, f"{year}-01-01", spec.get("half_life"))
        if spec["kind"] == "logit":
            m = SymLogit(spec["cols"], spec["lam"], market=spec["market"]).fit(tr, tr.y, w)
            out.setdefault("z", []).append(pd.Series(m.decision(te), index=te.index))
        else:
            rounds = spec["rounds"] if isinstance(spec["rounds"], (list, tuple)) else [spec["rounds"]]
            slope = fit_stretch(tr.p_open, tr.y) if spec["market"] else 1.0
            m = SymGBM(spec["params"], max(rounds), market=spec["market"],
                       seed=spec.get("seed", 0)).fit(tr, tr.y, w, market_slope=slope)
            for r in rounds:
                key = "z" if len(rounds) == 1 else r
                out.setdefault(key, []).append(pd.Series(m.decision(te, r), index=te.index))
    return pd.DataFrame({k: pd.concat(v) for k, v in out.items()})


def score(dev, z, market):
    """Log loss of out-of-sample log-odds on decided fights (market models: usable odds only)."""
    rows = dev.loc[z.index]
    keep = rows.is_target & (rows.market_ok if market else True)
    return log_loss(rows.y[keep], sigmoid(z[keep]))


def stretch_walk_forward(dev, years=OOF_YEARS, start=2013):
    """Opening odds stretched by a slope fitted only on earlier years."""
    parts = []
    for year in years:
        tr, te = fold_rows(dev, year, start, True)
        s = fit_stretch(tr.p_open, tr.y)
        lo = np.log(te.p_open / (1 - te.p_open))
        parts.append(pd.Series(s * lo, index=te.index))
    return pd.concat(parts)
