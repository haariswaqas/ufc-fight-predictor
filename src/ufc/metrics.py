"""Scoring: accuracy, log loss, Brier, calibration, and event-level bootstrap."""
import numpy as np
import pandas as pd

EPS = 1e-9


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())


def accuracy(y, p):
    """Share of fights where the higher-probability fighter won; exact 50/50 counts half."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1)).mean())


def summary(y, p):
    return {"n": int(len(y)), "accuracy": accuracy(y, p), "log_loss": log_loss(y, p),
            "brier": brier(y, p)}


def calibration_table(y, p, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0)):
    """Calibration from the favourite's side: predicted versus actual win rate per band."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    fav_p = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    band = pd.cut(fav_p, list(edges), include_lowest=True)
    t = pd.DataFrame({"band": band, "p": fav_p, "won": fav_won})
    return (t.groupby("band", observed=True)
             .agg(fights=("won", "size"), predicted=("p", "mean"), actual=("won", "mean"))
             .reset_index())


def event_bootstrap(dates, fn, reps=2000, seed=0):
    """Percentile interval for fn(row_index_array) when whole event dates are resampled."""
    dates = np.asarray(dates)
    uniq, inv = np.unique(dates, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        vals[r] = fn(np.concatenate([groups[i] for i in pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
