"""Market baselines built from the odds alone."""
import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import log_loss


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def fit_stretch(p, y):
    """One number s so that sigmoid(s * logit(p)) fits best. s > 1 means favourites
    win more often than the price says. No intercept, so it is symmetric in the fighters."""
    lp = logit(p)
    res = minimize_scalar(lambda s: log_loss(y, sigmoid(s * lp)), bounds=(0.5, 2.0), method="bounded")
    return float(res.x)


def stretch(p, s):
    return sigmoid(s * logit(p))


def flat_return(y, dec_a, dec_b, pick_a):
    """Return per unit staked when betting one unit on A (pick_a True) or B at the given prices."""
    y = np.asarray(y, float)
    win = np.where(pick_a, y == 1, y == 0)
    dec = np.where(pick_a, dec_a, dec_b)
    return float(np.where(win, dec - 1, -1).mean())
