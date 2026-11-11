"""Models for Stage 2. Every model is symmetric: P(A beats B) = 1 - P(B beats A).

A models use fight features only. B models start from the opening odds and learn a
correction. Logistic models are symmetric by construction (difference features, no
intercept). Tree models are trained on both orders of every fight and their two
predictions are averaged in log-odds.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .features import FEATURES
from .market import logit, sigmoid

# extra antisymmetric terms (A minus B) that let a linear model bend
EXTRA_ANTI = ["d_age_over32", "d_age_over35", "d_long_layoff", "d_few_fights", "d_age_x_exp"]
ANTI = FEATURES + EXTRA_ANTI
# symmetric context for tree models: the same whichever fighter is called A
SYM = ["m_age", "max_age", "min_n_fights", "m_log_n_fights", "m_elo", "m_slpm", "m_td_p15",
       "m_finish_rate", "lbs", "female", "five_round"]
FEATURE_SETS = {
    "all": ANTI,
    "no_recent": [c for c in ANTI if not c.startswith("d_r_")],
    "compact": ["d_elo", "d_win_rate", "d_recent_win_rate", "d_age", "d_age_over32", "d_age_over35",
                "d_debut", "d_log_n_fights", "d_days_since", "d_long_layoff", "d_reach_cm",
                "d_reach_x_heavy", "d_c_slpm", "d_c_sapm", "d_c_sig_acc", "d_c_sig_def", "d_c_td_p15",
                "d_c_td_def", "d_c_ctrl_share", "d_c_ctrla_share", "d_c_kda_p15", "d_ko_loss_rate",
                "d_finish_rate", "d_avg_opp_elo", "d_streak"],
}


def design(f, pre):
    """Add the extra antisymmetric terms and the symmetric context to the feature table."""
    p1 = pre[pre.side == 1].set_index("fight_id").loc[f.fight_id]
    p2 = pre[pre.side == 2].set_index("fight_id").loc[f.fight_id]
    sw = f.swapped.values

    def ab(col):
        return np.where(sw, p2[col].values, p1[col].values), np.where(sw, p1[col].values, p2[col].values)

    out = f.copy()
    age_a, age_b = ab("age")
    both_age = ~np.isnan(age_a) & ~np.isnan(age_b)
    for cut in (32, 35):
        d = np.maximum(age_a - cut, 0) - np.maximum(age_b - cut, 0)
        out[f"d_age_over{cut}"] = np.where(both_age, d, 0.0)
    lay_a, lay_b = ab("days_since")
    out["d_long_layoff"] = (np.nan_to_num(lay_a) > 450).astype(float) - (np.nan_to_num(lay_b) > 450)
    n_a, n_b = ab("n_fights")
    out["d_few_fights"] = (n_a < 3).astype(float) - (n_b < 3)
    # age matters more for the more experienced (older-in-the-cage) fighter
    out["d_age_x_exp"] = np.where(both_age, age_a * np.log1p(n_a) - age_b * np.log1p(n_b), 0.0) / 10

    def mean(col):
        a, b = ab(col)
        return np.nanmean(np.vstack([a, b]), axis=0) if not np.isnan(np.vstack([a, b])).all() else a

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["m_age"] = np.where(both_age, (age_a + age_b) / 2, np.nan)
            out["max_age"] = np.where(both_age, np.maximum(age_a, age_b), np.nan)
            out["min_n_fights"] = np.minimum(n_a, n_b)
            out["m_log_n_fights"] = (np.log1p(n_a) + np.log1p(n_b)) / 2
            for name, col in (("m_elo", "elo"), ("m_slpm", "c_slpm"), ("m_td_p15", "c_td_p15"),
                              ("m_finish_rate", "finish_rate")):
                out[name] = mean(col)
    for col in ("female", "five_round"):
        out[col] = out[col].astype(float)
    return out


# ---------------------------------------------------------------- logistic regression
class SymLogit:
    """L2-penalised logistic regression with no intercept, optional market log-odds term
    (its slope is not penalised) and sample weights. Inputs are clipped and scaled using
    the training rows only."""

    def __init__(self, cols, lam, market=False, clip_q=0.995):
        self.cols, self.lam, self.market, self.clip_q = list(cols), lam, market, clip_q

    def _x(self, df):
        x = np.clip(df[self.cols].values.astype(float), -self.cap_, self.cap_) / self.scale_
        return x

    def fit(self, df, y, w=None):
        raw = df[self.cols].values.astype(float)
        self.cap_ = np.maximum(np.quantile(np.abs(raw), self.clip_q, axis=0), 1e-9)
        self.scale_ = np.maximum(np.sqrt((np.clip(raw, -self.cap_, self.cap_) ** 2).mean(axis=0)), 1e-9)
        x = self._x(df)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        w = w / w.sum()
        m = logit(df.p_open.values) if self.market else None
        k = x.shape[1]

        def loss(theta):
            z = x @ theta[:k] + (theta[k] * m if self.market else 0)
            p = sigmoid(z)
            ll = -(w * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))).sum()
            g = w * (p - y)
            grad = np.concatenate([x.T @ g + 2 * self.lam * theta[:k],
                                   [g @ m] if self.market else []])
            return ll + self.lam * theta[:k] @ theta[:k], grad

        theta0 = np.concatenate([np.zeros(k), [1.0] if self.market else []])
        res = minimize(loss, theta0, jac=True, method="L-BFGS-B")
        self.coef_ = res.x[:k]
        self.market_slope_ = float(res.x[k]) if self.market else None
        return self

    def decision(self, df):
        z = self._x(df) @ self.coef_
        if self.market:
            z = z + self.market_slope_ * logit(df.p_open.values)
        return z

    def predict(self, df):
        return sigmoid(self.decision(df))


# ---------------------------------------------------------------- gradient boosting
class SymGBM:
    """LightGBM trained on both orders of each fight. With market=True the trees start
    from the (stretched) opening log-odds and learn only the correction."""

    def __init__(self, params, rounds, market=False, anti=ANTI, sym=SYM, seed=0):
        self.params, self.rounds, self.market = dict(params), rounds, market
        self.anti, self.sym, self.seed = list(anti), list(sym), seed

    def _xy(self, df, flip=False):
        a = df[self.anti].values.astype(float)
        return np.hstack([-a if flip else a, df[self.sym].values.astype(float)])

    def _fill(self, x):
        k = len(self.anti)
        s = x[:, k:]
        x[:, k:] = np.where(np.isnan(s), self.fill_, s)
        return x

    def fit(self, df, y, w=None, market_slope=1.0):
        y = np.asarray(y, float)
        self.market_slope_ = market_slope
        self.fill_ = np.nanmedian(df[self.sym].values.astype(float), axis=0)
        x = self._fill(np.vstack([self._xy(df), self._xy(df, flip=True)]))
        yy = np.concatenate([y, 1 - y])
        ww = None if w is None else np.concatenate([w, w])
        init = None
        if self.market:
            z = market_slope * logit(df.p_open.values)
            init = np.concatenate([z, -z])
        p = {"objective": "binary", "verbosity": -1, "seed": self.seed, "num_threads": 1,
             "deterministic": True, "force_row_wise": True, "bagging_fraction": 1.0, **self.params}
        self.booster_ = lgb.train(p, lgb.Dataset(x, yy, weight=ww, init_score=init, free_raw_data=False),
                                  num_boost_round=self.rounds)
        return self

    def decision(self, df, rounds=None):
        r = rounds or self.rounds
        za = self.booster_.predict(self._fill(self._xy(df)), raw_score=True, num_iteration=r)
        zb = self.booster_.predict(self._fill(self._xy(df, flip=True)), raw_score=True, num_iteration=r)
        z = (za - zb) / 2
        if self.market:
            z = z + self.market_slope_ * logit(df.p_open.values)
        return z

    def predict(self, df, rounds=None):
        return sigmoid(self.decision(df, rounds))


def recency_weights(dates, end, half_life_years):
    if not half_life_years:
        return None
    age = (pd.Timestamp(end) - pd.to_datetime(dates)).dt.days.values / 365.25
    return 0.5 ** (np.maximum(age, 0) / half_life_years)
