"""How and when a fight ends, given who wins.

For a fight and an assumed winner, the model walks through the rounds. In each round one
of three things happens: the winner finishes by knockout, finishes by submission, or the
fight continues. If it continues through the last scheduled round, the win is a decision.

    P(winner by KO in round r) = P(winner) x P(no finish before r | winner) x P(KO in r | reached r, winner)

P(winner) comes from the separate winner model (or the odds), so the two always agree.
Everything here is conditional on the winner, so no fighter-order symmetry is needed.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

W_FEATS = ["ko_win_rate", "sub_win_rate", "finish_rate", "ko_w_p15", "sub_w_p15", "early_fin_w_rate",
           "r1_fin_w_rate", "avg_fin_round_w", "dec_rate", "avg_fight_min", "c_slpm", "r_slpm", "c_sig_acc",
           "c_kd_p15", "r_kd_p15", "c_td_p15", "c_sub_p15", "r_sub_p15", "c_ctrl_share", "att_pm", "r_att_pm",
           "r1_att_pm", "late_att_pm", "fade", "age", "n_fights", "five_round_fights", "late_minutes", "elo",
           "win_rate", "r3_win_rate", "debut"]
L_FEATS = ["ko_loss_rate", "sub_loss_rate", "ko_l_p15", "sub_l_p15", "early_fin_l_rate", "c_kda_p15",
           "r_kda_p15", "c_sapm", "r_sapm", "c_sig_def", "c_td_def", "c_tda_p15", "c_ctrla_share", "dec_rate",
           "avg_fight_min", "oatt_pm", "r1_abs_pm", "age", "n_fights", "five_round_fights", "late_minutes",
           "elo", "win_rate", "debut"]
CONTEXT = ["lbs", "female", "five", "ufc_title", "p_w", "elo_p_w"]
ROUND = ["rnd", "rounds_left"]
X_COLS = ["w_" + c for c in W_FEATS] + ["l_" + c for c in L_FEATS] + CONTEXT + ROUND
# the same inputs plus the calendar year, so the model can follow shifts over time
# (five-round fights have had far fewer early finishes since about 2019)
X_YEAR = X_COLS + ["year"]


def recency(dates, end, half_life_years):
    """Sample weights that halve every `half_life_years` before `end` (None: equal weights)."""
    if not half_life_years:
        return None
    age = (pd.Timestamp(end) - pd.to_datetime(dates)).dt.days.values / 365.25
    return 0.5 ** (np.maximum(age, 0) / half_life_years)
METHODS = ("KO", "SUB", "DEC")


def scenarios(clean, pre):
    """Two rows per fight: one assuming fighter 1 wins, one assuming fighter 2 wins."""
    p = {s: pre[pre.side == s].set_index("fight_id").loc[clean.fight_id] for s in (1, 2)}
    out = []
    for w, lo in ((1, 2), (2, 1)):
        d = pd.DataFrame({"fight_id": clean.fight_id.values, "date": clean.date.values, "w_side": w,
                          "w_name": clean[f"fighter_{w}"].values, "l_name": clean[f"fighter_{lo}"].values})
        for c in W_FEATS:
            d["w_" + c] = p[w][c].values
        for c in L_FEATS:
            d["l_" + c] = p[lo][c].values
        # profile ages are missing mostly for fighters who left early: never let that show
        for c in ("w_age", "l_age"):
            d[c] = d[c].fillna(30.0)
        d["lbs"] = clean.lbs.fillna(170.0).values
        d["female"] = clean.female.astype(float).values
        d["five"] = (clean.sched_rounds == 5).astype(float).values
        d["ufc_title"] = clean.ufc_title.astype(float).values
        p1 = np.where(clean.market_ok, clean.f1_p_open, np.nan)
        d["p_w"] = p1 if w == 1 else 1 - p1
        d["elo_p_w"] = 1 / (1 + 10 ** ((p[lo].elo.values - p[w].elo.values) / 400))
        d["sched"] = clean.sched_rounds.values
        d["year"] = clean.date.dt.year.values + (clean.date.dt.day_of_year.values - 1) / 365.25
        won = clean.f1_won.values == (1.0 if w == 1 else 0.0)
        d["actual"] = clean.is_target.values & won
        d["method"] = clean.method_group.values
        d["end_round"] = np.where(clean.method_group.values == "DEC", clean.sched_rounds.values, clean["round"].values)
        d["usable"] = (clean.is_target & clean.method_group.isin(METHODS) & clean.sched_rounds.isin([3, 5])).values
        out.append(d)
    return pd.concat(out, ignore_index=True)


def round_rows(sc, training):
    """Expand scenarios to one row per round.
    training=True: only rounds that happened, with the outcome (0 continue, 1 KO, 2 submission).
    training=False: every scheduled round, for prediction."""
    last = sc.end_round if training else sc.sched
    reps = last.astype(int).values
    rr = sc.loc[sc.index.repeat(reps)].copy()
    rr["rnd"] = np.concatenate([np.arange(1, n + 1) for n in reps]).astype(float)
    rr["rounds_left"] = rr.sched - rr.rnd
    if training:
        ended = (rr.rnd == rr.end_round) & rr.method.isin(["KO", "SUB"])
        rr["y"] = np.where(ended & (rr.method == "KO"), 1, np.where(ended & (rr.method == "SUB"), 2, 0))
    return rr


class FinishGBM:
    """LightGBM over round rows: continue / knockout / submission."""

    def __init__(self, params, rounds, seed=0, cols=None):
        self.params, self.rounds, self.seed = dict(params), rounds, seed
        self.cols = list(cols or X_COLS)

    def fit(self, sc, weights=None):
        """weights: optional per-fight weights aligned with sc."""
        rr = round_rows(sc.assign(_w=1.0 if weights is None else weights), training=True)
        p = {"objective": "multiclass", "num_class": 3, "verbosity": -1, "seed": self.seed, "num_threads": 1,
             "deterministic": True, "force_row_wise": True, **self.params}
        self.booster_ = lgb.train(p, lgb.Dataset(rr[self.cols].values.astype(float), rr.y.values,
                                                 weight=rr._w.values), num_boost_round=self.rounds)
        return self

    def hazards(self, rr, rounds=None):
        return self.booster_.predict(rr[self.cols].values.astype(float), num_iteration=rounds or self.rounds)


class FinishLogit:
    """Multinomial logistic regression on the same rows (the simple comparison model)."""

    def __init__(self, c, cols=None):
        self.c = c
        self.cols = list(cols or X_COLS)

    def _x(self, rr):
        x = rr[self.cols].copy()
        x["p_w"] = x.p_w.fillna(x.elo_p_w)
        x = x.fillna(self.med_)
        for r in (1, 2, 3, 4, 5):                    # each round gets its own level, by format
            x[f"is_r{r}"] = (x.rnd == r).astype(float)
            x[f"is_r{r}_five"] = x[f"is_r{r}"] * x.five
        return x.values.astype(float)

    def fit(self, sc, weights=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        rr = round_rows(sc.assign(_w=1.0 if weights is None else weights), training=True)
        self.med_ = rr[self.cols].median()
        if "year" in self.cols:
            self.year0_ = float(rr.year.mean())
        x = self._with_trend(rr, self._x(rr))
        self.scaler_ = StandardScaler().fit(x)
        self.model_ = LogisticRegression(C=self.c, max_iter=3000).fit(
            self.scaler_.transform(x), rr.y.values, sample_weight=rr._w.values)
        return self

    def _with_trend(self, rr, x):
        """Let the effect of the five-round format, and of round 1, drift with time."""
        if "year" not in self.cols:
            return x
        t = (rr.year.values - self.year0_)[:, None]
        five, r1 = rr.five.values[:, None], (rr.rnd.values == 1)[:, None]
        return np.hstack([x, t * five, t * five * r1, t * r1])

    def hazards(self, rr, rounds=None):
        return self.model_.predict_proba(self.scaler_.transform(self._with_trend(rr, self._x(rr))))


def distribution(sc, hazards_fn):
    """Outcome probabilities for each scenario, conditional on that scenario's winner.
    Returns a DataFrame indexed like sc with columns ko_1..ko_5, sub_1..sub_5, dec."""
    rr = round_rows(sc, training=False)
    h = hazards_fn(rr)
    out = pd.DataFrame(0.0, index=sc.index, columns=[f"{m}_{r}" for m in ("ko", "sub") for r in range(1, 6)] + ["dec"])
    alive = pd.Series(1.0, index=sc.index)
    idx = rr.index.values
    for r in range(1, 6):
        m = rr.rnd.values == r
        rows = idx[m]
        a = alive.loc[rows].values
        out.loc[rows, f"ko_{r}"] = a * h[m, 1]
        out.loc[rows, f"sub_{r}"] = a * h[m, 2]
        alive.loc[rows] = a * h[m, 0]
    out["dec"] = alive
    return out


def outcome_label(method, end_round):
    return "dec" if method == "DEC" else f"{method.lower()}_{int(end_round)}"


def base_rate_distribution(train_sc, sc, by=("five",)):
    """Baseline: round-by-round finish rates counted in the training fights, by format."""
    rr = round_rows(train_sc[train_sc.actual & train_sc.usable], training=True)
    keys = list(by) + ["rnd"]
    tab = rr.groupby(keys).y.value_counts(normalize=True).unstack(fill_value=0.0).reindex(columns=[0, 1, 2], fill_value=0.0)

    def fn(pr):
        k = pd.MultiIndex.from_frame(pr[keys])
        return tab.reindex(k).fillna({0: 1.0, 1: 0.0, 2: 0.0}).values
    return distribution(sc, fn)
