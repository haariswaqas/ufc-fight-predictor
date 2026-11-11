"""Tests for the method and round model's data handling and probability arithmetic."""
import numpy as np
import pandas as pd
import pytest

from ufc.finish import (X_COLS, X_YEAR, FinishGBM, FinishLogit, base_rate_distribution, distribution, round_rows,
                        scenarios)

OUT = [f"{m}_{r}" for m in ("ko", "sub") for r in range(1, 6)] + ["dec"]


@pytest.fixture(scope="module")
def sc(clean, built, cfg):
    cut = cfg["splits"]["test_start"]
    c = clean[clean.date < cut].reset_index(drop=True)
    pre = built[1]
    return scenarios(c, pre[pre.date < cut])


def test_scenarios_never_include_the_test_period(sc, cfg):
    assert sc.date.max() < pd.Timestamp(cfg["splits"]["test_start"])


def test_exactly_one_scenario_per_decided_fight_is_the_actual_one(sc, clean):
    per_fight = sc.groupby("fight_id").actual.sum()
    decided = set(clean.fight_id[clean.is_target])
    assert (per_fight[per_fight.index.isin(decided)] == 1).all()
    assert (per_fight[~per_fight.index.isin(decided)] == 0).all()
    assert (sc.groupby("fight_id").size() == 2).all()


def test_winner_and_loser_columns_come_from_the_right_fighter(sc, built, clean):
    pre = built[1].set_index(["fight_id", "side"])
    s = sc.sample(300, random_state=4)
    for r in s.itertuples():
        assert r.w_elo == pre.loc[(r.fight_id, r.w_side), "elo"]
        assert r.l_elo == pre.loc[(r.fight_id, 3 - r.w_side), "elo"]
        assert r.w_ko_win_rate == pre.loc[(r.fight_id, r.w_side), "ko_win_rate"]
        assert r.l_ko_loss_rate == pre.loc[(r.fight_id, 3 - r.w_side), "ko_loss_rate"]
    c = clean.set_index("fight_id")
    won = sc[sc.actual]
    names = np.where(c.loc[won.fight_id, "f1_won"].values == 1, c.loc[won.fight_id, "fighter_1"].values,
                     c.loc[won.fight_id, "fighter_2"].values)
    assert (won.w_name.values == names).all()


def test_market_probabilities_of_the_two_scenarios_add_to_one(sc):
    g = sc.dropna(subset=["p_w"]).groupby("fight_id")
    assert np.allclose(g.p_w.sum(), 1)
    assert np.allclose(sc.groupby("fight_id").elo_p_w.sum(), 1)
    assert not sc[["w_age", "l_age", "lbs"]].isna().any().any()


def test_training_round_rows_encode_the_result(sc):
    a = sc[sc.actual & sc.usable]
    rr = round_rows(a, training=True)
    n = rr.groupby(level=0).size()
    assert (n == a.end_round.astype(int)).all()
    finishes = rr.groupby(level=0).y.apply(lambda y: (y > 0).sum())
    assert (finishes[a.method != "DEC"] == 1).all() and (finishes[a.method == "DEC"] == 0).all()
    last = rr.groupby(level=0).tail(1)
    fin = a.method != "DEC"
    assert (last.y[fin.values].values == np.where(a.method[fin] == "KO", 1, 2)).all()
    assert (rr.rounds_left >= 0).all() and rr.rnd.between(1, 5).all()
    # a decision always runs the full scheduled distance
    assert (a.end_round[a.method == "DEC"] == a.sched[a.method == "DEC"]).all()


def test_prediction_rows_cover_every_scheduled_round(sc):
    u = sc[sc.usable].head(400)
    rr = round_rows(u, training=False)
    assert (rr.groupby(level=0).size() == u.sched.astype(int)).all()
    assert "y" not in rr.columns


@pytest.mark.parametrize("kind", ["gbm", "logit", "base"])
def test_distributions_are_proper_probabilities(sc, kind):
    tr = sc[sc.actual & sc.usable & (sc.date >= "2013-03-01") & (sc.date < "2019-01-01")]
    te = sc[sc.usable & (sc.date >= "2019-01-01") & (sc.date < "2019-07-01")]
    if kind == "gbm":
        m = FinishGBM({"num_leaves": 6, "learning_rate": 0.05, "min_child_samples": 100}, 60, cols=X_YEAR).fit(tr)
        d = distribution(te, m.hazards)
    elif kind == "logit":
        d = distribution(te, FinishLogit(0.01, cols=X_YEAR).fit(tr).hazards)
    else:
        d = base_rate_distribution(tr, te)
    assert list(d.columns) == OUT and (d.index == te.index).all()
    assert np.allclose(d.sum(axis=1), 1) and (d.values >= 0).all()
    three = (te.sched == 3).values
    late = ["ko_4", "ko_5", "sub_4", "sub_5"]
    assert (d.loc[three, late].values == 0).all()            # a three-round fight cannot end in round 4
    assert (d.loc[~three, late].values > 0).all()
    assert d.dec.between(0.02, 0.98).all()


def test_weights_and_year_input_are_accepted(sc):
    tr = sc[sc.actual & sc.usable & (sc.date >= "2015-01-01") & (sc.date < "2018-01-01")]
    w = np.linspace(0.5, 1.0, len(tr))
    m = FinishGBM({"num_leaves": 4, "learning_rate": 0.1}, 20, cols=X_COLS).fit(tr, w)
    assert m.hazards(round_rows(tr.head(50), training=False)).shape[1] == 3
    assert set(X_YEAR) - set(X_COLS) == {"year"}
