"""Leak and correctness tests for the pace and timing features (method and round models)."""
import hashlib
import json

import numpy as np
import pandas as pd

from ufc import ROOT
from ufc.clean import ROUND_COLS, STAT_COLS
from ufc.features import FEATURES, build_all
from ufc.history import PACE

RAWP = ["rawp_tsecs", "rawp_r1_att", "rawp_r1_secs", "rawp_late_att", "rawp_late_secs", "rawp_early_fin_w",
        "rawp_n5", "rawp_reach3_w"]


def _pre(p, ids):
    return p.set_index(["fight_id", "side"]).loc[ids, PACE + RAWP]


def test_winner_feature_table_is_still_the_frozen_one():
    """Adding pace features must not move a single byte of the winner model's inputs."""
    frozen = json.load(open(ROOT / "config_frozen.json", encoding="utf-8"))
    sha = hashlib.sha256((ROOT / "data/features.csv").read_bytes()).hexdigest()
    assert sha == frozen["features_sha256"]
    assert not set(PACE) & {c[2:] for c in FEATURES}


def test_pace_features_have_no_gaps_in_the_modelling_era(built):
    p = built[1]
    p = p[p.date >= "2010-01-01"]
    # round-by-round data starts in February 2013, so the round-based values are blank before
    # the first debutants with round data have fought; everything else is complete from 2010
    round_based = ["r1_att_pm", "r_r1_att_pm", "r1_abs_pm", "late_att_pm", "fade"]
    assert not p[[c for c in PACE if c not in round_based]].isna().any().any()
    assert not p.loc[p.date >= "2014-01-01", round_based].isna().any().any()
    first = p.loc[p.r1_att_pm.notna(), "date"].min()
    assert pd.Timestamp("2013-02-01") < first < pd.Timestamp("2013-06-01")
    p = p[p.date >= "2014-01-01"]
    assert p.att_pm.between(1, 40).all() and p.r1_att_pm.between(1, 40).all()
    assert p.dec_rate.between(0, 1).all() and p.r3_win_rate.between(0, 1).all()
    assert (p.ko_w_p15 >= 0).all() and (p.avg_fin_round_w.between(1, 5)).all()


def test_pace_truncation_invariance(clean, built, cfg):
    full = built[1]
    rng = np.random.default_rng(21)
    dates = np.sort(clean.date.unique())
    for cut in list(rng.choice(dates[300:], 6, replace=False)) + [dates[-1]]:
        part = clean[clean.date <= cut].reset_index(drop=True)
        pre = build_all(part, cfg)[1]
        ids = pre[pre.date == cut].set_index(["fight_id", "side"]).index
        a, b = _pre(full, ids), _pre(pre, ids)
        assert np.allclose(a.values, b.values, rtol=1e-12, atol=1e-12, equal_nan=True)
        ids = pre.sample(400, random_state=2).set_index(["fight_id", "side"]).index
        a, b = _pre(full, ids), _pre(pre, ids)
        assert np.allclose(a.values, b.values, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_mutating_a_fight_leaves_its_own_pace_features_alone(clean, built, cfg):
    full = built[1]
    rng = np.random.default_rng(5)
    pool = clean[(clean.date >= "2014-01-01") & (clean.date <= "2023-12-31") & clean.is_target & clean.r1_secs.notna()]
    seen_later = 0
    for fid in rng.choice(pool.fight_id.values, 5, replace=False):
        m = clean.copy()
        i = m.index[m.fight_id == fid][0]
        for s in "12":
            for c in STAT_COLS + ROUND_COLS:
                m.loc[i, f"f{s}_{c}"] = m.loc[i, f"f{s}_{c}"] * 2 + 9
        m.loc[i, "f1_won"] = 1 - m.loc[i, "f1_won"]
        m.loc[i, "method_group"] = "SUB" if m.loc[i, "method_group"] != "SUB" else "KO"
        m.loc[i, "round"] = 1 if m.loc[i, "round"] > 1 else 3
        m.loc[i, "fight_secs"] = m.loc[i, "fight_secs"] / 2 + 5
        m.loc[i, ["r1_secs", "late_secs"]] = [123.0, 77.0]
        m.loc[i, "sched_rounds"] = 8 - m.loc[i, "sched_rounds"]        # 3 <-> 5
        pre = build_all(m, cfg)[1]
        same_day = full[full.date == clean.date[i]].set_index(["fight_id", "side"]).index
        assert np.array_equal(_pre(full, same_day).values, _pre(pre, same_day).values, equal_nan=True)
        keys = {clean.f1_key[i], clean.f2_key[i]}
        later = full[(full.date > clean.date[i]) & full.fighter.isin(keys)].set_index(["fight_id", "side"]).index
        if len(later):
            assert not np.allclose(_pre(full, later).values, _pre(pre, later).values, equal_nan=True)
            seen_later += 1
    assert seen_later >= 3


def test_pace_sums_match_an_independent_recomputation(built):
    bouts, pre, _ = built
    rng = np.random.default_rng(8)
    pool = np.flatnonzero((pre.date >= "2012-01-01").values)
    for j in rng.choice(pool, 250, replace=False):
        r = pre.iloc[j]
        past = bouts[(bouts.fighter == r.fighter) & (bouts.date < r.date)]
        won, lost = past.score == 1, past.score == 0
        fin = past.method_group.isin(["KO", "SUB"])
        assert np.isclose(r.rawp_tsecs, past.fight_secs.sum())
        assert r.rawp_n5 == (past.sched_rounds == 5).sum()
        assert r.rawp_early_fin_w == (won & fin & (past.end_round <= 2)).sum()
        assert r.rawp_reach3_w == (won & (past.end_round >= 3)).sum()
        rounds = past[past.r1_att.notna() & past.opp_r1_landed.notna()]
        assert np.isclose(r.rawp_r1_att, rounds.r1_att.sum())
        assert np.isclose(r.rawp_r1_secs, rounds.r1_secs.sum())
        assert np.isclose(r.rawp_late_att, rounds.late_att.sum())
        assert np.isclose(r.rawp_late_secs, rounds.late_secs.sum())
        assert r.five_round_fights == (past.sched_rounds == 5).sum()
        assert r.fights_reaching_r4 == (past.end_round >= 4).sum()
        assert np.isclose(r.late_minutes, np.maximum(past.fight_secs - 900, 0).sum() / 60)
        assert r.fights_reaching_r3 == (past.end_round >= 3).sum()


def test_round_seconds_are_consistent_with_the_fight_length(clean):
    c = clean[clean.r1_secs.notna()]
    assert len(c) > 6500
    assert c.r1_secs.between(1, 300).all()
    full_r1 = c[c["round"] > 1]
    assert (full_r1.r1_secs == 300).all()
    # first round + rounds 3 and later can never exceed the whole fight
    assert (c.r1_secs + c.late_secs <= c.fight_secs + 1e-9).all()
    three = c[(c["round"] == 3)]
    assert np.allclose(three.late_secs, three.fight_secs - 600)
    # round-level attempts never exceed the fight totals
    for s in "12":
        assert (c[f"f{s}_r1_att"] + c[f"f{s}_late_att"] <= c[f"f{s}_sig_str_att"] + 1e-9).all()


def test_two_debutants_get_identical_pace_values(built):
    pre = built[1]
    d = pre[(pre.debut == 1) & (pre.date >= "2014-01-01")]
    per_day = d.groupby("date")[["att_pm", "r1_att_pm", "dec_rate", "ko_w_p15", "avg_fight_min"]].nunique()
    assert (per_day == 1).all().all()
