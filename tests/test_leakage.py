"""Tests that the feature table contains nothing from the fight it describes or from later."""
import numpy as np
import pandas as pd

from ufc.clean import STAT_COLS
from ufc.features import BASE, FEATURES, INTERACTIONS, MARKET, STATIC, build_all

CHECK = FEATURES + ["y", "p_elo"]
LABELS = ["a_key", "b_key", "swapped"]


def _aligned(a, b, ids):
    return a.set_index("fight_id").loc[ids], b.set_index("fight_id").loc[ids]


def _assert_same(a, b):
    assert np.allclose(a[CHECK].values.astype(float), b[CHECK].values.astype(float),
                       rtol=1e-12, atol=1e-12, equal_nan=True)
    assert (a[LABELS].values == b[LABELS].values).all()


def test_truncation_invariance(clean, built, cfg):
    """Deleting every later date must not change any feature of the fights that remain."""
    full = built[2]
    rng = np.random.default_rng(7)
    dates = np.sort(clean.date.unique())
    cuts = list(rng.choice(dates[200:], 10, replace=False)) + [dates[60], dates[-1]]
    for cut in cuts:
        part = clean[clean.date <= cut].reset_index(drop=True)
        feats = build_all(part, cfg)[2]
        last = feats.fight_id[feats.date == cut]          # most exposed rows: nothing after them
        _assert_same(*_aligned(full, feats, last))
        sample = feats.fight_id.sample(min(300, len(feats)), random_state=1)
        _assert_same(*_aligned(full, feats, sample))


def test_mutating_a_fight_leaves_its_own_features_alone(clean, built, cfg):
    """Scramble one fight: its own features stay, later fights of those fighters change."""
    full = built[2]
    rng = np.random.default_rng(11)
    pool = clean[(clean.date >= "2012-01-01") & (clean.date <= "2022-12-31") & clean.is_target]
    checked_later = 0
    for fid in rng.choice(pool.fight_id.values, 6, replace=False):
        m = clean.copy()
        i = m.index[m.fight_id == fid][0]
        for s in "12":
            for c in STAT_COLS:
                m.loc[i, f"f{s}_{c}"] = m.loc[i, f"f{s}_{c}"] * 3 + 17
        m.loc[i, "f1_won"] = 1 - m.loc[i, "f1_won"]
        m.loc[i, "method_group"] = "SUB" if m.loc[i, "method_group"] != "SUB" else "KO"
        m.loc[i, "fight_secs"] = m.loc[i, "fight_secs"] / 2 + 5
        feats = build_all(m, cfg)[2]
        same_day = clean.fight_id[clean.date == clean.date[i]]
        a, b = _aligned(full, feats, same_day)
        assert np.array_equal(a[FEATURES + ["p_elo"]].values, b[FEATURES + ["p_elo"]].values)
        keys = {clean.f1_key[i], clean.f2_key[i]}
        later = clean[(clean.date > clean.date[i])
                      & (clean.f1_key.isin(keys) | clean.f2_key.isin(keys))]
        if len(later):
            a, b = _aligned(full, feats, later.fight_id)
            assert not np.allclose(a[FEATURES].values, b[FEATURES].values)
            checked_later += 1
    assert checked_later >= 3


def test_orientation_is_antisymmetric(clean, built, cfg):
    full = built[2]
    flipped = build_all(clean, cfg, flip=True)[2]
    assert (full.fight_id == flipped.fight_id).all()
    assert (full.swapped != flipped.swapped).all()
    assert np.allclose(full[FEATURES].values, -flipped[FEATURES].values)
    assert np.allclose(full.p_elo + flipped.p_elo, 1)
    t = full.is_target
    assert np.allclose(full.y[t] + flipped.y[t], 1)
    m = full.has_market
    assert np.allclose(full.p_open[m] + flipped.p_open[m], 1)
    assert (full.a_open_decimal[m].values == flipped.b_open_decimal[m].values).all()


def test_fighter_order_carries_no_information(built):
    f = built[2]
    f = f[f.is_target]
    for year, g in f.groupby(f.date.dt.year):
        se = 0.5 / np.sqrt(len(g))
        assert abs(g.y.mean() - 0.5) < 3.5 * se, (year, g.y.mean(), len(g))
    assert abs(f.y.mean() - 0.5) < 0.015
    # `swapped` records whether the original (winner-first) order was reversed, so it
    # does predict the result. It is bookkeeping only and must never be a model input.
    assert abs(np.corrcoef(f.swapped, f.y)[0, 1]) > 0.1
    assert "swapped" not in FEATURES


def test_static_attributes_have_no_missing_values_and_no_missing_flags(built):
    pre, f = built[1], built[2]
    assert not f[FEATURES].isna().any().any()
    p1 = pre[pre.side == 1].set_index("fight_id").loc[f.fight_id]
    p2 = pre[pre.side == 2].set_index("fight_id").loc[f.fight_id]
    for col in STATIC:
        either_missing = p1[col].isna().values | p2[col].isna().values
        assert either_missing.any()
        assert (f["d_" + col].values[either_missing] == 0).all()
    # the only absence indicator allowed is the debut flag, which history decides
    assert not [c for c in FEATURES if "missing" in c or "isna" in c]


def test_feature_list_is_closed_and_market_free(built):
    assert set(FEATURES) == {"d_" + c for c in BASE} | set(INTERACTIONS)
    assert not set(FEATURES) & set(MARKET)
    banned = ("odds", "open", "close", "decimal")
    assert not [c for c in FEATURES if any(b in c for b in banned)]
    raw_stats = {f"f{s}_{c}" for s in "12" for c in STAT_COLS}
    assert not raw_stats & set(built[2].columns)


def test_history_matches_an_independent_recomputation(built):
    """Re-derive record and raw sums with plain pandas filters (date strictly earlier)."""
    bouts, pre, _ = built
    rng = np.random.default_rng(3)
    for j in rng.choice(len(pre), 250, replace=False):
        r = pre.iloc[j]
        past = bouts[(bouts.fighter == r.fighter) & (bouts.date < r.date)]
        assert r.n_fights == len(past)
        assert r.raw_wins == (past.score == 1).sum()
        assert r.raw_losses == (past.score == 0).sum()
        assert r.raw_ko_l == ((past.score == 0) & (past.method_group == "KO")).sum()
        with_stats = past[past.sig_str_att.notna() & past.opp_sig_str_att.notna()]
        assert np.isclose(r.raw_secs, with_stats.fight_secs.sum())
        assert np.isclose(r.raw_sig_l, with_stats.sig_str_landed.sum())
        assert np.isclose(r.raw_td_l, with_stats.td_landed.sum())
        with_ctrl = past[past.ctrl_seconds.notna() & past.opp_ctrl_seconds.notna()]
        assert np.isclose(r.raw_ctrl, with_ctrl.ctrl_seconds.sum())
        if len(past):
            assert r.days_since == min((r.date - past.date.max()).days, 730)
        else:
            assert r.debut == 1 and np.isnan(r.days_since)


def test_two_debutants_get_identical_history_features(built):
    pre, f = built[1], built[2]
    p1 = pre[pre.side == 1].set_index("fight_id").loc[f.fight_id]
    p2 = pre[pre.side == 2].set_index("fight_id").loc[f.fight_id]
    both = (p1.debut.values == 1) & (p2.debut.values == 1) & (f.date >= "2005-01-01").values
    assert both.sum() > 100
    cols = [c for c in FEATURES if c.startswith(("d_c_", "d_r_"))] + ["d_win_rate", "d_elo"]
    assert np.allclose(f.loc[both, cols].values, 0)


def test_rates_are_in_sane_ranges(built):
    pre = built[1]
    p = pre[pre.date >= "2005-01-01"]
    for c in ("c_sig_acc", "c_sig_def", "c_td_acc", "c_td_def", "c_ctrl_share", "win_rate",
              "finish_rate"):
        assert p[c].between(0, 1).all(), c
    assert p.c_slpm.between(0, 15).all() and p.c_sapm.between(0, 15).all()
    assert p.age.dropna().between(18, 50).all()
    assert p.elo.between(1100, 2100).all()


def test_elo_matches_an_independent_implementation(clean, built, cfg):
    """Recompute Elo fight by fight with separate code and compare pre-fight ratings."""
    pre = built[1]
    e = cfg["elo"]
    rating, fights = {}, {}
    expected = {}
    for date, day in clean.groupby("date", sort=True):
        pending = []
        for r in day.itertuples():
            ra, rb = rating.get(r.f1_key, e["start"]), rating.get(r.f2_key, e["start"])
            expected[(r.fight_id, 1)], expected[(r.fight_id, 2)] = ra, rb
            if r.result == "no contest":
                score = None
            elif r.result == "draw":
                score = 0.5
            else:
                score = r.f1_won
            for key, mine, theirs, sc in ((r.f1_key, ra, rb, score),
                                          (r.f2_key, rb, ra, None if score is None else 1 - score)):
                change = 0.0
                if sc is not None:
                    k = e["k"] * (1 + e["new_boost"] / (1 + fights.get(key, 0)))
                    change = k * (sc - 1 / (1 + 10 ** ((theirs - mine) / 400)))
                pending.append((key, change))
        for key, change in pending:      # the whole night is applied at once
            rating[key] = rating.get(key, e["start"]) + change
            fights[key] = fights.get(key, 0) + 1
    got = pre.set_index(["fight_id", "side"]).elo
    want = pd.Series(expected).reindex(got.index)
    assert np.allclose(got.values, want.values)


def test_missing_reach_is_not_recognisable_as_a_zero(built):
    """Reach is missing mostly for fighters who were cut early. With height known, the
    reach difference must not sit at an exact zero that a model could learn to spot."""
    from ufc import ROOT
    f = built[2]
    raw = pd.read_csv(ROOT / "Datasets/ufc_fights_enriched.csv",
                      usecols=["fight_url", "f1_reach_cm", "f2_reach_cm", "f1_height_cm", "f2_height_cm"])
    raw["fight_id"] = raw.fight_url.str.extract(r"fight-details/(\w+)")[0]
    raw = raw.dropna(subset=["fight_id"]).set_index("fight_id").reindex(f.fight_id)
    one_side = (raw.f1_reach_cm.isna() ^ raw.f2_reach_cm.isna()).values
    heights = (raw.f1_height_cm.notna() & raw.f2_height_cm.notna()).values
    train = (f.split == "train").values
    rows = one_side & heights & train
    assert rows.sum() > 300
    assert (f.d_reach_cm.values[rows] == 0).mean() < 0.05
    # more-experienced side must not win unusually often where reach was filled in...
    # that pattern is real (those fighters were cut), but it must not be findable from a zero
    zero = (f.d_reach_cm.values == 0) & train & f.is_target.values
    assert (one_side & zero).sum() < 0.1 * zero.sum()
