"""The paper-trading path must build the same features as the historical pipeline."""
import numpy as np
import pandas as pd

from ufc.features import FEATURES
from ufc.models import ANTI, SYM, design
from ufc.paper import upcoming_features


def test_upcoming_card_gets_the_same_features_as_history(clean, built, cfg):
    """Replay a past card as if it were upcoming: hide it and everything after it, feed it in
    as an upcoming card, and compare with the features the full pipeline gave those fights."""
    _, pre, full = built
    full = design(full, pre)
    card_date = pd.Timestamp("2026-10-10")
    card = clean[clean.date == card_date]
    assert len(card) >= 8
    past = clean[clean.date < card_date].reset_index(drop=True)
    up = pd.DataFrame({
        "date": card.date.dt.strftime("%Y-%m-%d").values, "event": card.event.values,
        "fighter_1": card.fighter_1.values, "fighter_2": card.fighter_2.values,
        "bout_type": card.bout_type.values, "rounds": card.sched_rounds.astype(int).values,
        "f1_decimal": card.f1_open_decimal.values, "f2_decimal": card.f2_open_decimal.values,
        "bookmaker": "replay"})
    _, rows, notes = upcoming_features(up, past, cfg)
    want = full.set_index("fight_id").loc[card.fight_id]
    # the replayed fights get new ids, so their random order differs: compare in file order
    s_got = np.where(rows.swapped.values, -1.0, 1.0)[:, None]
    s_want = np.where(want.swapped.values, -1.0, 1.0)[:, None]
    known = np.array([n == "" for n in notes])            # fighters already in the data
    assert known.sum() >= 6
    got_x, want_x = rows[ANTI].values * s_got, want[ANTI].values * s_want
    assert np.allclose(got_x[known], want_x[known], atol=1e-9)
    assert np.allclose(rows[SYM].values[known], want[SYM].values[known], equal_nan=True)
    # a fighter new to the data may only differ in the profile-based terms
    history_cols = [i for i, c in enumerate(ANTI) if not any(k in c for k in ("age", "height", "reach", "southpaw"))]
    assert np.allclose(got_x[:, history_cols], want_x[:, history_cols], atol=1e-9)
    p_got = np.where(rows.swapped.values, 1 - rows.p_open.values, rows.p_open.values)
    p_want = np.where(want.swapped.values, 1 - want.p_open.values, want.p_open.values)
    priced = ~np.isnan(p_want)
    assert np.allclose(p_got[priced], p_want[priced], atol=1e-9)
    assert set(FEATURES) <= set(rows.columns)


def test_method_and_round_probabilities_for_an_upcoming_card(clean, built, cfg):
    """Replay a past card as upcoming: each priced fight's method/round probabilities sum to
    one, split between the fighters as the winner model says, and respect the fight length."""
    from ufc.paper import FINISH_COLS, fit_finish, load_finish_frozen, score_full
    card = clean[clean.date == pd.Timestamp("2026-10-10")]
    up = pd.DataFrame({
        "date": "2026-11-21", "event": "replay", "fighter_1": card.fighter_1.values, "fighter_2": card.fighter_2.values,
        "bout_type": card.bout_type.values, "rounds": card.sched_rounds.astype(int).values,
        "f1_decimal": card.f1_open_decimal.values, "f2_decimal": card.f2_open_decimal.values, "bookmaker": "replay"})
    up.loc[0, ["f1_decimal", "f2_decimal"]] = np.nan          # one fight without a price
    old = clean[clean.date < "2020-01-01"].reset_index(drop=True)
    pre = built[1]
    fin_fit = fit_finish(old, pre[pre.date < "2020-01-01"], load_finish_frozen())
    res, _, _, fin = score_full(up, cfg, finish=fin_fit)
    assert list(fin.columns) == FINISH_COLS and len(fin) == len(up)
    priced = res.model_p_f1.notna().values
    assert not priced[0] and fin.iloc[0].isna().all()
    assert np.allclose(fin[priced].sum(axis=1), 1)
    f1 = fin[[c for c in FINISH_COLS if c.startswith("f1_")]].sum(axis=1)
    assert np.allclose(f1[priced], res.model_p_f1[priced], atol=1e-4)
    three = (up.rounds == 3).values & priced
    late = [c for c in FINISH_COLS if c[-1] in "45"]
    assert (fin.loc[three, late].values == 0).all()
    assert (fin.loc[priced, [c for c in FINISH_COLS if c not in late]].values > 0).all()
