import numpy as np
import pandas as pd

from ufc.clean import fight_seconds, scheduled_rounds


def test_fight_seconds():
    assert fight_seconds(3, "5:00", "3 Rnd (5-5-5)") == 900
    assert fight_seconds(1, "1:00", "3 Rnd (5-5-5)") == 60
    assert fight_seconds(3, "2:27", "5 Rnd (5-5-5-5-5)") == 747
    assert fight_seconds(2, "3:00", "1 Rnd + OT (12-3)") == 900       # overtime counts as a round
    assert fight_seconds(3, "3:00", "1 Rnd + 2OT (15-3-3)") == 1260
    assert fight_seconds(1, "4:20", "No Time Limit") == 260
    assert fight_seconds(1, "0:59", float("nan")) == 59


def test_scheduled_rounds():
    assert scheduled_rounds("3 Rnd (5-5-5)") == 3
    assert scheduled_rounds("5 Rnd (5-5-5-5-5)") == 5
    assert scheduled_rounds("3 Rnd + OT (5-5-5-5)") == 3
    assert np.isnan(scheduled_rounds("No Time Limit"))


def test_row_accounting(clean):
    assert len(clean) == 8998
    assert clean.fight_id.is_unique
    assert int(clean.is_target.sum()) == 8842
    assert clean.date.is_monotonic_increasing
    assert clean.fight_secs.between(1, 60 * 60).all()


def test_no_fight_exceeds_its_schedule(clean):
    three = clean[clean.time_format == "3 Rnd (5-5-5)"]
    assert three.fight_secs.max() <= 900
    five = clean[clean.time_format == "5 Rnd (5-5-5-5-5)"]
    assert five.fight_secs.max() <= 1500


def test_bruno_silva_is_two_fighters(clean):
    keys = (set(clean.f1_key[clean.fighter_1 == "Bruno Silva"])
            | set(clean.f2_key[clean.fighter_2 == "Bruno Silva"]))
    assert keys == {"12ebd7d157e91701", "294aa73dbf37d281"}
    for k in keys:   # each of them stays within one weight band
        lbs = clean.lbs[(clean.f1_key == k) | (clean.f2_key == k)]
        assert lbs.max() - lbs.min() <= 20


def test_no_fighter_jumps_far_in_weight_within_a_year(clean):
    long = pd.concat([clean[["date", "f1_key", "lbs"]].rename(columns={"f1_key": "k"}),
                      clean[["date", "f2_key", "lbs"]].rename(columns={"f2_key": "k"})])
    long = long[(long.date >= "2001-01-01") & long.lbs.notna()].sort_values(["k", "date"])
    prev = long.groupby("k")[["date", "lbs"]].shift()
    close_in_time = (long.date - prev.date).dt.days <= 365
    jump = (long.lbs - prev.lbs).abs()
    # heavyweight is a 60 lb band, so one class up to 265 is allowed
    assert (jump[close_in_time] <= 80).all(), long[close_in_time & (jump > 80)]


def test_market_probabilities_are_valid(clean):
    m = clean[clean.has_market]
    assert len(m) == 6704
    assert m.f1_p_open.between(0.02, 0.98).all() and m.f1_p_close.between(0.02, 0.98).all()


def test_father_and_son_are_two_fighters(clean):
    senior = (set(clean.f1_key[clean.fighter_1 == "Lance Gibson"])
              | set(clean.f2_key[clean.fighter_2 == "Lance Gibson"]))
    junior = (set(clean.f1_key[clean.fighter_1 == "Lance Gibson Jr."])
              | set(clean.f2_key[clean.fighter_2 == "Lance Gibson Jr."]))
    assert senior and junior and not senior & junior


def _profiles(clean):
    cols = ["date", "k", "dob", "h", "r"]
    return pd.concat([
        clean[["date", "f1_key", "f1_dob", "f1_height_cm", "f1_reach_cm"]].set_axis(cols, axis=1),
        clean[["date", "f2_key", "f2_dob", "f2_height_cm", "f2_reach_cm"]].set_axis(cols, axis=1)])


def test_one_profile_per_fighter(clean):
    per_key = _profiles(clean).groupby("k")[["dob", "h", "r"]].agg(lambda s: s.nunique(dropna=False))
    assert (per_key == 1).all().all()


def test_no_career_spans_an_implausible_gap(clean):
    long = _profiles(clean).sort_values(["k", "date"])
    gap = long.groupby("k").date.diff().dt.days / 365.25
    assert gap.max() < 12          # the longest real returns are about 11 years


def test_suspect_opening_lines_are_flagged(clean):
    s = clean[clean.open_suspect]
    assert 1 <= len(s) <= 5
    assert "Vanessa Demopoulos" in set(s.fighter_1) | set(s.fighter_2)
    assert not (clean.market_ok & (clean.odds_conflict | clean.open_suspect)).any()


def test_reach_from_height_constants_match_the_pre_2010_fit(clean, cfg):
    p = _profiles(clean).groupby("k").agg(first=("date", "min"), h=("h", "first"), r=("r", "first")).dropna()
    p = p[p["first"] < "2010-01-01"]
    slope, intercept = np.polyfit(p.h, p.r, 1)
    rh = cfg["history"]["reach_from_height"]
    for height in (160, 180, 200):
        assert abs((intercept + slope * height) - (rh["intercept"] + rh["slope"] * height)) < 1.5
