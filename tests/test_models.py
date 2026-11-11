"""Stage 2 models must give the same answer whichever fighter is called A."""
import numpy as np
import pytest

from ufc.features import build_all
from ufc.models import ANTI, FEATURE_SETS, SYM, SymGBM, SymLogit, design


@pytest.fixture(scope="module")
def tables(clean, built, cfg):
    part = clean[clean.date < cfg["splits"]["test_start"]].reset_index(drop=True)
    _, pre, f = build_all(part, cfg)
    flipped = build_all(part, cfg, flip=True)[2]
    return design(f, pre), design(flipped, pre)


def _rows(d, market):
    tr = d.is_target & (d.date >= "2013-03-01") & (d.date < "2019-01-01")
    te = (d.date >= "2019-01-01") & (d.date < "2020-01-01")
    if market:
        tr, te = tr & d.market_ok, te & d.has_market
    return tr.values, te.values


def test_design_terms_flip_sign_and_context_does_not(tables):
    d, fl = tables
    assert (d.fight_id == fl.fight_id).all()
    assert np.allclose(d[ANTI].values, -fl[ANTI].values)
    assert np.allclose(d[SYM].values, fl[SYM].values, equal_nan=True)
    assert not d[ANTI].isna().any().any()


@pytest.mark.parametrize("market", [False, True])
def test_logistic_model_is_symmetric(tables, market):
    d, fl = tables
    tr, te = _rows(d, market)
    m = SymLogit(FEATURE_SETS["all"], 0.003, market=market).fit(d[tr], d.y[tr])
    assert np.allclose(m.predict(d[te]) + m.predict(fl[te]), 1, atol=1e-10)
    # training on the reversed table gives the same model
    m2 = SymLogit(FEATURE_SETS["all"], 0.003, market=market).fit(fl[tr], fl.y[tr])
    assert np.allclose(m.predict(d[te]), m2.predict(d[te]), atol=1e-4)


@pytest.mark.parametrize("market", [False, True])
def test_boosted_model_is_symmetric(tables, market):
    d, fl = tables
    tr, te = _rows(d, market)
    params = {"num_leaves": 6, "learning_rate": 0.05, "min_child_samples": 80, "feature_fraction": 0.7}
    m = SymGBM(params, 60, market=market).fit(d[tr], d.y[tr], market_slope=1.1)
    p, q = m.predict(d[te]), m.predict(fl[te])
    assert np.allclose(p + q, 1, atol=1e-10)
    assert p.std() > 0.02


def test_no_test_rows_reach_training(cfg):
    from ufc.walk import load_dev
    dev = load_dev(cfg)
    assert dev.date.max() < np.datetime64(cfg["splits"]["test_start"])
    assert set(dev.split) <= {"history", "train", "val"}
