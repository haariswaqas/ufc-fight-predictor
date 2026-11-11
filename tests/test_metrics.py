import numpy as np

from ufc.market import fit_stretch, flat_return, sigmoid, stretch
from ufc.metrics import accuracy, brier, calibration_table, event_bootstrap, log_loss


def test_log_loss_and_brier_by_hand():
    y, p = [1, 0], [0.8, 0.4]
    assert np.isclose(log_loss(y, p), -(np.log(0.8) + np.log(0.6)) / 2)
    assert np.isclose(brier(y, p), (0.04 + 0.16) / 2)
    assert np.isclose(log_loss([1, 0], [0.5, 0.5]), np.log(2))


def test_accuracy_counts_coin_flips_as_half():
    assert accuracy([1, 0, 1, 1], [0.9, 0.2, 0.3, 0.5]) == (1 + 1 + 0 + 0.5) / 4


def test_calibration_table_uses_the_favourite_side():
    # favourites: B at 0.70 (B won) and A at 0.72 (A won)
    t = calibration_table([0, 1], [0.3, 0.72])
    assert t.fights.sum() == 2 and (t.actual == 1).all()


def test_stretch_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 200_000)
    true = sigmoid(1.3 * np.log(p / (1 - p)))
    y = rng.uniform(size=p.size) < true
    s = fit_stretch(p, y)
    assert abs(s - 1.3) < 0.03
    assert np.allclose(stretch(p, s) + stretch(1 - p, s), 1)     # symmetric in the two fighters


def test_flat_return():
    # bet A at 2.0 and win (+1); bet B at 3.0 and lose (-1)
    r = flat_return([1, 1], np.array([2.0, 1.4]), np.array([1.8, 3.0]), np.array([True, False]))
    assert r == 0.0


def test_event_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 1.0, 600)
    dates = np.repeat(np.arange(60), 10)
    lo, hi = event_bootstrap(dates, lambda i: x[i].mean(), reps=500, seed=2)
    assert lo < x.mean() < hi and hi - lo < 0.5
