"""Unit tests for tracking + forecasting cores (no ROS, no torch)."""
import numpy as np

from crip_prediction.tracker import Tracker
from crip_prediction.forecaster import (
    ConstantVelocityForecaster, history_to_features, make_synthetic_tracks,
    INPUT_LEN, HORIZON)


def test_tracker_velocity_estimate():
    tr = Tracker(gate=1.0)
    t = 0.0
    tracks = []
    for i in range(6):
        tracks = tr.update([(i * 0.5, 0.0)], t)  # +1.0 m/s
        t += 0.5
    assert len(tracks) == 1
    assert abs(tracks[0].vx - 1.0) < 0.2
    assert abs(tracks[0].vy) < 0.05


def test_tracker_two_objects_no_swap():
    tr = Tracker(gate=1.0)
    t = 0.0
    for i in range(5):
        tr.update([(i * 0.3, 0.0), (i * 0.3, 3.0)], t)
        t += 0.5
    assert len(tr.tracks) == 2


def test_tracker_ages_out_stale():
    tr = Tracker(gate=1.0, max_age=1.0)
    tr.update([(0.0, 0.0)], 0.0)
    tr.update([(5.0, 5.0)], 3.0)
    assert len(tr.tracks) == 1


def test_constant_velocity_extrapolation():
    cv = ConstantVelocityForecaster()
    hist = [(0, 0), (1, 0), (2, 0), (3, 0)]
    pred = cv.predict(hist, horizon=3)
    assert pred[0] == (4.0, 0.0)
    assert pred[2] == (6.0, 0.0)


def test_constant_velocity_single_point_safe():
    cv = ConstantVelocityForecaster()
    pred = cv.predict([(1.0, 1.0)], horizon=3)
    assert len(pred) == 3
    assert pred[0] == (1.0, 1.0)  # no velocity -> stays put


def test_feature_shape_and_padding():
    f = history_to_features([(0, 0), (1, 1)])
    assert f.shape == (INPUT_LEN, 2)
    assert np.allclose(f[-1], [1, 1])
    assert np.allclose(f[0], [0, 0])


def test_synthetic_tracks_shape():
    tracks = make_synthetic_tracks(5, 20, np.random.default_rng(0))
    assert len(tracks) == 5
    assert tracks[0].shape == (20, 2)
