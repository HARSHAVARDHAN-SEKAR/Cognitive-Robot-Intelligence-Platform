"""Unit tests for the self-learning dataset + registry (no ROS, no torch)."""
import os
import tempfile

import numpy as np

from crip_learning.dataset import (ExperienceLog, build_forecast_dataset,
                                   reconstruct_tracks, dataset_stats,
                                   INPUT_LEN, HORIZON)
from crip_learning.registry import ModelRegistry


def _log_with_track(path, n=30):
    log = ExperienceLog(path)
    for i in range(n):
        log.append({"kind": "track", "t": i * 0.5, "id": 0,
                    "x": i * 0.2, "y": 0.0})
    return log


def test_log_append_and_read():
    with tempfile.TemporaryDirectory() as d:
        log = _log_with_track(os.path.join(d, "e.jsonl"))
        recs = log.read()
        assert len(recs) == 30
        assert all(r["kind"] == "track" for r in recs)


def test_reconstruct_tracks_orders_by_time():
    recs = [
        {"kind": "track", "t": 2.0, "id": 0, "x": 2.0, "y": 0.0},
        {"kind": "track", "t": 0.0, "id": 0, "x": 0.0, "y": 0.0},
        {"kind": "track", "t": 1.0, "id": 0, "x": 1.0, "y": 0.0},
    ]
    tracks = reconstruct_tracks(recs)
    assert tracks[0] == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


def test_build_forecast_dataset_shapes():
    recs = [{"kind": "track", "t": i * 0.5, "id": 0, "x": i * 0.2, "y": 0.0}
            for i in range(40)]
    X, Y = build_forecast_dataset(recs)
    assert X.shape[1:] == (INPUT_LEN, 2)
    assert Y.shape[1:] == (HORIZON, 2)
    assert len(X) == len(Y) and len(X) > 0


def test_build_forecast_dataset_too_short():
    recs = [{"kind": "track", "t": i, "id": 0, "x": i, "y": 0} for i in range(5)]
    X, Y = build_forecast_dataset(recs)
    assert len(X) == 0  # not enough points for a window


def test_dataset_stats():
    recs = [{"kind": "track", "t": i, "id": 0, "x": i, "y": 0} for i in range(10)]
    recs += [{"kind": "mission", "t": 1, "outcome": "success"},
             {"kind": "mission", "t": 2, "outcome": "failure"}]
    s = dataset_stats(recs)
    assert s["n_tracks"] == 1
    assert s["n_missions"] == 2
    assert abs(s["mission_success_rate"] - 0.5) < 1e-9


def test_registry_add_does_not_autopromote():
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(d)
        f = os.path.join(d, "m.onnx")
        open(f, "wb").write(b"x")
        v = reg.add_version(f, {"val_mse": 0.1})
        assert v == "v0001"
        assert reg.current_version() is None


def test_registry_promote_and_current():
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(d)
        f1 = os.path.join(d, "m1.onnx"); open(f1, "wb").write(b"a")
        f2 = os.path.join(d, "m2.onnx"); open(f2, "wb").write(b"b")
        v1 = reg.add_version(f1, {"val_mse": 0.10})
        v2 = reg.add_version(f2, {"val_mse": 0.05})
        reg.promote(v1)
        assert reg.current_version() == v1
        reg.promote(v2)
        assert reg.current_version() == v2
        assert reg.current_meta()["val_mse"] == 0.05
        assert os.path.exists(reg.current_model_path())


def test_registry_promote_unknown_raises():
    import pytest
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(d)
        with pytest.raises(ValueError):
            reg.promote("v9999")


def test_promotion_gate_logic():
    """The gate rule the pipeline uses, tested in isolation."""
    def should_promote(cand, current, margin):
        if current is None:
            return True
        return cand <= current * (1.0 - margin)

    assert should_promote(0.05, None, 0.02) is True          # first model
    assert should_promote(0.08, 0.10, 0.02) is True          # 20% better
    assert should_promote(0.099, 0.10, 0.02) is False        # <2% better
    assert should_promote(0.12, 0.10, 0.02) is False         # worse
