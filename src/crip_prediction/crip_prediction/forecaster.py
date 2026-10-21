#!/usr/bin/env python3
"""
crip_prediction.forecaster

Trajectory forecasting for the CRIP platform (Milestone 6/Module 6).

Given a track's recent position history, predict the next H positions. Two
paths, mirroring the M6 RL design:

  * LSTMForecaster -- loads a trained ONNX recurrent model via onnxruntime.
  * ConstantVelocityForecaster -- analytic fallback (extrapolate last
    velocity). Used when no ONNX model is present, so the node always runs.

Both expose the same interface:
    predict(history_xy: List[(x,y)], horizon: int, dt: float) -> List[(x,y)]

The training data contract is defined by `make_synthetic_tracks` and
`history_to_features` below, so a model trained offline transfers to runtime
without reshaping surprises.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np


# ---- shared feature contract ----
INPUT_LEN = 8     # number of past steps fed to the model
HORIZON = 8       # number of future steps predicted


def history_to_features(history_xy: List[Tuple[float, float]],
                        input_len: int = INPUT_LEN) -> np.ndarray:
    """Convert a position history into per-step delta features of fixed length.

    Uses displacement vectors (dx, dy) between consecutive points, left-padded
    with zeros if the history is short. Shape: (input_len, 2), float32.
    Predicting in delta-space makes the model translation-invariant."""
    pts = list(history_xy)[-(input_len + 1):]
    deltas = [(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
              for i in range(len(pts) - 1)]
    if len(deltas) < input_len:
        deltas = [(0.0, 0.0)] * (input_len - len(deltas)) + deltas
    return np.array(deltas[-input_len:], dtype=np.float32)


class ConstantVelocityForecaster:
    """Extrapolate the most recent velocity. Robust, parameter-free baseline."""

    def predict(self, history_xy, horizon=HORIZON, dt=0.5):
        if len(history_xy) >= 2:
            (x0, y0), (x1, y1) = history_xy[-2], history_xy[-1]
            vx, vy = (x1 - x0), (y1 - y0)  # per-step displacement
        else:
            x1, y1, vx, vy = (*history_xy[-1], 0.0, 0.0) if history_xy else (0, 0, 0, 0)
        out = []
        x, y = x1, y1
        for _ in range(horizon):
            x += vx
            y += vy
            out.append((float(x), float(y)))
        return out


class LSTMForecaster:
    def __init__(self, path: str, logger=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._fallback = ConstantVelocityForecaster()
        if logger:
            logger.info(f"forecaster: loaded ONNX model {path}")

    def predict(self, history_xy, horizon=HORIZON, dt=0.5):
        feats = history_to_features(history_xy)[None, :, :]  # (1, L, 2)
        out = self.session.run(None, {self.input_name: feats})[0]
        deltas = np.asarray(out).reshape(-1, 2)[:horizon]
        # integrate predicted deltas from the last known position
        x, y = history_xy[-1]
        pred = []
        for dx, dy in deltas:
            x += float(dx)
            y += float(dy)
            pred.append((x, y))
        # if the model returned fewer than horizon steps, top up with CV
        if len(pred) < horizon:
            pred += self._fallback.predict(
                history_xy + pred, horizon - len(pred), dt)
        return pred


def make_forecaster(path: Optional[str], logger=None):
    if path and os.path.exists(path):
        try:
            return LSTMForecaster(path, logger)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning(f"forecaster: ONNX load failed ({exc}); "
                               "using constant-velocity fallback")
    if logger:
        logger.info("forecaster: using constant-velocity fallback")
    return ConstantVelocityForecaster()


# ---- synthetic data for offline training/testing ----
def make_synthetic_tracks(n: int, steps: int, rng: np.random.Generator):
    """Generate plausible 2D pedestrian-like tracks: straight lines, gentle
    turns, and stop-and-go, in displacement space. Returns a list of arrays of
    shape (steps, 2) of absolute positions."""
    tracks = []
    for _ in range(n):
        kind = rng.integers(0, 3)
        x, y = rng.uniform(-4, 4, size=2)
        heading = rng.uniform(-math.pi, math.pi)
        speed = rng.uniform(0.05, 0.25)
        turn = rng.normal(0, 0.05) if kind == 1 else 0.0
        pts = []
        for s in range(steps):
            if kind == 2 and (s // 5) % 2 == 1:
                step_speed = 0.0          # stopped phase
            else:
                step_speed = speed
            heading += turn
            x += step_speed * math.cos(heading)
            y += step_speed * math.sin(heading)
            pts.append((x, y))
        tracks.append(np.array(pts, dtype=np.float32))
    return tracks
