#!/usr/bin/env python3
"""
crip_learning.dataset

Experience logging + dataset assembly for the self-learning loop (Module 8).

The robot logs experience as JSON lines. Two record kinds matter for the
current loop:

  {"kind": "track", "t": <float>, "id": <int>, "x": <float>, "y": <float>}
      A single tracked-object observation (from the prediction tracker).

  {"kind": "mission", "t": <float>, "mission": "...", "outcome": "...",
   "n_tasks": <int>}
      A completed mission and its outcome (from the brain).

`ExperienceLog` appends records to a JSONL file. `build_forecast_dataset`
reconstructs per-object trajectories from "track" records and turns them into
the same (INPUT_LEN, 2) -> (HORIZON, 2) displacement windows the forecaster
trains on -- so the loop learns the motion patterns the robot actually
observed, not just synthetic ones.

This module is ROS-free and unit-testable.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# kept consistent with crip_prediction.forecaster
INPUT_LEN = 8
HORIZON = 8


class ExperienceLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def reconstruct_tracks(records: List[dict]) -> Dict[int, List[Tuple[float, float]]]:
    """Group 'track' records by id into ordered position sequences."""
    by_id: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
    for r in records:
        if r.get("kind") == "track":
            by_id[int(r["id"])].append((float(r["t"]), float(r["x"]),
                                        float(r["y"])))
    tracks = {}
    for tid, pts in by_id.items():
        pts.sort(key=lambda p: p[0])
        tracks[tid] = [(x, y) for (_, x, y) in pts]
    return tracks


def build_forecast_dataset(records: List[dict],
                           input_len: int = INPUT_LEN,
                           horizon: int = HORIZON):
    """Turn logged tracks into (X, Y) displacement windows.
    X: (N, input_len, 2)  Y: (N, horizon, 2)  float32."""
    tracks = reconstruct_tracks(records)
    X, Y = [], []
    for pts in tracks.values():
        if len(pts) < input_len + horizon + 1:
            continue
        arr = np.array(pts, dtype=np.float32)
        deltas = np.diff(arr, axis=0)  # (len-1, 2)
        for i in range(len(deltas) - input_len - horizon):
            X.append(deltas[i:i + input_len])
            Y.append(deltas[i + input_len:i + input_len + horizon])
    if not X:
        return (np.zeros((0, input_len, 2), np.float32),
                np.zeros((0, horizon, 2), np.float32))
    return np.array(X, np.float32), np.array(Y, np.float32)


def dataset_stats(records: List[dict]) -> dict:
    tracks = reconstruct_tracks(records)
    missions = [r for r in records if r.get("kind") == "mission"]
    successes = sum(1 for m in missions if m.get("outcome") == "success")
    return {
        "n_records": len(records),
        "n_tracks": len(tracks),
        "total_track_points": sum(len(p) for p in tracks.values()),
        "n_missions": len(missions),
        "mission_success_rate": (successes / len(missions)) if missions else 0.0,
    }
