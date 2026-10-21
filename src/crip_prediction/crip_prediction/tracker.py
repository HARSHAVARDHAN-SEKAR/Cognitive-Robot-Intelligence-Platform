#!/usr/bin/env python3
"""
crip_prediction.tracker

Lightweight multi-object tracker (ROS-free, unit-testable).

Consumes timestamped 2D position observations and maintains a set of tracks.
Each track keeps a short history of (t, x, y) and an estimated velocity
(finite-difference, lightly smoothed). This is the data source the forecaster
turns into future-trajectory predictions.

Association is nearest-neighbour within a gating distance -- adequate for a
handful of dynamic obstacles. A track is dropped if unseen for max_age
seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Track:
    track_id: int
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (t,x,y)
    vx: float = 0.0
    vy: float = 0.0
    last_t: float = 0.0

    @property
    def x(self):
        return self.history[-1][1] if self.history else 0.0

    @property
    def y(self):
        return self.history[-1][2] if self.history else 0.0

    def positions(self, n: Optional[int] = None) -> List[Tuple[float, float]]:
        h = self.history if n is None else self.history[-n:]
        return [(x, y) for (_, x, y) in h]


class Tracker:
    def __init__(self, gate: float = 1.0, max_age: float = 2.0,
                 max_history: int = 20, vel_smooth: float = 0.5):
        self.gate = gate
        self.max_age = max_age
        self.max_history = max_history
        self.vel_smooth = vel_smooth
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def update(self, observations: List[Tuple[float, float]], t: float):
        """Associate observations (list of (x,y)) at time t; update tracks."""
        unmatched = set(range(len(observations)))
        # greedy nearest-neighbour association
        for tr in self.tracks.values():
            best_j, best_d = None, self.gate
            for j in unmatched:
                ox, oy = observations[j]
                d = math.hypot(ox - tr.x, oy - tr.y)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None:
                self._extend(tr, observations[best_j], t)
                unmatched.discard(best_j)

        # spawn new tracks for unmatched observations
        for j in unmatched:
            tr = Track(track_id=self._next_id)
            self._next_id += 1
            self._extend(tr, observations[j], t)
            self.tracks[tr.track_id] = tr

        # age out stale tracks
        stale = [tid for tid, tr in self.tracks.items()
                 if t - tr.last_t > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        return list(self.tracks.values())

    def _extend(self, tr: Track, obs, t: float):
        ox, oy = obs
        if tr.history:
            dt = max(t - tr.last_t, 1e-3)
            nvx = (ox - tr.x) / dt
            nvy = (oy - tr.y) / dt
            a = self.vel_smooth
            tr.vx = a * nvx + (1 - a) * tr.vx
            tr.vy = a * nvy + (1 - a) * tr.vy
        tr.history.append((t, ox, oy))
        if len(tr.history) > self.max_history:
            tr.history = tr.history[-self.max_history:]
        tr.last_t = t
