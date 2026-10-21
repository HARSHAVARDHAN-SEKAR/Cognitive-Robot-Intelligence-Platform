#!/usr/bin/env python3
"""
crip_dashboard.state

Pure-Python state aggregator for the dashboard (ROS-free, unit-testable).

The ROS node feeds raw values in via the `update_*` methods; the dashboard
server serializes `snapshot()` to JSON and pushes it to connected browsers.
Keeping this logic out of the ROS node means the serialization, trimming, and
staleness rules are testable without a ROS graph.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Tuple


class DashboardState:
    def __init__(self, trail_len: int = 200):
        self.start = time.time()
        # robot
        self.robot = {"x": 0.0, "y": 0.0, "yaw": 0.0, "t": 0.0}
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=trail_len)
        # static obstacles (from the known world; set once)
        self.obstacles: List[dict] = []
        # latest goal
        self.goal = None
        # predicted tracks: id -> list of (x, y)
        self.predictions: Dict[int, List[Tuple[float, float]]] = {}
        # rolling event/log feed
        self.events: Deque[dict] = deque(maxlen=50)
        # subsystem statuses
        self.vlm = {"risk": "none", "caption": "", "t": 0.0}
        self.learning = {"status": "idle", "t": 0.0}
        self.brain = {"last": "", "t": 0.0}
        self.why = ""          # latest causal explanation (M12)
        self.fleet = ""        # fleet summary line (M12)
        self.health = {}       # module health summary (improvement #7)

    # ---- updates ----
    def update_robot(self, x, y, yaw, t=None):
        t = t if t is not None else time.time()
        self.robot = {"x": x, "y": y, "yaw": yaw, "t": t}
        # only append to trail if moved enough (keeps it readable)
        if not self.trail or (abs(self.trail[-1][0] - x) > 0.05 or
                              abs(self.trail[-1][1] - y) > 0.05):
            self.trail.append((x, y))

    def set_obstacles(self, obstacles: List[dict]):
        self.obstacles = obstacles

    def update_goal(self, x, y):
        self.goal = {"x": x, "y": y, "t": time.time()}

    def update_prediction(self, track_id: int, points: List[Tuple[float, float]]):
        self.predictions[int(track_id)] = list(points)

    def update_vlm(self, risk, caption):
        self.vlm = {"risk": risk, "caption": caption, "t": time.time()}
        if risk in ("medium", "high"):
            self.add_event("vlm", f"{risk.upper()} risk: {caption}")

    def update_learning(self, status):
        self.learning = {"status": status, "t": time.time()}
        self.add_event("learning", status)

    def update_brain(self, text):
        self.brain = {"last": text, "t": time.time()}
        self.add_event("brain", text)

    def add_event(self, source, text):
        self.events.append({"source": source, "text": text,
                            "t": round(time.time() - self.start, 1)})

    # ---- serialize ----
    def snapshot(self) -> dict:
        now = time.time()
        return {
            "uptime": round(now - self.start, 1),
            "robot": self.robot,
            "trail": list(self.trail),
            "obstacles": self.obstacles,
            "goal": self.goal,
            "predictions": {str(k): v for k, v in self.predictions.items()},
            "vlm": self.vlm,
            "learning": self.learning,
            "brain": self.brain,
            "why": self.why,
            "fleet": self.fleet,
            "health": self.health,
            "events": list(self.events)[-20:],
        }
