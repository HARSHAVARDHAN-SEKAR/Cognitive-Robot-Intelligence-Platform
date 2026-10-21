#!/usr/bin/env python3
"""
crip_dashboard.health

System health monitor (review improvement #7). Pure Python, testable.

Modules heartbeat by calling `beat(name)` (the ROS node maps topic activity to
beats). The monitor tracks per-module rate and recency and classifies each as:

    OK       -- beating recently at a reasonable rate
    SLOW     -- alive but beating well below its expected rate
    STALE    -- no beat within the stale window
    MISSING  -- never seen at all (only for modules declared `expected`)

`summary()` returns a dashboard-ready dict; overall status is the worst
module's status. Alerts are emitted once per state *change* (no spam).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

OK = "ok"
SLOW = "slow"
STALE = "stale"
MISSING = "missing"

_ORDER = {OK: 0, SLOW: 1, STALE: 2, MISSING: 3}


@dataclass
class ModuleHealth:
    beats: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    expected_hz: float = 1.0
    last_status: str = MISSING


class HealthMonitor:
    def __init__(self, stale_after: float = 5.0,
                 expected: Optional[Dict[str, float]] = None,
                 on_alert: Optional[Callable[[str, str, str], None]] = None):
        """expected: module -> expected Hz. on_alert(module, old, new)."""
        self.stale_after = stale_after
        self.modules: Dict[str, ModuleHealth] = {}
        self.on_alert = on_alert
        for name, hz in (expected or {}).items():
            self.modules[name] = ModuleHealth(expected_hz=hz)

    def beat(self, name: str, t: Optional[float] = None):
        m = self.modules.setdefault(name, ModuleHealth())
        m.beats.append(t if t is not None else time.time())

    def _status(self, m: ModuleHealth, now: float) -> str:
        if not m.beats:
            return MISSING
        if now - m.beats[-1] > self.stale_after:
            return STALE
        if len(m.beats) >= 5:
            span = m.beats[-1] - m.beats[0]
            if span > 0:
                hz = (len(m.beats) - 1) / span
                if hz < 0.3 * m.expected_hz:
                    return SLOW
        return OK

    def summary(self, now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()
        mods = {}
        worst = OK
        for name, m in self.modules.items():
            st = self._status(m, now)
            if st != m.last_status:
                if self.on_alert:
                    self.on_alert(name, m.last_status, st)
                m.last_status = st
            hz = 0.0
            if len(m.beats) >= 2:
                span = m.beats[-1] - m.beats[0]
                hz = (len(m.beats) - 1) / span if span > 0 else 0.0
            mods[name] = {"status": st, "hz": round(hz, 2),
                          "age": (round(now - m.beats[-1], 1)
                                  if m.beats else None)}
            if _ORDER[st] > _ORDER[worst]:
                worst = st
        return {"overall": worst, "modules": mods}
