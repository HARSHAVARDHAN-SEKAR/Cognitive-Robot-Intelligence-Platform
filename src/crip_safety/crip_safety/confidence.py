#!/usr/bin/env python3
"""
crip_safety.confidence

Confidence-aware decision making (review improvement #2). Pure Python.

Every AI module reports its latest confidence (0..1) with a timestamp:
    perception 0.63, llm_plan 0.91, prediction 0.55, localization 0.88 ...

The ConfidenceManager fuses them into an overall action confidence and maps it
to a policy:

    ACT      -- confidence high: proceed at full capability
    DEGRADE  -- confidence moderate: proceed slowly (scale speed)
    HOLD     -- confidence low or a critical source stale/missing: stop and wait

Fusion rules (deliberately conservative, suitable for a safety context):
  * overall = weighted mean of fresh sources, but hard-capped by the WORST
    critical source (a chain is as strong as its weakest critical link).
  * a stale critical source (older than max_age) counts as 0 confidence.
  * missing critical sources -> HOLD.

The supervisor uses the resulting speed_scale as one more input; this never
*raises* what safety rules allow, only lowers it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ACT = "act"
DEGRADE = "degrade"
HOLD = "hold"


@dataclass
class SourceReport:
    confidence: float
    t: float


class ConfidenceManager:
    def __init__(self,
                 critical: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 act_threshold: float = 0.75,
                 degrade_threshold: float = 0.45,
                 max_age: float = 3.0):
        self.critical = set(critical or ["perception", "localization"])
        self.weights = weights or {}
        self.act_threshold = act_threshold
        self.degrade_threshold = degrade_threshold
        self.max_age = max_age
        self._reports: Dict[str, SourceReport] = {}

    # ---- intake ----
    def report(self, source: str, confidence: float, t: Optional[float] = None):
        confidence = max(0.0, min(1.0, float(confidence)))
        self._reports[source] = SourceReport(confidence, t if t is not None
                                             else time.time())

    # ---- fusion ----
    def _fresh_confidence(self, source: str, now: float) -> Optional[float]:
        rep = self._reports.get(source)
        if rep is None:
            return None
        if now - rep.t > self.max_age:
            return 0.0  # stale counts as zero confidence
        return rep.confidence

    def fuse(self, now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()

        # missing critical source -> immediate HOLD
        missing = [s for s in self.critical if s not in self._reports]
        if missing:
            return {"overall": 0.0, "policy": HOLD, "speed_scale": 0.0,
                    "reason": f"missing critical source(s): {sorted(missing)}",
                    "sources": {}}

        fresh: Dict[str, float] = {}
        for s in self._reports:
            c = self._fresh_confidence(s, now)
            if c is not None:
                fresh[s] = c

        # weighted mean over all fresh sources
        wsum = 0.0
        total = 0.0
        for s, c in fresh.items():
            w = self.weights.get(s, 1.0)
            wsum += w
            total += w * c
        mean = (total / wsum) if wsum > 0 else 0.0

        # hard cap by worst critical source
        worst_critical = min(fresh[s] for s in self.critical)
        overall = min(mean, worst_critical)

        if overall >= self.act_threshold:
            policy, scale = ACT, 1.0
        elif overall >= self.degrade_threshold:
            policy, scale = DEGRADE, 0.5
        else:
            policy, scale = HOLD, 0.0

        reason = (f"overall={overall:.2f} "
                  f"(mean={mean:.2f}, worst_critical={worst_critical:.2f})")
        return {"overall": overall, "policy": policy, "speed_scale": scale,
                "reason": reason, "sources": dict(fresh)}
