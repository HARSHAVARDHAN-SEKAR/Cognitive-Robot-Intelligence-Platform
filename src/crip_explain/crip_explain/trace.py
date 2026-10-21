#!/usr/bin/env python3
"""
crip_explain.trace

Explainable-AI core (Module 12): turn the stack's scattered signals into a
coherent answer to "why did you do that?". Pure-Python, unit-testable.

Every subsystem already emits evidence -- the brain explains its plan, the VLM
gives a risk rationale, the predictor forecasts a person's path, the safety
supervisor logs a veto. This module collects those as timestamped, typed
`Event`s in a ring buffer and, on request, assembles the most relevant recent
chain into a causal narrative:

    "Stopped because: safety vetoed forward motion (frontal obstacle 0.22m);
     vision flagged HIGH risk (person in path); a tracked person is predicted
     to cross ahead."

The selection favours the most decision-relevant events (vetoes and high-risk
assessments outrank routine status) and the most recent, so the explanation
reflects what actually drove the last action.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


# event source -> base salience (higher = more decision-relevant)
SALIENCE = {
    "safety": 5,
    "vlm": 4,
    "prediction": 3,
    "brain": 2,
    "navigation": 2,
    "learning": 1,
}


@dataclass
class Event:
    source: str
    text: str
    t: float = field(default_factory=time.time)
    decisive: bool = False     # set for vetoes / high-risk / failures


class ExplanationTrace:
    def __init__(self, maxlen: int = 200):
        self.events: Deque[Event] = deque(maxlen=maxlen)

    def add(self, source: str, text: str, decisive: bool = False):
        self.events.append(Event(source=source, text=text, decisive=decisive))

    def _score(self, e: Event, now: float) -> float:
        base = SALIENCE.get(e.source, 1)
        if e.decisive:
            base += 5
        # recency: decay over ~30s
        age = max(0.0, now - e.t)
        recency = max(0.0, 1.0 - age / 30.0)
        return base * (0.4 + 0.6 * recency)

    def why(self, top_k: int = 3, window_s: float = 30.0) -> dict:
        """Assemble a causal explanation from recent events."""
        now = time.time()
        recent = [e for e in self.events if now - e.t <= window_s]
        if not recent:
            return {"summary": "No recent decisions to explain.",
                    "factors": []}
        ranked = sorted(recent, key=lambda e: self._score(e, now), reverse=True)
        chosen = ranked[:top_k]
        # present in chronological order for a readable narrative
        chosen.sort(key=lambda e: e.t)
        factors = [{"source": e.source, "text": e.text,
                    "decisive": e.decisive} for e in chosen]

        # headline: the single most salient (usually a veto or high risk)
        head = max(recent, key=lambda e: self._score(e, now))
        verb = self._headline_verb(head)
        summary = f"{verb} because: " + "; ".join(f["text"] for f in factors)
        return {"summary": summary, "factors": factors,
                "primary_cause": {"source": head.source, "text": head.text}}

    @staticmethod
    def _headline_verb(e: Event) -> str:
        if e.source == "safety" and e.decisive:
            return "Stopped"
        if e.source == "vlm" and e.decisive:
            return "Slowed / rerouted"
        if e.source == "navigation":
            return "Navigated"
        if e.source == "brain":
            return "Acted"
        return "Responded"
