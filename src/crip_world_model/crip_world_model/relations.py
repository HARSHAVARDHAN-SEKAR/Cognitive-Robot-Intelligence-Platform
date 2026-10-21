#!/usr/bin/env python3
"""
crip_world_model.relations

Knowledge-graph layer over the semantic world model (review improvement #4).

The base world model stores entities (label, category, position). This layer
adds typed *relations* between them:

    bottle --on--> table --in--> kitchen --in--> building_a

Backed by a simple triple store (subject, predicate, object) that works both
in-memory and persisted alongside the entity DB. Supports:

  * add/query relations, with inverse lookup
  * automatic spatial inference: entities within `near_radius` of each other
    get a symmetric `near` relation derived from their positions
  * multi-hop containment queries: contents_of("kitchen") walks in/on edges
    transitively, so "what is in the kitchen" includes the bottle on the table.

Pure Python; the world-model node exposes it via the existing services.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

Triple = Tuple[str, str, str]

# predicates whose transitive closure means "inside"
CONTAINMENT = {"in", "on", "inside", "part_of"}


class RelationStore:
    def __init__(self, near_radius: float = 1.0):
        self.near_radius = near_radius
        self._triples: Set[Triple] = set()
        self._by_subject: Dict[str, Set[Triple]] = defaultdict(set)
        self._by_object: Dict[str, Set[Triple]] = defaultdict(set)

    # ---- edit ----
    def add(self, subject: str, predicate: str, obj: str):
        t = (subject.strip(), predicate.strip(), obj.strip())
        if t in self._triples or t[0] == t[2]:
            return
        self._triples.add(t)
        self._by_subject[t[0]].add(t)
        self._by_object[t[2]].add(t)

    def remove(self, subject: str, predicate: str, obj: str):
        t = (subject, predicate, obj)
        self._triples.discard(t)
        self._by_subject[subject].discard(t)
        self._by_object[obj].discard(t)

    # ---- query ----
    def relations_of(self, entity: str) -> List[Triple]:
        return sorted(self._by_subject[entity] | self._by_object[entity])

    def objects(self, subject: str, predicate: str) -> List[str]:
        return sorted(o for (s, p, o) in self._by_subject[subject]
                      if p == predicate)

    def subjects(self, predicate: str, obj: str) -> List[str]:
        return sorted(s for (s, p, o) in self._by_object[obj]
                      if p == predicate)

    def contents_of(self, container: str, max_depth: int = 6) -> List[str]:
        """Everything transitively in/on the container."""
        found: Set[str] = set()
        frontier = {container}
        for _ in range(max_depth):
            step: Set[str] = set()
            for c in frontier:
                for (s, p, o) in self._by_object[c]:
                    if p in CONTAINMENT and s not in found:
                        step.add(s)
            if not step:
                break
            found |= step
            frontier = step
        return sorted(found)

    def where_is(self, entity: str, max_depth: int = 6) -> List[str]:
        """Chain of containers outward: bottle -> [table, kitchen, ...]."""
        chain: List[str] = []
        cur = entity
        for _ in range(max_depth):
            nxt = [o for (s, p, o) in self._by_subject[cur]
                   if p in CONTAINMENT]
            if not nxt:
                break
            cur = nxt[0]
            if cur in chain:
                break
            chain.append(cur)
        return chain

    # ---- spatial inference ----
    def infer_near(self, positions: Dict[str, Tuple[float, float]]):
        """Derive symmetric `near` relations for entities within radius.
        positions: label -> (x, y). Re-derives from scratch each call."""
        # drop old inferred near edges
        for t in [t for t in self._triples if t[1] == "near"]:
            self.remove(*t)
        labels = list(positions.keys())
        for i, a in enumerate(labels):
            ax, ay = positions[a]
            for b in labels[i + 1:]:
                bx, by = positions[b]
                if math.hypot(ax - bx, ay - by) <= self.near_radius:
                    self.add(a, "near", b)
                    self.add(b, "near", a)

    # ---- serialize ----
    def to_lines(self) -> List[str]:
        return [f"{s} {p} {o}" for (s, p, o) in sorted(self._triples)]

    def load_lines(self, lines: List[str]):
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                self.add(*parts)
