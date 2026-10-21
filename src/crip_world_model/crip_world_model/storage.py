#!/usr/bin/env python3
"""
crip_world_model.storage

Persistence backends for the semantic world model.

Two interchangeable backends behind a common interface:
  * PostgresStore  -- production backend (roadmap target), used when a
                      PostgreSQL connection is available.
  * SqliteStore    -- zero-config fallback so the stack runs and the test
                      suite passes without a running database server.

The node picks Postgres if `psycopg2` is importable AND a connection
succeeds; otherwise it transparently falls back to SQLite.

An "entity" is deduplicated by (label, rounded grid cell): observing the
same labelled object near a place it was already seen updates that row
(running-average position, incremented observation_count) instead of
inserting a duplicate.
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Entity:
    label: str
    category: str
    x: float
    y: float
    z: float
    confidence: float
    observation_count: int = 1
    last_seen: float = field(default_factory=time.time)
    id: int = 0


def _grid_key(x: float, y: float, cell: float = 0.5) -> str:
    """Quantise a position to a grid cell for dedup."""
    return f"{round(x / cell)}:{round(y / cell)}"


class BaseStore:
    """Common dedup / merge logic shared by both backends."""

    merge_radius = 0.75  # metres; same-label obs within this fuse into one entity

    def upsert(self, e: Entity) -> Entity:
        raise NotImplementedError

    def query(self, label: Optional[str], category: Optional[str],
              near: Optional[tuple], max_distance: float,
              max_results: int) -> List[Entity]:
        raise NotImplementedError

    def have_i_seen(self, label: str):
        matches = self.query(label, None, None, 0.0, 0)
        if not matches:
            return False, 0, None
        total = sum(m.observation_count for m in matches)
        closest = max(matches, key=lambda m: m.confidence)
        return True, total, closest

    @staticmethod
    def _merge(existing: Entity, new: Entity) -> Entity:
        """Running-average position weighted by observation count."""
        n = existing.observation_count
        existing.x = (existing.x * n + new.x) / (n + 1)
        existing.y = (existing.y * n + new.y) / (n + 1)
        existing.z = (existing.z * n + new.z) / (n + 1)
        existing.observation_count = n + 1
        existing.confidence = max(existing.confidence, new.confidence)
        existing.category = new.category or existing.category
        existing.last_seen = new.last_seen
        return existing


# --------------------------------------------------------------------------
# SQLite backend
# --------------------------------------------------------------------------
class SqliteStore(BaseStore):
    def __init__(self, path: str = ":memory:"):
        import sqlite3
        self._sqlite3 = sqlite3
        # check_same_thread=False: rclpy callbacks may run on different threads
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    category TEXT,
                    x REAL, y REAL, z REAL,
                    confidence REAL,
                    observation_count INTEGER,
                    last_seen REAL
                )""")
            self.conn.commit()

    def _row_to_entity(self, r) -> Entity:
        return Entity(label=r[1], category=r[2], x=r[3], y=r[4], z=r[5],
                      confidence=r[6], observation_count=r[7],
                      last_seen=r[8], id=r[0])

    def upsert(self, e: Entity) -> Entity:
        with self.lock:
            cur = self.conn.execute(
                "SELECT * FROM entities WHERE label=?", (e.label,))
            for row in cur.fetchall():
                ex = self._row_to_entity(row)
                if math.dist((ex.x, ex.y), (e.x, e.y)) <= self.merge_radius:
                    merged = self._merge(ex, e)
                    self.conn.execute(
                        "UPDATE entities SET x=?,y=?,z=?,confidence=?,"
                        "observation_count=?,category=?,last_seen=? WHERE id=?",
                        (merged.x, merged.y, merged.z, merged.confidence,
                         merged.observation_count, merged.category,
                         merged.last_seen, merged.id))
                    self.conn.commit()
                    return merged
            cur = self.conn.execute(
                "INSERT INTO entities(label,category,x,y,z,confidence,"
                "observation_count,last_seen) VALUES(?,?,?,?,?,?,?,?)",
                (e.label, e.category, e.x, e.y, e.z, e.confidence,
                 e.observation_count, e.last_seen))
            self.conn.commit()
            e.id = cur.lastrowid
            return e

    def query(self, label, category, near, max_distance, max_results):
        with self.lock:
            sql = "SELECT * FROM entities WHERE 1=1"
            params = []
            if label:
                sql += " AND label=?"; params.append(label)
            if category:
                sql += " AND category=?"; params.append(category)
            rows = self.conn.execute(sql, params).fetchall()
        out = [self._row_to_entity(r) for r in rows]
        if near and max_distance > 0:
            out = [e for e in out
                   if math.dist((e.x, e.y), (near[0], near[1])) <= max_distance]
            out.sort(key=lambda e: math.dist((e.x, e.y), (near[0], near[1])))
        else:
            out.sort(key=lambda e: -e.confidence)
        if max_results and max_results > 0:
            out = out[:max_results]
        return out


# --------------------------------------------------------------------------
# PostgreSQL backend
# --------------------------------------------------------------------------
class PostgresStore(BaseStore):
    def __init__(self, dsn: str):
        import psycopg2
        self.psycopg2 = psycopg2
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.lock, self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    label TEXT NOT NULL,
                    category TEXT,
                    x DOUBLE PRECISION, y DOUBLE PRECISION, z DOUBLE PRECISION,
                    confidence REAL,
                    observation_count INTEGER,
                    last_seen DOUBLE PRECISION
                )""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_label ON entities(label)")

    def _row_to_entity(self, r) -> Entity:
        return Entity(id=r[0], label=r[1], category=r[2], x=r[3], y=r[4],
                      z=r[5], confidence=r[6], observation_count=r[7],
                      last_seen=r[8])

    def upsert(self, e: Entity) -> Entity:
        with self.lock, self.conn.cursor() as cur:
            cur.execute("SELECT * FROM entities WHERE label=%s", (e.label,))
            for row in cur.fetchall():
                ex = self._row_to_entity(row)
                if math.dist((ex.x, ex.y), (e.x, e.y)) <= self.merge_radius:
                    merged = self._merge(ex, e)
                    cur.execute(
                        "UPDATE entities SET x=%s,y=%s,z=%s,confidence=%s,"
                        "observation_count=%s,category=%s,last_seen=%s WHERE id=%s",
                        (merged.x, merged.y, merged.z, merged.confidence,
                         merged.observation_count, merged.category,
                         merged.last_seen, merged.id))
                    return merged
            cur.execute(
                "INSERT INTO entities(label,category,x,y,z,confidence,"
                "observation_count,last_seen) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) "
                "RETURNING id",
                (e.label, e.category, e.x, e.y, e.z, e.confidence,
                 e.observation_count, e.last_seen))
            e.id = cur.fetchone()[0]
            return e

    def query(self, label, category, near, max_distance, max_results):
        with self.lock, self.conn.cursor() as cur:
            sql = "SELECT * FROM entities WHERE TRUE"
            params = []
            if label:
                sql += " AND label=%s"; params.append(label)
            if category:
                sql += " AND category=%s"; params.append(category)
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = [self._row_to_entity(r) for r in rows]
        if near and max_distance > 0:
            out = [e for e in out
                   if math.dist((e.x, e.y), (near[0], near[1])) <= max_distance]
            out.sort(key=lambda e: math.dist((e.x, e.y), (near[0], near[1])))
        else:
            out.sort(key=lambda e: -e.confidence)
        if max_results and max_results > 0:
            out = out[:max_results]
        return out


def make_store(logger=None) -> BaseStore:
    """Pick Postgres if reachable, else SQLite (file or memory)."""
    dsn = os.environ.get('CRIP_PG_DSN')
    if dsn:
        try:
            store = PostgresStore(dsn)
            if logger:
                logger.info('world_model: using PostgreSQL backend')
            return store
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning(f'world_model: Postgres unavailable ({exc}); '
                               'falling back to SQLite')
    path = os.environ.get('CRIP_SQLITE_PATH', ':memory:')
    if logger:
        logger.info(f'world_model: using SQLite backend ({path})')
    return SqliteStore(path)
