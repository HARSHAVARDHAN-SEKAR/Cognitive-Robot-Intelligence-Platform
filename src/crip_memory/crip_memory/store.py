#!/usr/bin/env python3
"""
crip_memory.store

Vector store for mission experiences (Milestone 8).

Two backends behind one interface, mirroring the M2 world-model storage:

  * PgVectorStore -- PostgreSQL + pgvector extension (production). Reuses the
    same CRIP_PG_DSN as the world model.
  * NumpyStore    -- in-process brute-force cosine search (offline fallback,
    optionally persisted to a .npz file). Used when Postgres/pgvector is
    unavailable, so memory works with zero setup.

A "memory" is a mission experience: text + metadata + an embedding vector.
Search returns the k most cosine-similar memories. Vectors are L2-normalized
upstream, so cosine == dot product.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Memory:
    text: str
    metadata: dict = field(default_factory=dict)
    vector: Optional[np.ndarray] = None
    score: float = 0.0
    id: int = 0
    created: float = field(default_factory=time.time)


class NumpyStore:
    backend = "numpy"

    def __init__(self, dim: int, path: Optional[str] = None):
        self.dim = dim
        self.path = path
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._meta: List[dict] = []
        self._texts: List[str] = []
        self._created: List[float] = []
        if path and os.path.exists(path):
            self._load()

    def add(self, text: str, vector: np.ndarray, metadata: dict) -> int:
        v = vector.astype(np.float32).reshape(1, -1)
        if v.shape[1] != self.dim:
            raise ValueError(f"dim mismatch {v.shape[1]} != {self.dim}")
        self._vecs = np.vstack([self._vecs, v])
        self._texts.append(text)
        self._meta.append(metadata)
        self._created.append(time.time())
        if self.path:
            self._save()
        return len(self._texts) - 1

    def search(self, vector: np.ndarray, k: int = 5) -> List[Memory]:
        if len(self._texts) == 0:
            return []
        q = vector.astype(np.float32).reshape(-1)
        sims = self._vecs @ q                      # cosine (normalized)
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            out.append(Memory(text=self._texts[i], metadata=self._meta[i],
                              score=float(sims[i]), id=int(i),
                              created=self._created[i]))
        return out

    def count(self) -> int:
        return len(self._texts)

    def _save(self):
        np.savez(self.path, vecs=self._vecs,
                 texts=np.array(self._texts, dtype=object),
                 meta=np.array([json.dumps(m) for m in self._meta], dtype=object),
                 created=np.array(self._created))

    def _load(self):
        d = np.load(self.path, allow_pickle=True)
        self._vecs = d["vecs"].astype(np.float32)
        self._texts = list(d["texts"])
        self._meta = [json.loads(m) for m in d["meta"]]
        self._created = list(d["created"])
        self.dim = self._vecs.shape[1] if self._vecs.size else self.dim


class PgVectorStore:
    backend = "pgvector"

    def __init__(self, dsn: str, dim: int, embed_backend: str):
        import psycopg2
        self.psycopg2 = psycopg2
        self.dim = dim
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        # separate table per (embed backend, dim) to avoid mixing vector spaces
        self.table = f"memories_{embed_backend}_{dim}"
        self._init_schema()

    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    metadata JSONB,
                    created DOUBLE PRECISION,
                    embedding vector({self.dim})
                )""")

    def add(self, text: str, vector: np.ndarray, metadata: dict) -> int:
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in vector.reshape(-1)) + "]"
        with self.conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table}(text, metadata, created, embedding) "
                f"VALUES (%s, %s, %s, %s) RETURNING id",
                (text, json.dumps(metadata), time.time(), vec_literal))
            return cur.fetchone()[0]

    def search(self, vector: np.ndarray, k: int = 5) -> List[Memory]:
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in vector.reshape(-1)) + "]"
        with self.conn.cursor() as cur:
            # cosine distance operator <=> ; similarity = 1 - distance
            cur.execute(
                f"SELECT id, text, metadata, created, "
                f"1 - (embedding <=> %s) AS sim "
                f"FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
                (vec_literal, vec_literal, k))
            rows = cur.fetchall()
        out = []
        for (mid, text, meta, created, sim) in rows:
            md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            out.append(Memory(text=text, metadata=md, score=float(sim),
                              id=int(mid), created=float(created or 0)))
        return out

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return int(cur.fetchone()[0])


def make_store(dim: int, embed_backend: str, logger=None):
    dsn = os.environ.get("CRIP_PG_DSN")
    if dsn:
        try:
            s = PgVectorStore(dsn, dim, embed_backend)
            if logger:
                logger.info(f"memory: pgvector store (table={s.table})")
            return s
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning(f"memory: pgvector unavailable ({exc}); "
                               "using numpy store")
    path = os.environ.get("CRIP_MEMORY_PATH")  # optional persistence
    if logger:
        logger.info(f"memory: numpy store (dim={dim}"
                    + (f", path={path}" if path else "") + ")")
    return NumpyStore(dim, path)
