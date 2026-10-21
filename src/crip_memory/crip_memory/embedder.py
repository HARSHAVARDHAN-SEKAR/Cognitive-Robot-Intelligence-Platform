#!/usr/bin/env python3
"""
crip_memory.embedder

Text-embedding backends for the CRIP semantic memory (Milestone 8).

Switchable via CRIP_EMBED_BACKEND, mirroring the LLM/VLM backend pattern:

  * "hashing" (default) -- a dependency-free hashed-n-gram embedder. It is NOT
    a dummy: hashing character/word n-grams into a fixed vector and
    L2-normalizing yields real lexical similarity, so "inspect the fire
    extinguisher" retrieves "check fire extinguishers" far above unrelated
    text. Runs anywhere, deterministic, no model download.
  * "local"  -- a sentence-transformers model (semantic similarity).
  * "api"    -- a hosted embedding endpoint (OpenAI-compatible /v1/embeddings,
    configurable via CRIP_EMBED_URL / CRIP_EMBED_MODEL / CRIP_EMBED_KEY).

All backends return L2-normalized float32 vectors, so cosine similarity is a
plain dot product. The store records which backend/dim produced a vector;
mixing backends is avoided by keying the store on (backend, dim).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional

import numpy as np


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BaseEmbedder:
    name = "base"
    dim = 0

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_many(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts \
            else np.zeros((0, self.dim), dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    """Hashed word + character-trigram n-grams -> fixed vector."""
    name = "hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode()).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    def _sign(self, token: str) -> float:
        h = hashlib.md5(token.encode()).digest()
        return 1.0 if h[4] & 1 else -1.0

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        # unigrams + bigrams
        grams = list(toks)
        grams += [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
        # character trigrams (captures morphology / typos)
        joined = " ".join(toks)
        grams += [joined[i:i+3] for i in range(max(0, len(joined) - 2))]
        for g in grams:
            idx = self._hash(g)
            vec[idx] += self._sign(g)
        return _l2norm(vec)


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers model (semantic). Heavy; opt-in."""
    name = "local"

    def __init__(self, model: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model or os.environ.get(
            "CRIP_EMBED_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def embed_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


class ApiEmbedder(BaseEmbedder):
    """OpenAI-compatible /v1/embeddings endpoint."""
    name = "api"

    def __init__(self):
        self.url = os.environ.get("CRIP_EMBED_URL",
                                  "https://api.openai.com/v1/embeddings")
        self.model = os.environ.get("CRIP_EMBED_MODEL", "text-embedding-3-small")
        self.key = os.environ.get("CRIP_EMBED_KEY") or os.environ.get(
            "OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("CRIP_EMBED_KEY/OPENAI_API_KEY not set")
        # dim is discovered on first call
        self.dim = int(os.environ.get("CRIP_EMBED_DIM", "1536"))

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: List[str]) -> np.ndarray:
        import json
        import urllib.request
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        vecs = [np.asarray(d["embedding"], dtype=np.float32)
                for d in data["data"]]
        self.dim = len(vecs[0])
        return np.stack([_l2norm(v) for v in vecs])


def make_embedder(logger=None) -> BaseEmbedder:
    backend = os.environ.get("CRIP_EMBED_BACKEND", "hashing").lower()
    try:
        if backend == "local":
            e = LocalEmbedder()
            if logger:
                logger.info(f"memory: local embedder ({e.model_name}, dim={e.dim})")
            return e
        if backend == "api":
            e = ApiEmbedder()
            if logger:
                logger.info(f"memory: api embedder ({e.model}, dim={e.dim})")
            return e
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning(f"memory: backend '{backend}' unavailable ({exc}); "
                           "falling back to hashing")
    e = HashingEmbedder()
    if logger:
        logger.info(f"memory: hashing embedder (dim={e.dim})")
    return e
