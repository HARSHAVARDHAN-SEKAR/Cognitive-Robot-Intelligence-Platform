"""Unit tests for the semantic memory embedder + vector store (no ROS)."""
import os
import tempfile

import numpy as np

from crip_memory.embedder import HashingEmbedder, make_embedder
from crip_memory.store import NumpyStore


def test_hashing_embedder_normalized_and_deterministic():
    e = HashingEmbedder()
    v = e.embed("go to the charging station")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    assert np.allclose(e.embed("hello world"), e.embed("hello world"))


def test_hashing_similarity_ordering():
    e = HashingEmbedder()

    def sim(a, b):
        return float(np.dot(e.embed(a), e.embed(b)))

    q = "inspect the fire extinguisher on floor two"
    rel = "check fire extinguishers on the second floor"
    un = "deliver coffee to the kitchen"
    assert sim(q, rel) > sim(q, un)
    assert abs(sim(q, q) - 1.0) < 1e-5


def test_default_backend_is_hashing(monkeypatch):
    monkeypatch.delenv("CRIP_EMBED_BACKEND", raising=False)
    assert make_embedder().name == "hashing"


def test_unknown_backend_falls_back(monkeypatch):
    monkeypatch.setenv("CRIP_EMBED_BACKEND", "does_not_exist")
    assert make_embedder().name == "hashing"


def test_store_add_and_search():
    e = HashingEmbedder()
    s = NumpyStore(dim=e.dim)
    items = [
        "inspected fire extinguisher on floor two",
        "delivered package to the loading dock",
        "checked fire safety equipment second floor",
    ]
    for t in items:
        s.add(t, e.embed(t), {"k": t[:4]})
    assert s.count() == 3
    res = s.search(e.embed("check the fire extinguishers"), k=2)
    assert len(res) == 2
    assert "fire" in res[0].text
    # scores must be sorted descending
    assert res[0].score >= res[1].score


def test_store_dim_mismatch_raises():
    s = NumpyStore(dim=8)
    import pytest
    with pytest.raises(ValueError):
        s.add("x", np.zeros(4, dtype=np.float32), {})


def test_store_persistence_roundtrip():
    e = HashingEmbedder()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.npz")
        s = NumpyStore(dim=e.dim, path=path)
        for t in ["alpha task done", "beta task failed", "gamma patrol"]:
            s.add(t, e.embed(t), {"t": t})
        s2 = NumpyStore(dim=e.dim, path=path)
        assert s2.count() == 3
        res = s2.search(e.embed("beta"), k=1)
        assert "beta" in res[0].text


def test_search_empty_store():
    e = HashingEmbedder()
    s = NumpyStore(dim=e.dim)
    assert s.search(e.embed("anything"), k=5) == []
