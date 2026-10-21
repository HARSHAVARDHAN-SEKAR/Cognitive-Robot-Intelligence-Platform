#!/usr/bin/env python3
"""
crip_learning.registry

A tiny versioned model registry for the self-learning loop.

Layout on disk:
    <root>/
        v0001/  model.onnx  meta.json
        v0002/  model.onnx  meta.json
        current -> v0002        (symlink, or current.txt fallback on FS w/o symlinks)

Each version's meta.json records when it was trained, the eval metric, and
the dataset size. `promote` only switches `current` to a new version -- the
pipeline decides whether to call it based on the evaluation gate.

ROS-free and unit-testable.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import List, Optional


class ModelRegistry:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ----- versions -----
    def _versions(self) -> List[str]:
        return sorted(d for d in os.listdir(self.root)
                      if d.startswith("v") and
                      os.path.isdir(os.path.join(self.root, d)))

    def next_version_name(self) -> str:
        vs = self._versions()
        n = (int(vs[-1][1:]) + 1) if vs else 1
        return f"v{n:04d}"

    def add_version(self, model_src_path: str, meta: dict) -> str:
        """Copy a model file into a new version dir; return the version name."""
        name = self.next_version_name()
        vdir = os.path.join(self.root, name)
        os.makedirs(vdir, exist_ok=True)
        shutil.copy(model_src_path, os.path.join(vdir, "model.onnx"))
        meta = dict(meta)
        meta.setdefault("created", time.time())
        meta["version"] = name
        with open(os.path.join(vdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return name

    def version_meta(self, version: str) -> Optional[dict]:
        p = os.path.join(self.root, version, "meta.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    # ----- current pointer -----
    def _current_link(self):
        return os.path.join(self.root, "current")

    def _current_txt(self):
        return os.path.join(self.root, "current.txt")

    def promote(self, version: str):
        if version not in self._versions():
            raise ValueError(f"unknown version {version}")
        link = self._current_link()
        # Prefer a symlink; fall back to a text pointer on filesystems
        # that don't support symlinks.
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(version, link)
        except (OSError, NotImplementedError):
            with open(self._current_txt(), "w") as f:
                f.write(version)

    def current_version(self) -> Optional[str]:
        link = self._current_link()
        if os.path.islink(link):
            return os.readlink(link)
        if os.path.exists(self._current_txt()):
            with open(self._current_txt()) as f:
                return f.read().strip() or None
        return None

    def current_model_path(self) -> Optional[str]:
        v = self.current_version()
        if not v:
            return None
        p = os.path.join(self.root, v, "model.onnx")
        return p if os.path.exists(p) else None

    def current_meta(self) -> Optional[dict]:
        v = self.current_version()
        return self.version_meta(v) if v else None
