#!/usr/bin/env python3
"""
crip_vlm.vlm

Vision-Language reasoning core for CRIP (Milestone 4).

Takes an encoded image (PNG/JPEG bytes) and returns a structured scene
assessment: caption, observed labels, a risk level, rationale, and a
recommended action. This is the document's Module 4 ("What is happening? ->
Oil leak -> Risk = High -> Recommend reroute -> Notify operator").

Three interchangeable backends, chosen by CRIP_VLM_BACKEND:

  * "anthropic" -- Anthropic Messages API with an image block (vision).
  * "ollama"    -- a local vision model (LLaVA / Qwen-VL) via Ollama.
  * "mock"      -- offline, deterministic. Derives a coarse assessment from
                   simple image statistics so the pipeline runs with no model
                   and no GPU. Default.

All backends return the SAME dict schema:

  {
    "caption": "...",
    "labels": ["...", ...],
    "risk_level": "none|low|medium|high",
    "rationale": "...",
    "recommendation": "...",
    "confidence": 0.0-1.0
  }
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


RISK_LEVELS = ("none", "low", "medium", "high")

SYSTEM_PROMPT = """\
You are the vision system for an autonomous mobile robot. Look at the image
and assess the scene for navigation and safety. Respond with ONLY a JSON
object, no prose, no markdown fences.

Schema:
{
  "caption": "<one sentence describing what you see>",
  "labels": ["<object or condition>", ...],
  "risk_level": "<none|low|medium|high>",
  "rationale": "<why this risk level>",
  "recommendation": "<suggested action, e.g. reroute / notify operator / proceed>",
  "confidence": <0.0-1.0>
}

Consider hazards a ground robot cares about: people in the path, spills,
fire/smoke, blocked routes, fragile objects, stairs, wet floors."""


@dataclass
class Assessment:
    caption: str = ""
    labels: List[str] = field(default_factory=list)
    risk_level: str = "none"
    rationale: str = ""
    recommendation: str = ""
    confidence: float = 0.0
    backend: str = ""
    raw: str = ""


class VLMError(Exception):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise VLMError(f"no JSON object in VLM output: {text!r}")
    return json.loads(text[start:end + 1])


def _assessment_from_dict(d: dict, backend: str, raw: str = "") -> Assessment:
    risk = str(d.get("risk_level", "none")).strip().lower()
    if risk not in RISK_LEVELS:
        risk = "none"
    labels = d.get("labels", []) or []
    if not isinstance(labels, list):
        labels = [str(labels)]
    conf = d.get("confidence", 0.0)
    try:
        conf = max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.0
    return Assessment(
        caption=str(d.get("caption", "")).strip(),
        labels=[str(x).strip() for x in labels if str(x).strip()],
        risk_level=risk,
        rationale=str(d.get("rationale", "")).strip(),
        recommendation=str(d.get("recommendation", "")).strip(),
        confidence=conf,
        backend=backend,
        raw=raw,
    )


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class BaseVLM:
    def analyze(self, image_bytes: bytes, media_type: str,
                question: str = "") -> Assessment:
        raise NotImplementedError


class MockVLM(BaseVLM):
    """Offline heuristic 'vision'. It cannot truly see, but derives a stable,
    plausible assessment from cheap image statistics so the whole pipeline is
    runnable and testable without a model or GPU."""

    def analyze(self, image_bytes, media_type, question="") -> Assessment:
        stats = self._image_stats(image_bytes)
        labels = ["floor", "open_space"]
        risk, rationale, rec = "none", "scene appears clear", "proceed"

        # Heuristics on average channel intensities (purely illustrative).
        if stats:
            r, g, b, brightness = stats
            if r > g + 40 and r > b + 40:
                labels.append("red_object")
                risk, rationale = "medium", "prominent red region (possible hazard marker)"
                rec = "slow and assess"
            if brightness < 40:
                labels.append("low_light")
                if risk == "none":
                    risk, rationale = "low", "low light reduces perception confidence"
                    rec = "reduce speed"
        caption = "Heuristic scene view: " + ", ".join(labels)
        conf = 0.35  # mock is never very confident
        if question:
            caption += f" (re: {question})"
        return Assessment(caption=caption, labels=labels, risk_level=risk,
                          rationale=rationale, recommendation=rec,
                          confidence=conf, backend="mock",
                          raw="(mock backend)")

    @staticmethod
    def _image_stats(image_bytes):
        """Return (avgR, avgG, avgB, brightness) if Pillow is available, else
        None. Kept optional so mock works even with no imaging libs."""
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((32, 32))
            raw = img.tobytes()  # RGBRGB... , 32*32*3 bytes
            n = len(raw) // 3
            r = sum(raw[0::3]) / n
            g = sum(raw[1::3]) / n
            b = sum(raw[2::3]) / n
            return r, g, b, (r + g + b) / 3.0
        except Exception:  # noqa: BLE001
            return None


class AnthropicVLM(BaseVLM):
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("CRIP_VLM_MODEL",
                                             "claude-sonnet-4-6")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise VLMError("ANTHROPIC_API_KEY not set")

    def analyze(self, image_bytes, media_type, question="") -> Assessment:
        import urllib.request
        b64 = base64.b64encode(image_bytes).decode()
        user_content = [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type,
                                         "data": b64}},
            {"type": "text",
             "text": question or "Assess this scene for the robot."},
        ]
        body = json.dumps({
            "model": self.model,
            "max_tokens": 512,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_content}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"content-type": "application/json",
                     "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        return _assessment_from_dict(_extract_json(text), "anthropic", text)


class OllamaVLM(BaseVLM):
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.model = model or os.environ.get("CRIP_VLM_MODEL", "llava")
        self.host = host or os.environ.get("OLLAMA_HOST",
                                           "http://127.0.0.1:11434")

    def analyze(self, image_bytes, media_type, question="") -> Assessment:
        import urllib.request
        b64 = base64.b64encode(image_bytes).decode()
        body = json.dumps({
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": question or "Assess this scene for the robot.",
            "images": [b64],
            "stream": False,
            "format": "json",
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        text = data.get("response", "")
        return _assessment_from_dict(_extract_json(text), "ollama", text)


def make_vlm(logger=None) -> BaseVLM:
    backend = os.environ.get("CRIP_VLM_BACKEND", "mock").lower()
    try:
        if backend == "anthropic":
            v = AnthropicVLM()
            if logger:
                logger.info("vlm: using Anthropic vision backend")
            return v
        if backend == "ollama":
            v = OllamaVLM()
            if logger:
                logger.info(f"vlm: using Ollama backend ({v.model})")
            return v
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning(f"vlm: backend '{backend}' unavailable ({exc}); "
                           "falling back to mock")
    if logger:
        logger.info("vlm: using mock (offline) backend")
    return MockVLM()
