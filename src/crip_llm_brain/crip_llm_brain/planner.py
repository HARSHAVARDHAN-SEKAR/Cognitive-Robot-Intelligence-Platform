#!/usr/bin/env python3
"""
crip_llm_brain.planner

The reasoning core of the CRIP LLM Brain (Milestone 3).

A mission like "go to the charging station, wait 5 seconds, then come back"
is turned into a structured plan -- a list of typed tasks -- by an LLM. The
LLM never drives motors; it only emits a plan, which a downstream executor
runs through the safety-supervised controller.

Three interchangeable backends behind one interface, chosen by the
CRIP_LLM_BACKEND environment variable:

  * "anthropic" -- Anthropic Messages API (needs ANTHROPIC_API_KEY).
  * "ollama"    -- a local model (Llama/Qwen) via the Ollama HTTP API.
  * "mock"      -- deterministic, offline rule-based planner. Default, so the
                   stack runs and tests pass with no API key and no GPU.

All backends return the SAME JSON plan schema:

  {
    "tasks": [
      {"action": "navigate_to", "target_label": "charging_station",
       "duration": 0, "note": "drive to the charging station"},
      {"action": "wait", "target_label": "", "duration": 5.0,
       "note": "pause on arrival"}
    ],
    "summary": "Go to the charging station and wait."
  }

Grounding (resolving target_label -> map coordinates against the world model)
happens in the node, NOT here -- this module is pure planning and is unit
testable without ROS.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


VALID_ACTIONS = {"navigate_to", "wait", "report", "search_for"}

SYSTEM_PROMPT = """\
You are the task planner for an autonomous mobile robot. Convert the user's
mission into a JSON plan. Respond with ONLY a JSON object, no prose, no
markdown fences.

Schema:
{
  "tasks": [
    {"action": "<navigate_to|wait|report|search_for>",
     "target_label": "<semantic label or empty string>",
     "duration": <seconds, number>,
     "note": "<short human-readable description>"}
  ],
  "summary": "<one sentence describing the plan>"
}

Rules:
- navigate_to requires target_label (a place/object name). duration is 0.
- wait requires duration in seconds; target_label is empty.
- report and search_for use target_label as the subject.
- Use only labels the user names; do not invent locations.
- Keep the plan minimal and ordered.

Known semantic labels currently in the robot's memory: {known_labels}
"""


@dataclass
class PlanTask:
    action: str
    target_label: str = ""
    duration: float = 0.0
    note: str = ""


@dataclass
class Plan:
    tasks: List[PlanTask] = field(default_factory=list)
    summary: str = ""
    raw: str = ""            # raw model output, for debugging


class PlannerError(Exception):
    pass


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating fences."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # else find the outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise PlannerError(f"no JSON object in model output: {text!r}")
    return json.loads(text[start:end + 1])


def _plan_from_dict(d: dict, raw: str = "") -> Plan:
    tasks = []
    for t in d.get("tasks", []):
        action = str(t.get("action", "")).strip()
        if action not in VALID_ACTIONS:
            raise PlannerError(f"invalid action {action!r}")
        tasks.append(PlanTask(
            action=action,
            target_label=str(t.get("target_label", "")).strip(),
            duration=float(t.get("duration", 0) or 0),
            note=str(t.get("note", "")).strip(),
        ))
    if not tasks:
        raise PlannerError("plan contains no tasks")
    return Plan(tasks=tasks, summary=str(d.get("summary", "")).strip(), raw=raw)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class BasePlanner:
    def plan(self, mission: str, known_labels: List[str]) -> Plan:
        raise NotImplementedError


class MockPlanner(BasePlanner):
    """Deterministic offline planner. Good enough to demo and test the whole
    pipeline without any model. Parses a few common mission shapes."""

    def plan(self, mission: str, known_labels: List[str]) -> Plan:
        m = mission.lower()
        tasks: List[PlanTask] = []

        # find any known label mentioned in the mission
        mentioned = [lbl for lbl in known_labels if lbl.replace("_", " ") in m
                     or lbl in m]

        # "wait N seconds"
        wait_match = re.search(r"wait\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?", m)

        if "go to" in m or "navigate" in m or "inspect" in m or "visit" in m:
            if mentioned:
                for lbl in mentioned:
                    verb = "inspect" if "inspect" in m else "navigate to"
                    tasks.append(PlanTask("navigate_to", lbl, 0.0,
                                          f"{verb} the {lbl.replace('_',' ')}"))
            else:
                # nothing known -> a search task the node may reject/handle
                tasks.append(PlanTask("search_for", "", 0.0,
                                      "explore to find the requested target"))

        if wait_match:
            tasks.append(PlanTask("wait", "", float(wait_match.group(1)),
                                  f"wait {wait_match.group(1)} seconds"))

        if "return" in m or "come back" in m or "charging" in m and "go to" not in m:
            if "charging_station" in known_labels:
                tasks.append(PlanTask("navigate_to", "charging_station", 0.0,
                                      "return to the charging station"))

        if "report" in m or "tell me" in m:
            subj = mentioned[0] if mentioned else ""
            tasks.append(PlanTask("report", subj, 0.0, "report findings"))

        if not tasks:
            # fallback: treat the whole mission as a search
            tasks.append(PlanTask("search_for", "", 0.0,
                                  f"interpret mission: {mission}"))

        summary = "Mock plan: " + "; ".join(t.note for t in tasks)
        return Plan(tasks=tasks, summary=summary, raw="(mock backend)")


class AnthropicPlanner(BasePlanner):
    """Cloud backend via the Anthropic Messages API."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("CRIP_LLM_MODEL",
                                             "claude-sonnet-4-6")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise PlannerError("ANTHROPIC_API_KEY not set")

    def plan(self, mission: str, known_labels: List[str]) -> Plan:
        import urllib.request
        sys_prompt = SYSTEM_PROMPT.format(
            known_labels=", ".join(known_labels) or "(none yet)")
        body = json.dumps({
            "model": self.model,
            "max_tokens": 1024,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": mission}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = "".join(blk.get("text", "") for blk in data.get("content", [])
                       if blk.get("type") == "text")
        return _plan_from_dict(_extract_json(text), raw=text)


class OllamaPlanner(BasePlanner):
    """Local backend via the Ollama HTTP API (Llama/Qwen/etc)."""

    def __init__(self, model: Optional[str] = None,
                 host: Optional[str] = None):
        self.model = model or os.environ.get("CRIP_LLM_MODEL", "llama3.1")
        self.host = host or os.environ.get("OLLAMA_HOST",
                                           "http://127.0.0.1:11434")

    def plan(self, mission: str, known_labels: List[str]) -> Plan:
        import urllib.request
        sys_prompt = SYSTEM_PROMPT.format(
            known_labels=", ".join(known_labels) or "(none yet)")
        body = json.dumps({
            "model": self.model,
            "system": sys_prompt,
            "prompt": mission,
            "stream": False,
            "format": "json",
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        text = data.get("response", "")
        return _plan_from_dict(_extract_json(text), raw=text)


def make_planner(logger=None) -> BasePlanner:
    """Select a backend from CRIP_LLM_BACKEND, with safe fallback to mock."""
    backend = os.environ.get("CRIP_LLM_BACKEND", "mock").lower()
    try:
        if backend == "anthropic":
            p = AnthropicPlanner()
            if logger:
                logger.info("llm_brain: using Anthropic backend")
            return p
        if backend == "ollama":
            p = OllamaPlanner()
            if logger:
                logger.info(f"llm_brain: using Ollama backend ({p.model})")
            return p
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning(f"llm_brain: backend '{backend}' unavailable "
                           f"({exc}); falling back to mock")
    if logger:
        logger.info("llm_brain: using mock (offline) backend")
    return MockPlanner()
