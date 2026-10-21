"""Unit tests for the LLM planner (no ROS, no network)."""
import os

import pytest

from crip_llm_brain.planner import (
    MockPlanner, make_planner, _extract_json, _plan_from_dict, PlannerError)


KNOWN = ["charging_station", "obstacle", "person"]


def test_mock_navigate_and_wait():
    p = MockPlanner().plan(
        "Go to the charging station and wait 5 seconds", KNOWN)
    actions = [(t.action, t.target_label, t.duration) for t in p.tasks]
    assert ("navigate_to", "charging_station", 0.0) in actions
    assert any(a == "wait" and d == 5.0 for a, _, d in actions)


def test_mock_does_not_hallucinate_unknown_place():
    p = MockPlanner().plan("go to the helipad", KNOWN)
    # must NOT emit a navigate_to with an unknown label
    assert all(not (t.action == "navigate_to" and t.target_label == "helipad")
               for t in p.tasks)


def test_extract_json_fenced():
    d = _extract_json('```json\n{"tasks":[{"action":"wait","duration":3}],'
                      '"summary":"s"}\n```')
    plan = _plan_from_dict(d)
    assert plan.tasks[0].action == "wait"
    assert plan.tasks[0].duration == 3.0


def test_extract_json_with_prose():
    d = _extract_json('Sure! Here is the plan: {"tasks":'
                      '[{"action":"report","target_label":"person"}],'
                      '"summary":"report"} Hope that helps.')
    plan = _plan_from_dict(d)
    assert plan.tasks[0].action == "report"


def test_invalid_action_rejected():
    with pytest.raises(PlannerError):
        _plan_from_dict({"tasks": [{"action": "self_destruct"}]})


def test_empty_plan_rejected():
    with pytest.raises(PlannerError):
        _plan_from_dict({"tasks": []})


def test_default_backend_is_mock(monkeypatch):
    monkeypatch.delenv("CRIP_LLM_BACKEND", raising=False)
    assert make_planner().__class__.__name__ == "MockPlanner"


def test_unknown_backend_falls_back_to_mock(monkeypatch):
    monkeypatch.setenv("CRIP_LLM_BACKEND", "does_not_exist")
    assert make_planner().__class__.__name__ == "MockPlanner"


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.setenv("CRIP_LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # no key -> falls back to mock rather than crashing
    assert make_planner().__class__.__name__ == "MockPlanner"
