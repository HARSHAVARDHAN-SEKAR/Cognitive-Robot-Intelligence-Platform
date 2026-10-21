"""Unit tests for the VLM core (no ROS, no network)."""
import io

import pytest

from crip_vlm.vlm import (
    MockVLM, make_vlm, _extract_json, _assessment_from_dict, VLMError,
    RISK_LEVELS)


def _png(color):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(b, "PNG")
    return b.getvalue()


def test_mock_clear_scene_low_risk():
    a = MockVLM().analyze(_png((120, 120, 120)), "image/png")
    assert a.risk_level in RISK_LEVELS
    assert a.backend == "mock"
    assert a.labels  # non-empty


def test_mock_red_region_flags_medium():
    a = MockVLM().analyze(_png((220, 30, 30)), "image/png")
    assert "red_object" in a.labels
    assert a.risk_level == "medium"


def test_mock_dark_scene_flags_low_light():
    a = MockVLM().analyze(_png((8, 8, 8)), "image/png")
    assert "low_light" in a.labels


def test_extract_json_fenced_with_prose():
    d = _extract_json('ok:\n```json\n{"caption":"c","labels":["person"],'
                      '"risk_level":"high","rationale":"r",'
                      '"recommendation":"stop","confidence":0.8}\n```')
    a = _assessment_from_dict(d, "x")
    assert a.risk_level == "high"
    assert a.labels == ["person"]
    assert a.confidence == 0.8


def test_invalid_risk_normalized():
    a = _assessment_from_dict({"risk_level": "catastrophic"}, "x")
    assert a.risk_level == "none"


def test_confidence_clamped():
    a = _assessment_from_dict({"confidence": 5.0}, "x")
    assert a.confidence == 1.0
    b = _assessment_from_dict({"confidence": -2}, "x")
    assert b.confidence == 0.0


def test_no_json_raises():
    with pytest.raises(VLMError):
        _extract_json("the model said no")


def test_default_backend_is_mock(monkeypatch):
    monkeypatch.delenv("CRIP_VLM_BACKEND", raising=False)
    assert make_vlm().__class__.__name__ == "MockVLM"


def test_anthropic_without_key_falls_back(monkeypatch):
    monkeypatch.setenv("CRIP_VLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_vlm().__class__.__name__ == "MockVLM"
