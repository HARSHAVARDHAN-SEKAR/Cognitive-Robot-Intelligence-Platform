"""Unit tests for the confidence manager (improvement #2)."""
import time
from crip_safety.confidence import ConfidenceManager, ACT, DEGRADE, HOLD


def full(cm=None):
    cm = cm or ConfidenceManager()
    cm.report("perception", 0.9)
    cm.report("localization", 0.9)
    return cm


def test_missing_critical_holds():
    cm = ConfidenceManager()
    r = cm.fuse()
    assert r["policy"] == HOLD and r["speed_scale"] == 0.0


def test_high_confidence_acts():
    r = full().fuse()
    assert r["policy"] == ACT and r["speed_scale"] == 1.0


def test_weak_critical_caps_and_degrades():
    cm = full()
    cm.report("perception", 0.5)
    r = cm.fuse()
    assert r["policy"] == DEGRADE
    assert r["overall"] <= 0.5          # capped by worst critical


def test_very_low_critical_holds():
    cm = full()
    cm.report("perception", 0.2)
    assert cm.fuse()["policy"] == HOLD


def test_stale_critical_counts_as_zero():
    cm = ConfidenceManager(max_age=1.0)
    cm.report("perception", 0.95, t=time.time() - 10)
    cm.report("localization", 0.9)
    assert cm.fuse()["policy"] == HOLD


def test_weak_noncritical_lowers_but_does_not_cap():
    cm = full()
    cm.report("prediction", 0.3)
    r = cm.fuse()
    assert r["overall"] > 0.3


def test_confidence_clamped_to_unit_range():
    cm = full()
    cm.report("perception", 7.0)
    cm.report("localization", -3.0)
    r = cm.fuse()
    assert 0.0 <= r["overall"] <= 1.0
