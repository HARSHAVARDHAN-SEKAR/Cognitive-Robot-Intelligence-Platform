"""Unit tests for the health monitor (improvement #7)."""
from crip_dashboard.health import HealthMonitor, OK, SLOW, STALE, MISSING


def test_ok_when_beating_at_rate():
    hm = HealthMonitor(expected={"m": 10.0})
    for i in range(20):
        hm.beat("m", 1000.0 + i * 0.1)
    assert hm.summary(now=1002.0)["modules"]["m"]["status"] == OK


def test_slow_when_far_below_expected_rate():
    hm = HealthMonitor(expected={"m": 20.0})
    for i in range(10):
        hm.beat("m", 1000.0 + i * 0.5)   # 2 Hz vs 20 expected
    assert hm.summary(now=1005.0)["modules"]["m"]["status"] == SLOW


def test_stale_when_no_recent_beat():
    hm = HealthMonitor(stale_after=5.0, expected={"m": 1.0})
    hm.beat("m", 100.0)
    assert hm.summary(now=200.0)["modules"]["m"]["status"] == STALE


def test_missing_when_never_beaten():
    hm = HealthMonitor(expected={"m": 1.0})
    s = hm.summary()
    assert s["modules"]["m"]["status"] == MISSING
    assert s["overall"] == MISSING


def test_overall_is_worst():
    hm = HealthMonitor(stale_after=5.0, expected={"good": 1.0, "bad": 1.0})
    for i in range(6):
        hm.beat("good", 1000.0 + i)
    hm.beat("bad", 100.0)
    assert hm.summary(now=1006.0)["overall"] == STALE


def test_alert_only_on_change():
    alerts = []
    hm = HealthMonitor(expected={"m": 1.0},
                       on_alert=lambda n, o, s: alerts.append((n, o, s)))
    for i in range(6):
        hm.beat("m", 1000.0 + i)
    hm.summary(now=1006.0)
    n = len(alerts)
    hm.summary(now=1006.1)
    hm.summary(now=1006.2)
    assert len(alerts) == n


def test_recovery_alert():
    alerts = []
    hm = HealthMonitor(stale_after=5.0, expected={"m": 1.0},
                       on_alert=lambda n, o, s: alerts.append((n, o, s)))
    hm.beat("m", 100.0)
    hm.summary(now=200.0)            # -> stale
    hm.beat("m", 200.1)
    hm.summary(now=200.2)            # -> ok again
    assert alerts[-1] == ("m", STALE, OK)
