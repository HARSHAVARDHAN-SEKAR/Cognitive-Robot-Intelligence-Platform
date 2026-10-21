"""Unit tests for the dashboard state aggregator (no ROS, no web server)."""
import json

from crip_dashboard.state import DashboardState


def test_snapshot_is_json_serializable():
    s = DashboardState()
    s.update_robot(1.0, 2.0, 0.5)
    json.dumps(s.snapshot())  # must not raise


def test_trail_dedup_on_small_moves():
    s = DashboardState()
    s.update_robot(0.0, 0.0, 0.0)
    s.update_robot(0.01, 0.0, 0.0)   # tiny move -> ignored
    s.update_robot(1.0, 0.0, 0.0)    # big move -> recorded
    assert len(s.snapshot()["trail"]) == 2


def test_obstacles_set_once():
    s = DashboardState()
    obs = [{"x": 2, "y": 1.5, "r": 0.5, "shape": "box"}]
    s.set_obstacles(obs)
    assert s.snapshot()["obstacles"] == obs


def test_goal_and_prediction():
    s = DashboardState()
    s.update_goal(3.0, 3.0)
    s.update_prediction(0, [(3, 2), (3, 1)])
    snap = s.snapshot()
    assert snap["goal"]["x"] == 3.0
    assert "0" in snap["predictions"]
    assert len(snap["predictions"]["0"]) == 2


def test_vlm_risk_generates_event():
    s = DashboardState()
    s.update_vlm("high", "smoke detected")
    snap = s.snapshot()
    assert snap["vlm"]["risk"] == "high"
    assert any(e["source"] == "vlm" for e in snap["events"])


def test_low_risk_no_event():
    s = DashboardState()
    s.update_vlm("none", "clear")
    snap = s.snapshot()
    assert all(e["source"] != "vlm" for e in snap["events"])


def test_event_feed_trimmed():
    s = DashboardState()
    for i in range(60):
        s.add_event("brain", f"event {i}")
    snap = s.snapshot()
    assert len(snap["events"]) <= 20      # snapshot trims to last 20


def test_learning_and_brain_events():
    s = DashboardState()
    s.update_learning("v0002: PROMOTED")
    s.update_brain("navigating to dock")
    sources = {e["source"] for e in s.snapshot()["events"]}
    assert "learning" in sources and "brain" in sources
