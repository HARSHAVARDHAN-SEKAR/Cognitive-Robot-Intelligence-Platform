"""Unit tests for the explainability trace (no ROS)."""
from crip_explain.trace import ExplanationTrace


def test_empty_trace():
    tr = ExplanationTrace()
    ans = tr.why()
    assert ans["factors"] == []
    assert "No recent" in ans["summary"]


def test_safety_veto_is_primary_cause():
    tr = ExplanationTrace()
    tr.add("brain", "navigating to charging_station")
    tr.add("vlm", "HIGH risk: person in path", decisive=True)
    tr.add("safety", "VETO frontal obstacle 0.22m", decisive=True)
    ans = tr.why()
    assert ans["primary_cause"]["source"] == "safety"
    assert ans["summary"].startswith("Stopped because:")


def test_decisive_outranks_routine():
    tr = ExplanationTrace()
    for i in range(5):
        tr.add("brain", f"routine status {i}")
    tr.add("safety", "VETO obstacle", decisive=True)
    ans = tr.why(top_k=1)
    assert "VETO" in ans["summary"]


def test_factors_chronological():
    tr = ExplanationTrace()
    tr.add("vlm", "first event", decisive=True)
    tr.add("safety", "second event", decisive=True)
    ans = tr.why(top_k=2)
    texts = [f["text"] for f in ans["factors"]]
    assert texts.index("first event") < texts.index("second event")


def test_stale_events_drop_out_of_window():
    tr = ExplanationTrace()
    tr.add("safety", "old veto", decisive=True)
    tr.events[-1].t -= 120          # 2 minutes ago
    tr.add("brain", "currently navigating")
    ans = tr.why()
    assert "old veto" not in ans["summary"]


def test_top_k_limits_factors():
    tr = ExplanationTrace()
    for i in range(6):
        tr.add("vlm", f"risk event {i}", decisive=True)
    ans = tr.why(top_k=2)
    assert len(ans["factors"]) == 2


def test_headline_verbs():
    tr = ExplanationTrace()
    tr.add("vlm", "MEDIUM risk spill", decisive=True)
    ans = tr.why()
    assert ans["summary"].startswith("Slowed / rerouted because:")
