"""Unit tests for the knowledge-graph relations layer (improvement #4)."""
from crip_world_model.relations import RelationStore


def kitchen_graph():
    kg = RelationStore()
    kg.add("bottle", "on", "table")
    kg.add("table", "in", "kitchen")
    kg.add("kitchen", "in", "building_a")
    return kg


def test_where_is_walks_containment_chain():
    kg = kitchen_graph()
    assert kg.where_is("bottle") == ["table", "kitchen", "building_a"]


def test_contents_are_transitive():
    kg = kitchen_graph()
    assert set(kg.contents_of("kitchen")) == {"table", "bottle"}
    assert set(kg.contents_of("building_a")) == {"kitchen", "table", "bottle"}


def test_near_inference_from_positions():
    kg = RelationStore(near_radius=1.0)
    kg.infer_near({"a": (0, 0), "b": (0.5, 0.5), "c": (5, 5)})
    assert kg.objects("a", "near") == ["b"]
    assert kg.objects("c", "near") == []


def test_near_reinference_clears_stale():
    kg = RelationStore(near_radius=1.0)
    kg.infer_near({"a": (0, 0), "b": (0.5, 0)})
    kg.infer_near({"a": (0, 0), "b": (9, 9)})
    assert kg.objects("a", "near") == []


def test_self_loop_rejected_and_dedup():
    kg = RelationStore()
    kg.add("x", "in", "x")
    kg.add("a", "on", "b")
    kg.add("a", "on", "b")
    assert kg.to_lines() == ["a on b"]


def test_serialize_round_trip():
    kg = kitchen_graph()
    kg2 = RelationStore()
    kg2.load_lines(kg.to_lines())
    assert kg2.where_is("bottle") == ["table", "kitchen", "building_a"]


def test_containment_cycle_terminates():
    kg = RelationStore()
    kg.add("a", "in", "b")
    kg.add("b", "in", "a")
    # must not loop forever; chain stops when a repeat is seen
    assert len(kg.where_is("a")) <= 6
