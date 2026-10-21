"""Unit tests for the fleet-sync core (no ROS, no MQTT)."""
from crip_fleet.sync import InMemoryBroker, FleetSync


def two_robots():
    b = InMemoryBroker()
    return FleetSync("alpha", b), FleetSync("beta", b)


def test_peer_learns_shared_entity():
    a, b = two_robots()
    a.share_entity("charging_station", "landmark", 3.0, 1.0, 0.9)
    labels = [e.label for e in b.fleet_entities()]
    assert "charging_station" in labels


def test_robot_ignores_own_echo():
    a, b = two_robots()
    got = []
    a.on_peer_entity(lambda e: got.append(e))
    a.share_entity("door", "static", 0.0, 4.0, 0.8)
    assert got == []          # own entity is not a "peer" entity


def test_peer_callback_fires_for_peer_entity():
    a, b = two_robots()
    got = []
    b.on_peer_entity(lambda e: got.append((e.label, e.robot_id)))
    a.share_entity("pallet", "static", 1.0, 1.0, 0.7)
    assert ("pallet", "alpha") in got


def test_higher_confidence_wins_merge():
    a, b = two_robots()
    a.share_entity("charging_station", "landmark", 3.0, 1.0, 0.8)
    b.share_entity("charging_station", "landmark", 3.05, 1.02, 0.95)
    ent = [e for e in a.fleet_entities() if e.label == "charging_station"][0]
    assert ent.confidence == 0.95
    assert ent.robot_id == "beta"


def test_distinct_cells_stay_separate():
    a, b = two_robots()
    a.share_entity("box", "static", 0.0, 0.0, 0.9)
    a.share_entity("box", "static", 4.0, 4.0, 0.9)
    boxes = [e for e in a.fleet_entities() if e.label == "box"]
    assert len(boxes) == 2


def test_mission_and_model_sharing():
    a, b = two_robots()
    a.share_mission("inspect floor two", "success")
    assert b.peer_missions and b.peer_missions[0]["mission"] == "inspect floor two"
    b.announce_model("v0007", 0.0009)
    assert a.peer_models and a.peer_models[0]["version"] == "v0007"


def test_three_robot_propagation():
    b = InMemoryBroker()
    a = FleetSync("alpha", b)
    bb = FleetSync("beta", b)
    c = FleetSync("gamma", b)
    a.share_entity("fire_exit", "landmark", -4.0, -4.0, 0.7)
    assert any(e.label == "fire_exit" for e in bb.fleet_entities())
    assert any(e.label == "fire_exit" for e in c.fleet_entities())


def test_malformed_payload_ignored():
    b = InMemoryBroker()
    a = FleetSync("alpha", b)
    b.publish("crip/fleet/entity", "not json{{")
    # should not raise; no entity added
    assert a.fleet_entities() == []
