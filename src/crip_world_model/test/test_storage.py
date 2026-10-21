"""Unit tests for the world-model storage backend (no ROS required)."""
import time

from crip_world_model.storage import SqliteStore, Entity


def make_store():
    return SqliteStore(':memory:')


def test_merge_nearby_same_label():
    s = make_store()
    s.upsert(Entity('charging_station', 'landmark', 1.0, 1.0, 0, 0.8,
                    last_seen=time.time()))
    e = s.upsert(Entity('charging_station', 'landmark', 1.2, 1.1, 0, 0.9,
                        last_seen=time.time()))
    assert e.observation_count == 2
    assert e.confidence == 0.9            # keeps the higher confidence
    assert abs(e.x - 1.1) < 1e-6          # running-average position
    assert len(s.query(None, None, None, 0, 0)) == 1


def test_far_same_label_stays_separate():
    s = make_store()
    s.upsert(Entity('charging_station', 'landmark', 1.0, 1.0, 0, 0.8,
                    last_seen=time.time()))
    s.upsert(Entity('charging_station', 'landmark', 4.5, -4.0, 0, 0.6,
                    last_seen=time.time()))
    assert len(s.query(None, None, None, 0, 0)) == 2


def test_have_i_seen_counts_across_entities():
    s = make_store()
    s.upsert(Entity('person', 'dynamic', 0, 0, 0, 0.7, last_seen=time.time()))
    s.upsert(Entity('person', 'dynamic', 0.1, 0.1, 0, 0.9, last_seen=time.time()))
    s.upsert(Entity('person', 'dynamic', 5.0, 5.0, 0, 0.5, last_seen=time.time()))
    seen, times, closest = s.have_i_seen('person')
    assert seen is True
    assert times == 3
    assert closest.confidence == 0.9


def test_have_i_seen_unknown():
    s = make_store()
    seen, times, closest = s.have_i_seen('forklift')
    assert seen is False
    assert times == 0
    assert closest is None


def test_spatial_query():
    s = make_store()
    s.upsert(Entity('box', 'static', 1.0, 1.0, 0, 1.0, last_seen=time.time()))
    s.upsert(Entity('cyl', 'static', 6.0, 6.0, 0, 1.0, last_seen=time.time()))
    near = s.query(None, None, (1.0, 1.0), 2.0, 0)
    labels = [e.label for e in near]
    assert 'box' in labels and 'cyl' not in labels


def test_category_filter():
    s = make_store()
    s.upsert(Entity('person', 'dynamic', 0, 0, 0, 1.0, last_seen=time.time()))
    s.upsert(Entity('wall', 'static', 1, 1, 0, 1.0, last_seen=time.time()))
    dyn = s.query(None, 'dynamic', None, 0, 0)
    assert len(dyn) == 1 and dyn[0].label == 'person'
