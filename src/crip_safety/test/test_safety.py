"""Unit tests for the safety rule engine (pure logic, no ROS)."""
from crip_safety.rules import (
    SafetyEngine, SafetyContext, SafetyConfig, Command, Decision,
    EStopRule, CollisionRule, GeofenceRule, SpeedLimitRule, RateLimitRule,
    SideClearanceRule)


def eng():
    return SafetyEngine()


def test_clear_path_allows_motion():
    r = eng().check(Command(0.2, 0.0),
                    SafetyContext(front=5.0, prev_v=0.2))
    assert not r.vetoed
    assert r.command.v > 0


def test_frontal_obstacle_vetoes_forward():
    r = eng().check(Command(0.25, 0.5),
                    SafetyContext(front=0.2, prev_v=0.2))
    assert r.vetoed
    assert r.command.v == 0.0


def test_veto_is_not_reaccelerated_by_rate_limit():
    """A safety stop must reach zero, not be smoothed back up."""
    r = eng().check(Command(0.25, 0.0),
                    SafetyContext(front=0.2, prev_v=0.25))
    assert r.command.v == 0.0


def test_slow_band_scales_speed():
    r = eng().check(Command(0.26, 0.0),
                    SafetyContext(front=0.7, prev_v=0.26))
    assert not r.vetoed
    assert 0.0 < r.command.v < 0.26


def test_speed_limit_clamps():
    r = eng().check(Command(5.0, 9.0),
                    SafetyContext(front=5.0, prev_v=0.26, prev_w=1.0))
    assert r.command.v <= 0.26 + 1e-9
    assert abs(r.command.w) <= 1.0 + 1e-9


def test_geofence_recovers_at_boundary():
    """At the edge the robot must not go forward; with space behind it
    REVERSES (turning alone cannot change x/y) and turns hard."""
    r = eng().check(Command(0.2, 0.0),
                    SafetyContext(front=5.0, x=4.7, prev_v=0.2))
    assert r.vetoed
    assert r.command.v <= 0.0          # never forward
    assert abs(r.command.w) > 0.1      # actively turning away


def test_geofence_blocked_behind_turns_only():
    r = eng().check(Command(0.2, 0.0),
                    SafetyContext(front=5.0, rear=0.3, x=4.7, prev_v=0.2))
    assert r.vetoed
    assert r.command.v == 0.0          # cannot reverse into obstacle
    assert abs(r.command.w) > 0.1


def test_geofence_allows_inside():
    r = eng().check(Command(0.2, 0.0),
                    SafetyContext(front=5.0, x=0.0, prev_v=0.2))
    assert not r.vetoed


def test_side_clearance_blocks_turn_into_obstacle():
    r = eng().check(Command(0.1, 0.5),
                    SafetyContext(front=5.0, left=0.1, prev_v=0.1, prev_w=0.5))
    assert abs(r.command.w) < 1e-9


def test_estop_forces_full_stop():
    r = eng().check(Command(0.25, 0.5),
                    SafetyContext(front=5.0, estop=True, prev_v=0.2))
    assert r.vetoed
    assert r.command.v == 0.0 and r.command.w == 0.0


def test_rate_limit_bounds_acceleration():
    r = eng().check(Command(0.26, 0.0),
                    SafetyContext(front=5.0, prev_v=0.0, dt=0.1))
    # max_accel 1.5 * dt 0.1 = 0.15
    assert r.command.v <= 0.15 + 1e-9


def test_deceleration_always_allowed():
    r = eng().check(Command(0.0, 0.0),
                    SafetyContext(front=5.0, prev_v=0.26))
    assert r.command.v == 0.0


def test_audit_line_names_acting_rules():
    r = eng().check(Command(0.25, 0.0),
                    SafetyContext(front=0.2, prev_v=0.2))
    line = r.audit_line()
    assert "VETO" in line
    assert "collision" in line


def test_audit_line_clean_when_allowed():
    r = eng().check(Command(0.2, 0.0),
                    SafetyContext(front=5.0, prev_v=0.2))
    assert r.audit_line().startswith("allow")


def test_pipeline_runs_all_rules_in_order():
    r = eng().check(Command(0.2, 0.0), SafetyContext(front=5.0, prev_v=0.2))
    names = [d.rule for d in r.decisions]
    assert names == ["estop", "geofence", "collision", "rear_collision",
                     "side_clearance", "speed_limit", "rate_limit"]


def test_custom_config_tighter_speed():
    e = SafetyEngine(cfg=SafetyConfig(max_v=0.1))
    r = e.check(Command(0.26, 0.0), SafetyContext(front=5.0, prev_v=0.1))
    assert r.command.v <= 0.1 + 1e-9


def test_rear_collision_vetoes_reverse_into_obstacle():
    r = eng().check(Command(-0.2, 0.0),
                    SafetyContext(front=5.0, rear=0.2, prev_v=-0.1))
    assert r.vetoed
    assert r.command.v == 0.0


def test_sign_flip_reverse_passes_rate_limit():
    """A forward->reverse flip is a recovery manoeuvre; the rate limiter must
    not smooth it away (that caused the boundary oscillation bug)."""
    r = eng().check(Command(-0.15, 0.0),
                    SafetyContext(front=5.0, rear=5.0, prev_v=0.25))
    assert r.command.v < 0.0
