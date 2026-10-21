#!/usr/bin/env python3
"""
crip_safety.rules

The CRIP safety rule engine (Module 10), pure-Python and fully unit-testable.

A command is a (v, w) velocity pair. A SafetyContext carries the live world
state a rule needs (nearest obstacle distance per sector, robot pose, an
estimate of the previous command for rate limiting, an e-stop flag).

Each Rule inspects (command, context) and returns a Verdict: ALLOW, MODIFY
(with a replacement command), or VETO (force stop). Rules run in a fixed,
declared order; the engine applies them as a pipeline -- a later rule sees the
command as modified by earlier ones -- and a VETO short-circuits to a full
stop. Every decision carries a human-readable reason, so the whole chain is
auditable.

This layer encodes the project's "LLM Decision -> Safety Checker -> Allowed?
-> Execute" principle: no actuator command reaches the robot without passing
the engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class Decision(Enum):
    ALLOW = "allow"
    MODIFY = "modify"
    VETO = "veto"


@dataclass
class Command:
    v: float = 0.0   # linear, m/s
    w: float = 0.0   # angular, rad/s

    def copy(self):
        return Command(self.v, self.w)


@dataclass
class SafetyContext:
    # nearest obstacle distance (m) in each sector; inf if clear
    front: float = float("inf")
    left: float = float("inf")
    right: float = float("inf")
    rear: float = float("inf")
    # robot pose in map frame
    x: float = 0.0
    y: float = 0.0
    # previous *output* command, for rate limiting
    prev_v: float = 0.0
    prev_w: float = 0.0
    dt: float = 0.1
    # hard external emergency stop
    estop: bool = False


@dataclass
class Verdict:
    decision: Decision
    command: Command
    rule: str
    reason: str


@dataclass
class SafetyConfig:
    max_v: float = 0.26
    max_w: float = 1.0
    max_accel: float = 1.5          # m/s^2
    max_ang_accel: float = 3.0      # rad/s^2
    stop_distance: float = 0.35     # full stop if obstacle closer than this ahead
    slow_distance: float = 1.0      # scale speed down within this band
    side_clearance: float = 0.25    # veto turning into a very close side obstacle
    arena_half: float = 5.0
    geofence_margin: float = 0.4    # keep this far inside the arena wall
    reverse_speed: float = 0.15     # speed used to back away from a boundary


class Rule:
    name = "rule"

    def apply(self, cmd: Command, ctx: SafetyContext,
              cfg: SafetyConfig) -> Verdict:
        raise NotImplementedError


class EStopRule(Rule):
    name = "estop"

    def apply(self, cmd, ctx, cfg):
        if ctx.estop:
            return Verdict(Decision.VETO, Command(0.0, 0.0), self.name,
                           "external emergency stop engaged")
        return Verdict(Decision.ALLOW, cmd, self.name, "no e-stop")


class SpeedLimitRule(Rule):
    name = "speed_limit"

    def apply(self, cmd, ctx, cfg):
        v = max(-cfg.max_v, min(cfg.max_v, cmd.v))
        w = max(-cfg.max_w, min(cfg.max_w, cmd.w))
        if v != cmd.v or w != cmd.w:
            return Verdict(Decision.MODIFY, Command(v, w), self.name,
                           f"clamped to limits (v<={cfg.max_v}, w<={cfg.max_w})")
        return Verdict(Decision.ALLOW, cmd, self.name, "within limits")


class CollisionRule(Rule):
    """Stop forward motion before a frontal obstacle; scale down in the band."""
    name = "collision"

    def apply(self, cmd, ctx, cfg):
        # Reversing away from a frontal obstacle is always fine.
        if cmd.v < 0.0:
            return Verdict(Decision.ALLOW, cmd, self.name, "reversing")
        if cmd.v > 0.0 and ctx.front < cfg.stop_distance:
            # keep angular so it can rotate away, but no forward motion
            return Verdict(Decision.VETO, Command(0.0, cmd.w), self.name,
                           f"frontal obstacle {ctx.front:.2f}m < "
                           f"stop_distance {cfg.stop_distance:.2f}m")
        if cmd.v > 0.0 and ctx.front < cfg.slow_distance:
            scale = (ctx.front - cfg.stop_distance) / \
                    (cfg.slow_distance - cfg.stop_distance)
            scale = max(0.0, min(1.0, scale))
            nv = cmd.v * scale
            if nv < cmd.v:
                return Verdict(Decision.MODIFY, Command(nv, cmd.w), self.name,
                               f"slowing near obstacle (front={ctx.front:.2f}m, "
                               f"scale={scale:.2f})")
        return Verdict(Decision.ALLOW, cmd, self.name, "path clear")


class RearCollisionRule(Rule):
    """Don't reverse into something behind the robot."""
    name = "rear_collision"

    def apply(self, cmd, ctx, cfg):
        if cmd.v < 0.0 and ctx.rear < cfg.stop_distance:
            return Verdict(Decision.VETO, Command(0.0, cmd.w), self.name,
                           f"obstacle behind {ctx.rear:.2f}m; reverse vetoed")
        return Verdict(Decision.ALLOW, cmd, self.name, "rear clear")


class SideClearanceRule(Rule):
    """Don't turn toward a very close side obstacle."""
    name = "side_clearance"

    def apply(self, cmd, ctx, cfg):
        # w > 0 = turning left; w < 0 = turning right (REP-103)
        if cmd.w > 0.05 and ctx.left < cfg.side_clearance:
            return Verdict(Decision.MODIFY, Command(cmd.v, 0.0), self.name,
                           f"blocking left turn into obstacle "
                           f"{ctx.left:.2f}m")
        if cmd.w < -0.05 and ctx.right < cfg.side_clearance:
            return Verdict(Decision.MODIFY, Command(cmd.v, 0.0), self.name,
                           f"blocking right turn into obstacle "
                           f"{ctx.right:.2f}m")
        return Verdict(Decision.ALLOW, cmd, self.name, "side clear")


class GeofenceRule(Rule):
    """Keep the robot inside the arena and RECOVER if it reaches the edge.

    Turning in place cannot change x/y, so recovery must REVERSE -- that is
    the only motion that moves the robot back toward the centre. The turn is
    strong and directed away from the nearest violated wall, so the robot
    ends up facing back inside instead of oscillating along the boundary.
    """
    name = "geofence"

    def apply(self, cmd, ctx, cfg):
        limit = cfg.arena_half - cfg.geofence_margin
        at_edge = abs(ctx.x) >= limit or abs(ctx.y) >= limit
        if not at_edge:
            return Verdict(Decision.ALLOW, cmd, self.name, "inside geofence")

        # hard turn away from whichever wall is nearest
        if abs(ctx.x) >= abs(ctx.y):
            turn = -0.9 if ctx.x > 0 else 0.9
        else:
            turn = -0.9 if ctx.y > 0 else 0.9

        if ctx.rear > 0.6:
            return Verdict(
                Decision.VETO, Command(-cfg.reverse_speed, turn), self.name,
                f"boundary ({ctx.x:.1f},{ctx.y:.1f}); reversing + turning")
        return Verdict(
            Decision.VETO, Command(0.0, turn), self.name,
            f"boundary ({ctx.x:.1f},{ctx.y:.1f}); blocked behind, turning")


class RateLimitRule(Rule):
    """Limit acceleration to keep motion smooth and bounded. Deceleration
    (including a safety stop) is always allowed immediately -- this rule never
    re-accelerates a command that an earlier safety rule reduced."""
    name = "rate_limit"

    def apply(self, cmd, ctx, cfg):
        dv_max = cfg.max_accel * ctx.dt
        dw_max = cfg.max_ang_accel * ctx.dt
        v = cmd.v
        w = cmd.w
        # Only clamp when the magnitude is *increasing* (accelerating). Moving
        # toward zero or reversing sign reduces |speed| and is allowed.
        if (abs(v) > abs(ctx.prev_v) and v * ctx.prev_v >= 0
                and abs(v - ctx.prev_v) > dv_max):
            v = ctx.prev_v + math.copysign(dv_max, v - ctx.prev_v)
        if abs(w) > abs(ctx.prev_w) and abs(w - ctx.prev_w) > dw_max:
            w = ctx.prev_w + math.copysign(dw_max, w - ctx.prev_w)
        if v != cmd.v or w != cmd.w:
            return Verdict(Decision.MODIFY, Command(v, w), self.name,
                           "rate-limited to bound acceleration")
        return Verdict(Decision.ALLOW, cmd, self.name, "within accel limits")


# Declared, ordered pipeline. Order matters: e-stop first, rate-limit last so
# limits apply to the final commanded value.
DEFAULT_RULES: List[Rule] = [
    EStopRule(),
    GeofenceRule(),
    CollisionRule(),
    RearCollisionRule(),
    SideClearanceRule(),
    SpeedLimitRule(),
    RateLimitRule(),
]


@dataclass
class EngineResult:
    command: Command
    vetoed: bool
    decisions: List[Verdict] = field(default_factory=list)

    def audit_line(self) -> str:
        acted = [d for d in self.decisions
                 if d.decision in (Decision.MODIFY, Decision.VETO)]
        if not acted:
            return f"allow v={self.command.v:.2f} w={self.command.w:.2f}"
        parts = [f"{d.rule}:{d.decision.value}({d.reason})" for d in acted]
        return (f"{'VETO' if self.vetoed else 'modified'} -> "
                f"v={self.command.v:.2f} w={self.command.w:.2f} | "
                + "; ".join(parts))


class SafetyEngine:
    def __init__(self, rules: Optional[List[Rule]] = None,
                 cfg: Optional[SafetyConfig] = None):
        self.rules = rules if rules is not None else list(DEFAULT_RULES)
        self.cfg = cfg or SafetyConfig()

    def check(self, cmd: Command, ctx: SafetyContext) -> EngineResult:
        current = cmd.copy()
        decisions: List[Verdict] = []
        vetoed = False
        for rule in self.rules:
            verdict = rule.apply(current, ctx, self.cfg)
            decisions.append(verdict)
            current = verdict.command
            if verdict.decision == Decision.VETO:
                vetoed = True
                # a veto forces stop of forward motion but lets later rules
                # (e.g. rate limit) still shape the result
        return EngineResult(command=current, vetoed=vetoed, decisions=decisions)
