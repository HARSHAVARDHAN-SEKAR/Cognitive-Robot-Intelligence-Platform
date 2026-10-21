#!/usr/bin/env python3
"""
crip_rl_nav.env

A lightweight 2D navigation environment for training the CRIP RL local
planner (Milestone 6). Pure Python + NumPy: no ROS, no Gazebo, no GPU
required to *define* it, so it is unit-testable here and trains fast on CPU
or GPU via Stable-Baselines3.

It deliberately mirrors the runtime observation/action contract of the
inference node so a policy trained here transfers to the robot:

  Observation (Box, float32):
    [ lidar_0 .. lidar_{N-1},      # N downsampled range readings, normalized 0..1
      goal_dist_norm,              # distance to goal / max_range, clipped 0..1
      goal_heading_sin,            # sin of bearing to goal in robot frame
      goal_heading_cos ]           # cos of bearing to goal in robot frame

  Action (Box, float32, [-1, 1]):
    [ v_cmd, w_cmd ]               # scaled to [0, v_max] and [-w_max, w_max]

  Reward (per the project's reward shaping):
    + reach_bonus     on reaching the goal
    + progress        * (prev_dist - dist)        toward-goal shaping
    - collision_pen   on hitting an obstacle/wall (episode ends)
    - time_pen        each step                    discourages dawdling
    - spin_pen        * |w|                        discourages oscillation

The world is a square arena with circular obstacles; the lidar is simulated
by ray-casting against arena walls and obstacle disks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except ImportError:  # allow import/test without gymnasium installed
    gym = None
    spaces = None
    _HAVE_GYM = False


@dataclass
class EnvConfig:
    arena_half: float = 5.0          # arena spans [-half, half] in x and y
    n_beams: int = 16                # downsampled lidar beams in the observation
    max_range: float = 8.0
    v_max: float = 0.26
    w_max: float = 1.0
    dt: float = 0.1
    robot_radius: float = 0.30
    goal_tol: float = 0.30
    max_steps: int = 400
    # reward weights
    reach_bonus: float = 100.0
    progress_w: float = 20.0
    collision_pen: float = 50.0
    time_pen: float = 0.2
    spin_pen: float = 0.1
    obstacles: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(2.0, 1.5, 0.6), (-1.5, -2.0, 0.5)])


class CripNavCore:
    """ROS-free simulation core. Usable directly in tests; the Gym wrapper
    below just adds the gymnasium API on top."""

    def __init__(self, cfg: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.x = self.y = self.yaw = 0.0
        self.gx = self.gy = 0.0
        self.steps = 0
        self.prev_dist = 0.0

    # ----- geometry -----
    def _dist_goal(self):
        return math.hypot(self.gx - self.x, self.gy - self.y)

    def _cast_ray(self, angle) -> float:
        """Distance to the nearest wall/obstacle along a world-frame ray."""
        cfg = self.cfg
        max_r = cfg.max_range
        dx, dy = math.cos(angle), math.sin(angle)
        # march (cheap + robust for a training sim)
        step = 0.05
        r = 0.0
        while r < max_r:
            px = self.x + dx * r
            py = self.y + dy * r
            if abs(px) >= cfg.arena_half or abs(py) >= cfg.arena_half:
                return r
            for (ox, oy, orad) in cfg.obstacles:
                if (px - ox) ** 2 + (py - oy) ** 2 <= orad ** 2:
                    return r
            r += step
        return max_r

    def _lidar(self) -> np.ndarray:
        n = self.cfg.n_beams
        beams = np.empty(n, dtype=np.float32)
        for i in range(n):
            a = self.yaw + (2 * math.pi * i / n) - math.pi
            beams[i] = self._cast_ray(a) / self.cfg.max_range
        return beams

    def _collision(self) -> bool:
        cfg = self.cfg
        if (abs(self.x) >= cfg.arena_half - cfg.robot_radius or
                abs(self.y) >= cfg.arena_half - cfg.robot_radius):
            return True
        for (ox, oy, orad) in cfg.obstacles:
            if math.hypot(self.x - ox, self.y - oy) <= orad + cfg.robot_radius:
                return True
        return False

    def _obs(self) -> np.ndarray:
        d = self._dist_goal()
        bearing = math.atan2(self.gy - self.y, self.gx - self.x) - self.yaw
        extra = np.array([
            min(d / self.cfg.max_range, 1.0),
            math.sin(bearing),
            math.cos(bearing),
        ], dtype=np.float32)
        return np.concatenate([self._lidar(), extra]).astype(np.float32)

    # ----- API -----
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        # random free start + goal
        self.x, self.y = self._sample_free()
        self.yaw = float(self.rng.uniform(-math.pi, math.pi))
        self.gx, self.gy = self._sample_free(min_from=(self.x, self.y), min_d=2.0)
        self.steps = 0
        self.prev_dist = self._dist_goal()
        return self._obs()

    def _sample_free(self, min_from=None, min_d=0.0):
        cfg = self.cfg
        for _ in range(200):
            x = float(self.rng.uniform(-cfg.arena_half + 0.6, cfg.arena_half - 0.6))
            y = float(self.rng.uniform(-cfg.arena_half + 0.6, cfg.arena_half - 0.6))
            ok = all(math.hypot(x - ox, y - oy) > orad + cfg.robot_radius + 0.2
                     for (ox, oy, orad) in cfg.obstacles)
            if ok and (min_from is None or
                       math.hypot(x - min_from[0], y - min_from[1]) >= min_d):
                return x, y
        return 0.0, 0.0

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # map [-1,1] -> velocity ranges; forward-only linear (diff drive)
        v = float((a[0] + 1.0) * 0.5 * cfg.v_max)
        w = float(a[1] * cfg.w_max)
        self.yaw = (self.yaw + w * cfg.dt + math.pi) % (2 * math.pi) - math.pi
        self.x += v * math.cos(self.yaw) * cfg.dt
        self.y += v * math.sin(self.yaw) * cfg.dt
        self.steps += 1

        dist = self._dist_goal()
        reward = -cfg.time_pen
        reward += cfg.progress_w * (self.prev_dist - dist)
        reward -= cfg.spin_pen * abs(a[1])
        self.prev_dist = dist

        terminated = False
        truncated = False
        if self._collision():
            reward -= cfg.collision_pen
            terminated = True
        elif dist <= cfg.goal_tol:
            reward += cfg.reach_bonus
            terminated = True
        elif self.steps >= cfg.max_steps:
            truncated = True

        return self._obs(), float(reward), terminated, truncated, {"dist": dist}

    @property
    def obs_dim(self):
        return self.cfg.n_beams + 3

    @property
    def act_dim(self):
        return 2


if _HAVE_GYM:
    class CripNavEnv(gym.Env):
        """Gymnasium wrapper around CripNavCore for Stable-Baselines3."""
        metadata = {"render_modes": []}

        def __init__(self, cfg: Optional[EnvConfig] = None):
            super().__init__()
            self.core = CripNavCore(cfg)
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.core.obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self.core.reset(seed)
            return obs, {}

        def step(self, action):
            return self.core.step(action)
else:
    CripNavEnv = None
