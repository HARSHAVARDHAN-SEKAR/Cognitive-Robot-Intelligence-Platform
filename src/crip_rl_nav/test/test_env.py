"""Unit tests for the RL navigation env core (no ROS, no torch needed)."""
import math

import numpy as np

from crip_rl_nav.env import CripNavCore, EnvConfig


def test_obs_shape_and_dtype():
    core = CripNavCore(EnvConfig(), seed=0)
    obs = core.reset(seed=0)
    assert obs.shape == (core.obs_dim,)
    assert obs.dtype == np.float32
    assert core.obs_dim == EnvConfig().n_beams + 3
    assert core.act_dim == 2


def test_progress_reward_positive_toward_goal():
    core = CripNavCore(EnvConfig(), seed=1)
    core.reset(seed=1)
    core.x, core.y, core.yaw = 0.0, 0.0, 0.0
    core.gx, core.gy = 3.0, 0.0
    core.prev_dist = core._dist_goal()
    _, r, term, trunc, _ = core.step(np.array([1.0, 0.0], np.float32))
    assert r > 0
    assert not term


def test_collision_terminates_with_penalty():
    cfg = EnvConfig()
    core = CripNavCore(cfg, seed=2)
    core.reset(seed=2)
    core.x = cfg.arena_half - cfg.robot_radius - 0.01
    core.y, core.yaw = 0.0, 0.0
    core.gx, core.gy = 0.0, 0.0
    core.prev_dist = core._dist_goal()
    _, r, term, trunc, _ = core.step(np.array([1.0, 0.0], np.float32))
    assert term
    assert r < -10


def test_reach_goal_bonus():
    cfg = EnvConfig()
    core = CripNavCore(cfg, seed=3)
    core.reset(seed=3)
    core.x, core.y, core.yaw = 0.0, 0.0, 0.0
    core.gx, core.gy = 0.05, 0.0   # within goal_tol after a tiny step
    core.prev_dist = core._dist_goal()
    _, r, term, _, info = core.step(np.array([1.0, 0.0], np.float32))
    assert term
    assert info["dist"] <= cfg.goal_tol
    assert r > cfg.reach_bonus - cfg.collision_pen  # bonus dominates


def test_env_is_solvable_by_greedy_policy():
    """Sanity: a simple goal-seeking policy should reach most goals. If this
    fails the reward/observation contract is broken and training cannot work."""
    core = CripNavCore(EnvConfig(), seed=0)

    def greedy(obs):
        bearing = math.atan2(obs[-2], obs[-1])
        w = max(-1.0, min(1.0, bearing))
        v = 1.0 if abs(bearing) < 0.5 else -0.2
        return np.array([v, w], np.float32)

    reached = 0
    trials = 30
    for t in range(trials):
        o = core.reset(seed=500 + t)
        for _ in range(core.cfg.max_steps):
            o, r, term, trunc, info = core.step(greedy(o))
            if term and info["dist"] <= core.cfg.goal_tol:
                reached += 1
            if term or trunc:
                break
    assert reached >= trials * 0.5, f"only reached {reached}/{trials}"


def test_action_clipping():
    core = CripNavCore(EnvConfig(), seed=4)
    core.reset(seed=4)
    # extreme actions must not throw and must keep state finite
    o, r, term, trunc, _ = core.step(np.array([10.0, -10.0], np.float32))
    assert np.all(np.isfinite(o))
    assert math.isfinite(r)
