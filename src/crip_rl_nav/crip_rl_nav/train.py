#!/usr/bin/env python3
"""
crip_rl_nav.train

Offline training harness for the CRIP RL local planner (Milestone 6).

This is an OFFLINE step. It is intentionally NOT run as part of the ROS stack
or the test suite: training PPO/SAC to convergence wants many environment
steps and benefits from a GPU. The output is a small policy network that the
inference node (rl_controller) loads at runtime.

Usage:
    python3 -m crip_rl_nav.train --algo ppo --timesteps 1_000_000 \
        --out models/policy.onnx

Dependencies (install in the training environment, NOT required at runtime):
    pip install gymnasium stable-baselines3 torch onnx

The exported ONNX takes a float32 observation vector of length obs_dim and
returns a float32 action vector of length 2 in [-1, 1]. The runtime node
applies the same [-1,1] -> (v, w) mapping as the training env, so the policy
transfers without retraining.
"""
import argparse
import os

import numpy as np


def build_env():
    from crip_rl_nav.env import CripNavEnv, EnvConfig
    return CripNavEnv(EnvConfig())


def train(algo: str, timesteps: int, out_path: str, seed: int = 0):
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(build_env, n_envs=8, seed=seed)
    Algo = {"ppo": PPO, "sac": SAC}[algo.lower()]
    policy_kwargs = dict(net_arch=[128, 128])
    model = Algo("MlpPolicy", env, verbose=1, seed=seed,
                 policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=timesteps)

    # Save SB3 zip alongside the ONNX for retraining/inspection.
    zip_path = os.path.splitext(out_path)[0] + ".zip"
    model.save(zip_path)
    print(f"saved SB3 model -> {zip_path}")

    export_onnx(model, out_path)
    print(f"exported ONNX policy -> {out_path}")


def export_onnx(model, out_path: str):
    """Export the deterministic actor (action mean) to ONNX.

    We export the raw actor network rather than policy._predict, because the
    distribution sampling path contains data-dependent validation that the
    tracer cannot capture. For a deterministic policy the action is just the
    network's mean output, which is exactly what we run at inference time.
    Works for PPO (ActorCriticPolicy) and SAC (SACPolicy)."""
    import torch

    obs_dim = model.observation_space.shape[0]
    policy = model.policy

    class ActorMean(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            self.is_sac = hasattr(policy, "actor")

        def forward(self, obs):
            if self.is_sac:
                # SAC: actor returns the mean action (already tanh-squashed)
                return self.policy.actor(obs, deterministic=True)
            # PPO/A2C: features -> shared/pi latent -> action_net (mean)
            features = self.policy.extract_features(obs)
            if self.policy.share_features_extractor:
                latent_pi, _ = self.policy.mlp_extractor(features)
            else:
                pi_features, _ = features
                latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            return self.policy.action_net(latent_pi)

    wrapper = ActorMean(policy).eval()
    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Use the legacy TorchScript exporter (dynamo=False): robust for these
    # small MLP actors and avoids data-dependent export guards.
    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def main():
    ap = argparse.ArgumentParser(description="Train the CRIP RL local planner.")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", default="models/policy.onnx")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.algo, args.timesteps, args.out, args.seed)


if __name__ == "__main__":
    main()
