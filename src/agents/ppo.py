"""PPO agent helpers (Stable-Baselines3)."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..env import ForestFireEnv, RichObservationEncoder
from .policies import Policy


def compute_rich_obs(env) -> np.ndarray:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    enc = RichObservationEncoder(base.coarse, base.block)
    assert base.grid is not None
    return enc.encode(base.grid)


def rich_obs_dim(env) -> int:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    return RichObservationEncoder(base.coarse, base.block).dim


class RichObsWrapper(gym.ObservationWrapper):
    """Expose structured spatial observation for PPO."""

    def __init__(self, env: ForestFireEnv):
        super().__init__(env)
        dim = rich_obs_dim(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(dim,), dtype=np.float32
        )

    def observation(self, obs):
        return compute_rich_obs(self.env)


class PPOPolicy(Policy):
    """Deterministic PPO policy using rich observations from a live env."""

    def __init__(self, model_path: str | Path, env: ForestFireEnv):
        from stable_baselines3 import PPO

        path = Path(model_path)
        load = path if path.suffix == ".zip" else Path(str(path) + ".zip")
        if not load.exists() and path.exists():
            load = path
        stem = str(load.with_suffix("")) if load.suffix == ".zip" else str(load)
        self.model = PPO.load(stem)
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        rich = compute_rich_obs(self.env)
        action, _ = self.model.predict(rich, deterministic=True)
        return int(action)


def make_ppo_policy(model_path: str | Path, env: ForestFireEnv):
    return PPOPolicy(model_path, env)
