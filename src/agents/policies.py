"""Policy interface and baseline policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..ca import BURNING
from ..env import NOOP_ACTION, ForestFireEnv


class Policy(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    def __call__(self, obs: np.ndarray) -> int:
        return self.act(obs)


class RandomPolicy(Policy):
    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, self.n_actions))


class HeuristicPolicy(Policy):
    """Always clear the fire-centroid region (action 0) while fire is active."""

    def __init__(self, env: ForestFireEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        assert self.env.grid is not None
        if int(np.sum(self.env.grid == BURNING)) == 0:
            return NOOP_ACTION
        return 0


class NoOpPolicy(Policy):
    def act(self, obs: np.ndarray) -> int:
        return NOOP_ACTION


def random_policy(env: ForestFireEnv, rng: np.random.Generator) -> int:
    return int(rng.integers(0, env.action_space.n))


def heuristic_policy(env: ForestFireEnv) -> int:
    assert env.grid is not None
    if int(np.sum(env.grid == BURNING)) == 0:
        return NOOP_ACTION
    return 0
