"""Tabular Q-learning agent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..env import ForestFireEnv, obs_to_state_index
from .policies import Policy


class QLearningAgent:
    """Tabular Q-learning (state-index API, compatible with saved q_table.npz)."""

    def __init__(
        self,
        n_actions: int,
        n_states_: int,
        alpha: float = 0.25,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.997,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.Q = self.rng.normal(0.1, 0.01, size=(n_states_, n_actions)).astype(np.float64)

    def act(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        target = r if done else r + self.gamma * np.max(self.Q[s2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, Q=self.Q, epsilon=np.array([self.epsilon]))

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        self.Q = data["Q"]
        self.epsilon = float(data["epsilon"][0])


QAgent = QLearningAgent


class QPolicy(Policy):
    def __init__(self, agent: QLearningAgent, env: ForestFireEnv):
        self.agent = agent
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        s = obs_to_state_index(
            obs, self.env.n_regions, self.env.n_fire_bins, self.env.n_tree_bins
        )
        return self.agent.act(s, explore=False)


def q_policy(agent: QLearningAgent, obs: np.ndarray, env: ForestFireEnv) -> int:
    s = obs_to_state_index(obs, env.n_regions, env.n_fire_bins, env.n_tree_bins)
    return agent.act(s, explore=False)
