"""Gymnasium environment and observation helpers."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .ca import (
    BURNED,
    BURNING,
    TREE,
    ForestFireCA,
    apply_firebreak,
    fire_centroid,
    random_forest,
    summary,
)

RELATIVE_DELTAS = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]
NOOP_ACTION = len(RELATIVE_DELTAS)


class CompactObservationEncoder:
    """Tabular observation: (fire_region, fire_bin, tree_bin)."""

    def __init__(self, size: int, coarse: int, n_fire_bins: int = 5, n_tree_bins: int = 5):
        self.size = size
        self.coarse = coarse
        self.block = size // coarse
        self.n_regions = coarse * coarse
        self.n_fire_bins = n_fire_bins
        self.n_tree_bins = n_tree_bins

    @property
    def n_states(self) -> int:
        return self.n_regions * self.n_fire_bins * self.n_tree_bins

    def encode(self, grid: np.ndarray, initial_trees: int) -> np.ndarray:
        fr, fc = fire_centroid(grid)
        region_r = min(int(fr // self.block), self.coarse - 1)
        region_c = min(int(fc // self.block), self.coarse - 1)
        fire_region = region_r * self.coarse + region_c
        n_burn = int(np.sum(grid == BURNING))
        area = self.size * self.size
        fire_bin = min(self.n_fire_bins - 1, int(n_burn / max(1, area * 0.015)))
        n_tree = int(np.sum(grid == TREE))
        tree_frac = n_tree / max(1, initial_trees)
        tree_bin = min(self.n_tree_bins - 1, int(tree_frac * self.n_tree_bins))
        return np.array([fire_region, fire_bin, tree_bin], dtype=np.int64)

    @staticmethod
    def to_state_index(
        obs: np.ndarray, n_regions: int = 25, n_fire: int = 5, n_tree: int = 5
    ) -> int:
        fr, fb, tb = int(obs[0]), int(obs[1]), int(obs[2])
        return fr * (n_fire * n_tree) + fb * n_tree + tb


class RichObservationEncoder:
    """Structured spatial observation for PPO (region densities + centroid)."""

    def __init__(self, coarse: int, block: int):
        self.coarse = coarse
        self.block = block

    @property
    def dim(self) -> int:
        return 3 * self.coarse * self.coarse + 2

    def encode(self, grid: np.ndarray) -> np.ndarray:
        fire = (grid == BURNING).astype(np.float32)
        tree = (grid == TREE).astype(np.float32)
        burned = (grid == BURNED).astype(np.float32)

        def pool(m: np.ndarray) -> np.ndarray:
            return m.reshape(self.coarse, self.block, self.coarse, self.block).mean(
                axis=(1, 3)
            ).ravel()

        fr, fc = fire_centroid(grid)
        cr = fr / max(1, grid.shape[0] - 1)
        cc = fc / max(1, grid.shape[1] - 1)
        return np.concatenate(
            [pool(fire), pool(tree), pool(burned), np.array([cr, cc], dtype=np.float32)]
        ).astype(np.float32)


class ForestFireEnv(gym.Env):
    """
    Firebreak control on a forest-fire CA.

    Each step the agent clears fuel in a region relative to the fire centroid
    (or no-op), then the CA advances one tick.
    Compact observation is exposed by default for tabular agents.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        size: int = 30,
        coarse: int = 5,
        tree_density: float = 0.60,
        n_ignitions: int = 3,
        p_spread: float = 0.90,
        p_grow: float = 0.0,
        max_steps: int = 70,
        clear_full_region: bool = False,
        cells_per_action: int = 8,
        wind: tuple[int, int] = (0, 1),
        wind_boost: float = 2.0,
        p_spot: float = 0.28,
        spot_min: int = 3,
        spot_max: int = 6,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert size % coarse == 0, "size must be divisible by coarse"
        self.size = size
        self.coarse = coarse
        self.block = size // coarse
        self.tree_density = tree_density
        self.n_ignitions = n_ignitions
        self.max_steps = max_steps
        self.clear_full_region = clear_full_region
        self.cells_per_action = cells_per_action
        self.render_mode = render_mode

        self.ca = ForestFireCA(
            p_spread=p_spread,
            p_grow=p_grow,
            wind=(int(wind[0]), int(wind[1])),
            wind_boost=float(wind_boost),
            p_spot=float(p_spot),
            spot_min=int(spot_min),
            spot_max=int(spot_max),
        )
        # Convenience mirrors (used by observation helpers / PPO)
        self.p_spread = p_spread
        self.p_grow = p_grow
        self.wind = self.ca.wind
        self.wind_boost = self.ca.wind_boost
        self.p_spot = self.ca.p_spot
        self.spot_min = self.ca.spot_min
        self.spot_max = self.ca.spot_max

        self.obs_encoder = CompactObservationEncoder(size, coarse)
        self.n_regions = self.obs_encoder.n_regions
        self.n_fire_bins = self.obs_encoder.n_fire_bins
        self.n_tree_bins = self.obs_encoder.n_tree_bins
        self.action_space = spaces.Discrete(NOOP_ACTION + 1)
        self.observation_space = spaces.MultiDiscrete(
            [self.n_regions, self.n_fire_bins, self.n_tree_bins]
        )

        self._rng = np.random.default_rng(seed)
        self.grid: np.ndarray | None = None
        self.initial_trees = 0
        self.steps = 0
        self._last_burned = 0

    def _encode_obs(self) -> np.ndarray:
        assert self.grid is not None
        return self.obs_encoder.encode(self.grid, self.initial_trees)

    def _action_to_region(self, action: int, fire_region: int) -> int | None:
        if action >= NOOP_ACTION:
            return None
        dr, dc = RELATIVE_DELTAS[action]
        rr, cc = divmod(fire_region, self.coarse)
        nr = int(np.clip(rr + dr, 0, self.coarse - 1))
        nc = int(np.clip(cc + dc, 0, self.coarse - 1))
        return nr * self.coarse + nc

    def _region_cells(self, region: int) -> list[tuple[int, int]]:
        assert self.grid is not None
        rr, cc = divmod(region, self.coarse)
        r0, c0 = rr * self.block, cc * self.block
        cells = [
            (r, c)
            for r in range(r0, r0 + self.block)
            for c in range(c0, c0 + self.block)
        ]
        burning = [(r, c) for r, c in cells if self.grid[r, c] == BURNING]
        trees = [(r, c) for r, c in cells if self.grid[r, c] == TREE]
        ordered = burning + trees
        if self.clear_full_region:
            return ordered
        return ordered[: self.cells_per_action]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.grid = random_forest(
            self.size,
            tree_density=self.tree_density,
            n_ignitions=self.n_ignitions,
            rng=self._rng,
        )
        self.initial_trees = int(np.sum(self.grid == TREE)) + int(
            np.sum(self.grid == BURNING)
        )
        self.steps = 0
        self._last_burned = int(np.sum(self.grid == BURNED))
        obs = self._encode_obs()
        info = {
            "grid": self.grid.copy(),
            "summary": summary(self.grid),
            "initial_trees": self.initial_trees,
        }
        return obs, info

    def step(self, action: int):
        assert self.grid is not None
        fire_region = int(self._encode_obs()[0])
        region = self._action_to_region(int(action), fire_region)

        cleared = 0
        cleared_burning = 0
        if region is not None:
            cells = self._region_cells(region)
            before_burn = int(np.sum(self.grid == BURNING))
            cleared = apply_firebreak(self.grid, cells)
            cleared_burning = before_burn - int(np.sum(self.grid == BURNING))

        self.grid = self.ca.step(self.grid, rng=self._rng)
        self.steps += 1

        burned_now = int(np.sum(self.grid == BURNED))
        new_burned = burned_now - self._last_burned
        self._last_burned = burned_now
        trees = int(np.sum(self.grid == TREE))
        burning = int(np.sum(self.grid == BURNING))

        reward = 0.1 * (trees / max(1, self.initial_trees))
        reward -= 0.2 * new_burned
        reward += 1.5 * cleared_burning
        reward -= 0.01 * max(0, cleared - cleared_burning)

        terminated = burning == 0
        truncated = self.steps >= self.max_steps
        if terminated or truncated:
            saved = trees / max(1, self.initial_trees)
            reward += 8.0 * saved
            reward -= 3.0 * (burned_now / max(1, self.initial_trees))

        obs = self._encode_obs()
        info = {
            "grid": self.grid.copy(),
            "summary": summary(self.grid),
            "cleared": cleared,
            "cleared_burning": cleared_burning,
            "trees_saved_frac": trees / max(1, self.initial_trees),
            "burned": burned_now,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.grid is None:
            return None
        from .rendering import grid_to_rgb

        return grid_to_rgb(self.grid)


def obs_to_state_index(
    obs: np.ndarray, n_regions: int = 25, n_fire: int = 5, n_tree: int = 5
) -> int:
    return CompactObservationEncoder.to_state_index(obs, n_regions, n_fire, n_tree)


def n_states(n_regions: int = 25, n_fire: int = 5, n_tree: int = 5) -> int:
    return n_regions * n_fire * n_tree
