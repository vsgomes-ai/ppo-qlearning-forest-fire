"""Forest-fire cellular automaton (class-based API)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .states import BURNED, BURNING, EMPTY, TREE


def _shift_bool(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    src_r0, src_r1 = max(0, dr), min(h, h + dr)
    src_c0, src_c1 = max(0, dc), min(w, w + dc)
    dst_r0, dst_r1 = max(0, -dr), min(h, h - dr)
    dst_c0, dst_c1 = max(0, -dc), min(w, w - dc)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


@dataclass
class ForestFireCA:
    """Vectorised forest-fire CA with wind bias and ember spotting."""

    p_spread: float = 0.90
    p_grow: float = 0.0
    wind: tuple[int, int] = (0, 1)
    wind_boost: float = 2.0
    p_spot: float = 0.28
    spot_min: int = 3
    spot_max: int = 6

    def step(
        self,
        grid: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        nxt = grid.copy()
        nxt[grid == BURNING] = BURNED

        h, w = grid.shape
        burning = grid == BURNING
        tree = grid == TREE
        wind_r, wind_c = int(self.wind[0]), int(self.wind[1])

        best_align = np.full(grid.shape, -np.inf, dtype=np.float64)
        has_nb = np.zeros(grid.shape, dtype=bool)

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbour_burning = _shift_bool(burning, dr, dc)
                m = neighbour_burning & tree
                if not m.any():
                    continue
                align = -(dr * wind_r + dc * wind_c)
                has_nb |= m
                best_align = np.where(m & (align > best_align), align, best_align)

        p_map = np.full(grid.shape, self.p_spread, dtype=np.float64)
        if self.wind_boost > 0:
            downwind = has_nb & (best_align > 0)
            upwind = has_nb & (best_align < 0)
            p_map = np.where(
                downwind,
                np.minimum(1.0, self.p_spread * (1.0 + self.wind_boost * best_align)),
                p_map,
            )
            p_map = np.where(
                upwind,
                self.p_spread * max(0.05, 1.0 - 0.5 * self.wind_boost),
                p_map,
            )

        ignite = has_nb & tree & (rng.random(grid.shape) < p_map)
        nxt[ignite] = BURNING

        if self.p_spot > 0 and (wind_r != 0 or wind_c != 0):
            perp_r, perp_c = -wind_c, wind_r
            for br, bc in np.argwhere(burning):
                if rng.random() >= self.p_spot:
                    continue
                dist = int(rng.integers(self.spot_min, self.spot_max + 1))
                lateral = int(rng.integers(-1, 2))
                tr = br + wind_r * dist + perp_r * lateral
                tc = bc + wind_c * dist + perp_c * lateral
                if 0 <= tr < h and 0 <= tc < w and grid[tr, tc] == TREE:
                    nxt[tr, tc] = BURNING

        if self.p_grow > 0:
            grow = (grid == EMPTY) & (rng.random(grid.shape) < self.p_grow)
            nxt[grow] = TREE
        return nxt

    @staticmethod
    def random_forest(
        size: int,
        tree_density: float = 0.60,
        n_ignitions: int = 3,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        grid = np.where(
            rng.random((size, size)) < tree_density, TREE, EMPTY
        ).astype(np.int8)
        for _ in range(n_ignitions):
            trees = np.argwhere(grid == TREE)
            if len(trees) == 0:
                r, c = rng.integers(0, size, size=2)
            else:
                r, c = trees[rng.integers(0, len(trees))]
            grid[r, c] = BURNING
        return grid

    @staticmethod
    def apply_firebreak(
        grid: np.ndarray, cells: list[tuple[int, int]] | np.ndarray
    ) -> int:
        changed = 0
        for r, c in cells:
            if grid[r, c] in (TREE, BURNING):
                grid[r, c] = EMPTY
                changed += 1
        return changed

    @staticmethod
    def fire_centroid(grid: np.ndarray) -> tuple[float, float]:
        pts = np.argwhere(grid == BURNING)
        if len(pts) == 0:
            h, w = grid.shape
            return (h - 1) / 2.0, (w - 1) / 2.0
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())

    @staticmethod
    def summary(grid: np.ndarray) -> dict[str, int]:
        return {
            "empty": int(np.sum(grid == EMPTY)),
            "tree": int(np.sum(grid == TREE)),
            "burning": int(np.sum(grid == BURNING)),
            "burned": int(np.sum(grid == BURNED)),
        }


# Functional aliases (backward compatible)
def random_forest(*args, **kwargs) -> np.ndarray:
    return ForestFireCA.random_forest(*args, **kwargs)


def apply_firebreak(grid, cells) -> int:
    return ForestFireCA.apply_firebreak(grid, cells)


def fire_centroid(grid) -> tuple[float, float]:
    return ForestFireCA.fire_centroid(grid)


def summary(grid) -> dict[str, int]:
    return ForestFireCA.summary(grid)


def step_ca(
    grid: np.ndarray,
    p_spread: float = 1.0,
    p_grow: float = 0.0,
    wind: tuple[int, int] = (0, 0),
    wind_boost: float = 0.0,
    p_spot: float = 0.0,
    spot_min: int = 2,
    spot_max: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    ca = ForestFireCA(
        p_spread=p_spread,
        p_grow=p_grow,
        wind=wind,
        wind_boost=wind_boost,
        p_spot=p_spot,
        spot_min=spot_min,
        spot_max=spot_max,
    )
    return ca.step(grid, rng=rng)


def count_burning_neighbours(grid: np.ndarray) -> np.ndarray:
    burning = (grid == BURNING).astype(np.int16)
    padded = np.pad(burning, 1, mode="constant", constant_values=0)
    total = np.zeros_like(burning, dtype=np.int16)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += padded[
                1 + dr : 1 + dr + burning.shape[0],
                1 + dc : 1 + dc + burning.shape[1],
            ]
    return total
