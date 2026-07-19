"""Environment configuration (single source of truth)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .env import ForestFireEnv


@dataclass(frozen=True)
class EnvConfig:
    """Canonical experiment configuration used by all policies."""

    size: int = 30
    coarse: int = 5
    tree_density: float = 0.60
    n_ignitions: int = 3
    p_spread: float = 0.90
    p_grow: float = 0.0
    clear_full_region: bool = False
    cells_per_action: int = 8
    wind: tuple[int, int] = (0, 1)
    wind_boost: float = 2.0
    p_spot: float = 0.28
    spot_min: int = 3
    spot_max: int = 6
    max_steps: int = 70
    seed: int | None = None
    render_mode: str | None = None

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides) -> EnvConfig:
        data = self.to_kwargs()
        data.update(overrides)
        valid = {f.name for f in fields(self)}
        return EnvConfig(**{k: v for k, v in data.items() if k in valid})


DEFAULT_CONFIG = EnvConfig()


def make_default_env(seed: int | None = None, **overrides) -> ForestFireEnv:
    """Factory used by training, evaluation and rendering."""
    from .env import ForestFireEnv

    cfg = DEFAULT_CONFIG.with_overrides(seed=seed, **overrides)
    return ForestFireEnv(**cfg.to_kwargs())
