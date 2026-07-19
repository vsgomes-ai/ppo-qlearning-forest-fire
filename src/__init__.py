"""Forest-fire cellular automaton controlled with reinforcement learning."""

from .ca import ForestFireCA
from .config import EnvConfig, make_default_env
from .env import ForestFireEnv

__all__ = [
    "EnvConfig",
    "ForestFireCA",
    "ForestFireEnv",
    "make_default_env",
]
