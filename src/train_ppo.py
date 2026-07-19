"""Train PPO (CLI)."""

from .agents.ppo import RichObsWrapper, compute_rich_obs, rich_obs_dim
from .trainers.ppo_trainer import EpisodeRewardCallback, main, train

__all__ = [
    "train",
    "main",
    "compute_rich_obs",
    "rich_obs_dim",
    "RichObsWrapper",
    "EpisodeRewardCallback",
]

if __name__ == "__main__":
    main()
