"""Rendering package (2D plots + 3D forest scenes)."""

from ._legacy_viz import (  # noqa: F401
    grid_to_rgb,
    plot_grid,
    plot_learning_curve,
    plot_policy_comparison,
    save_episode_gif,
    save_episode_gif_3d,
    save_episode_gif_3d_compare,
    save_episode_gif_3d_grid4,
    save_grid_figure,
    save_grid_figure_3d,
)

__all__ = [
    "grid_to_rgb",
    "plot_grid",
    "save_grid_figure",
    "save_episode_gif",
    "plot_learning_curve",
    "plot_policy_comparison",
    "save_episode_gif_3d",
    "save_episode_gif_3d_compare",
    "save_episode_gif_3d_grid4",
    "save_grid_figure_3d",
]
