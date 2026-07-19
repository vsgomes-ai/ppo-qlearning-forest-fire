"""Q-learning trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..agents import QLearningAgent
from ..config import make_default_env
from ..env import n_states, obs_to_state_index


class QTrainer:
    """Train a tabular Q-learning agent on ``ForestFireEnv``."""

    def __init__(
        self,
        episodes: int = 3000,
        seed: int = 42,
        size: int = 30,
        coarse: int = 5,
        out_dir: str | Path = "results",
    ):
        self.episodes = episodes
        self.seed = seed
        self.size = size
        self.coarse = coarse
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        env = make_default_env(seed=self.seed, size=self.size, coarse=self.coarse)
        ns = n_states(env.n_regions, env.n_fire_bins, env.n_tree_bins)
        agent = QLearningAgent(n_actions=env.action_space.n, n_states_=ns, seed=self.seed)

        episode_returns: list[float] = []
        trees_saved: list[float] = []

        for ep in range(self.episodes):
            obs, info = env.reset(seed=self.seed + ep)
            s = obs_to_state_index(obs, env.n_regions, env.n_fire_bins, env.n_tree_bins)
            total_r = 0.0
            done = False
            while not done:
                a = agent.act(s, explore=True)
                obs2, r, terminated, truncated, info = env.step(a)
                s2 = obs_to_state_index(
                    obs2, env.n_regions, env.n_fire_bins, env.n_tree_bins
                )
                done = terminated or truncated
                agent.update(s, a, r, s2, done)
                s = s2
                total_r += r
            agent.decay_epsilon()
            episode_returns.append(total_r)
            trees_saved.append(float(info["trees_saved_frac"]))

            if (ep + 1) % 200 == 0:
                avg_r = np.mean(episode_returns[-200:])
                avg_t = np.mean(trees_saved[-200:])
                print(
                    f"ep {ep+1:5d} | eps={agent.epsilon:.3f} | "
                    f"R̄={avg_r:.2f} | trees_saved̄={avg_t:.3f}"
                )

        q_path = self.out_dir / "q_table.npz"
        agent.save(q_path)
        metrics = {
            "episodes": self.episodes,
            "final_epsilon": agent.epsilon,
            "mean_return_last_200": float(np.mean(episode_returns[-200:])),
            "mean_trees_saved_last_200": float(np.mean(trees_saved[-200:])),
        }
        with open(self.out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        np.save(self.out_dir / "episode_returns.npy", np.array(episode_returns))
        np.save(self.out_dir / "trees_saved.npy", np.array(trees_saved))

        from ..rendering import plot_learning_curve

        plot_learning_curve(episode_returns, self.out_dir / "learning_curve.png")
        print(f"Saved Q-table to {q_path}")
        return {**metrics, "episode_returns": episode_returns, "trees_saved": trees_saved}


def train(
    episodes: int = 3000,
    seed: int = 42,
    size: int = 30,
    coarse: int = 5,
    out_dir: str | Path = "results",
) -> dict:
    return QTrainer(
        episodes=episodes, seed=seed, size=size, coarse=coarse, out_dir=out_dir
    ).train()


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning on Forest Fire CA")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--coarse", type=int, default=5)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        seed=args.seed,
        size=args.size,
        coarse=args.coarse,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
