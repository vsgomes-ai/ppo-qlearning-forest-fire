"""PPO trainer (Stable-Baselines3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..agents import RichObsWrapper
from ..config import make_default_env


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, check_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        if self.n_calls % self.check_freq == 0 and self.episode_rewards:
            recent = self.episode_rewards[-50:]
            print(
                f"steps={self.num_timesteps:6d} | "
                f"R̄_last50={np.mean(recent):.2f} | n_ep={len(self.episode_rewards)}"
            )
        return True


class PPOTrainer:
    def __init__(self, seed: int = 42, out_dir: str | Path = "results"):
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _make_env(self):
        def _thunk():
            env = make_default_env(seed=self.seed)
            return Monitor(RichObsWrapper(env))

        return _thunk

    def train(self, timesteps: int = 200_000) -> PPO:
        venv = DummyVecEnv([self._make_env()])
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=self.seed,
        )
        callback = EpisodeRewardCallback(check_freq=4096)
        model.learn(total_timesteps=timesteps, callback=callback)

        model_path = self.out_dir / "ppo_forest_fire"
        model.save(str(model_path))

        metrics = {
            "algorithm": "PPO",
            "timesteps": timesteps,
            "n_episodes_logged": len(callback.episode_rewards),
            "mean_return_last_50": float(np.mean(callback.episode_rewards[-50:]))
            if callback.episode_rewards
            else None,
        }
        with open(self.out_dir / "ppo_train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        np.save(
            self.out_dir / "ppo_episode_returns.npy",
            np.array(callback.episode_rewards),
        )

        if callback.episode_rewards:
            from ..rendering import plot_learning_curve

            plot_learning_curve(
                callback.episode_rewards,
                self.out_dir / "ppo_learning_curve.png",
                window=min(50, max(5, len(callback.episode_rewards) // 10)),
            )

        print(f"Modelo PPO salvo em {model_path}.zip")
        print(json.dumps(metrics, indent=2))
        return model


def train(
    timesteps: int = 200_000,
    seed: int = 42,
    out_dir: str | Path = "results",
) -> PPO:
    return PPOTrainer(seed=seed, out_dir=out_dir).train(timesteps=timesteps)


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Forest Fire CA")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    train(timesteps=args.timesteps, seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
