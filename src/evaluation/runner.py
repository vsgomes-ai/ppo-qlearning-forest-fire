"""Single-episode execution under a policy."""

from __future__ import annotations

from ..env import ForestFireEnv


class EpisodeRunner:
    """Runs one episode under a policy callable ``obs -> action``."""

    @staticmethod
    def run(
        env: ForestFireEnv,
        policy_fn,
        seed: int,
        collect_frames: bool = False,
    ) -> dict:
        obs, info = env.reset(seed=seed)
        total_r = 0.0
        frames = [info["grid"].copy()] if collect_frames else []
        done = False
        while not done:
            action = policy_fn(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
            if collect_frames:
                frames.append(info["grid"].copy())
        return {
            "return": total_r,
            "trees_saved_frac": float(info["trees_saved_frac"]),
            "burned": int(info["burned"]),
            "frames": frames,
        }


run_episode = EpisodeRunner.run
