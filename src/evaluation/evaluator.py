"""Multi-policy evaluation on shared seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..agents import QAgent, heuristic_policy, make_ppo_policy, q_policy, random_policy
from ..config import make_default_env
from ..rendering import plot_policy_comparison, save_episode_gif, save_grid_figure
from .runner import EpisodeRunner
from .stats import compare_all_policies


def _resolve_ppo_path(ppo_path: str | Path | None) -> Path | None:
    if not ppo_path:
        return None
    base = Path(ppo_path)
    for c in (Path(str(base) + ".zip"), base.with_suffix(".zip"), base):
        if c.exists():
            return c
    return None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


class PolicyEvaluator:
    """Evaluate Random / Heuristic / Q / PPO on identical episode seeds."""

    def __init__(
        self,
        q_path: str | Path = "results/q_table.npz",
        ppo_path: str | Path | None = "results/ppo_forest_fire",
        n_episodes: int = 100,
        seed: int = 1000,
        out_dir: str | Path = "results",
        save_demos: bool = True,
    ):
        self.q_path = Path(q_path)
        self.ppo_path = ppo_path
        self.n_episodes = n_episodes
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.save_demos = save_demos
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> dict:
        env = make_default_env(seed=self.seed)
        ns = env.n_regions * env.n_fire_bins * env.n_tree_bins
        agent = QAgent(n_actions=env.action_space.n, n_states_=ns)
        agent.load(self.q_path)
        agent.epsilon = 0.0
        rng = np.random.default_rng(self.seed)

        policies = {
            "aleatório": lambda obs: random_policy(env, rng),
            "heurística": lambda obs: heuristic_policy(env),
            "Q-learning": lambda obs: q_policy(agent, obs, env),
        }
        ppo_file = _resolve_ppo_path(self.ppo_path)
        if ppo_file is not None:
            load_stem = ppo_file.with_suffix("") if ppo_file.suffix == ".zip" else ppo_file
            policies["PPO"] = make_ppo_policy(load_stem, env)
        else:
            print("Aviso: modelo PPO nao encontrado — avaliando so baselines/Q.")

        stats: dict[str, dict[str, float]] = {}
        saved_series: dict[str, np.ndarray] = {}
        return_series: dict[str, np.ndarray] = {}
        seeds = np.arange(self.seed, self.seed + self.n_episodes, dtype=np.int64)

        for name, policy in policies.items():
            returns, saved = [], []
            for i in range(self.n_episodes):
                collect = self.save_demos and i == 0
                result = EpisodeRunner.run(
                    env, policy, seed=int(seeds[i]), collect_frames=collect
                )
                returns.append(result["return"])
                saved.append(result["trees_saved_frac"])
                if collect:
                    safe = name.replace("/", "-")
                    save_grid_figure(
                        result["frames"][0],
                        self.out_dir / f"demo_{safe}_t0.png",
                        title=f"{name} - inicio",
                    )
                    save_grid_figure(
                        result["frames"][-1],
                        self.out_dir / f"demo_{safe}_tf.png",
                        title=f"{name} - fim",
                    )
                    save_episode_gif(
                        result["frames"], self.out_dir / f"demo_{safe}.gif", fps=5
                    )
            returns_arr = np.asarray(returns, dtype=np.float64)
            saved_arr = np.asarray(saved, dtype=np.float64)
            return_series[name] = returns_arr
            saved_series[name] = saved_arr
            stats[name] = {
                "return": float(np.mean(returns_arr)),
                "return_std": float(np.std(returns_arr)),
                "trees_saved_frac": float(np.mean(saved_arr)),
                "trees_saved_std": float(np.std(saved_arr)),
            }
            print(
                f"{name:12s} | R={stats[name]['return']:.2f}±{stats[name]['return_std']:.2f} | "
                f"saved={stats[name]['trees_saved_frac']:.3f}±{stats[name]['trees_saved_std']:.3f}"
            )

        paired_saved = compare_all_policies(saved_series, reference="PPO")
        paired_return = compare_all_policies(return_series, reference="PPO")
        stats["_paired_ttest"] = {
            "trees_saved_frac": _json_safe(paired_saved),
            "return": _json_safe(paired_return),
            "note": (
                "Two-sided paired t-tests on shared seeds. "
                "Positive mean_diff means PPO > baseline."
            ),
        }

        print("\nPaired t-tests (PPO vs others, trees_saved_frac):")
        for name, row in paired_saved["comparisons"].items():
            print(
                f"  PPO vs {name:12s} | Δ={row['mean_diff']:+.4f} | "
                f"t={row['t']:.2f} | p={row['p']:.4g} | "
                f"dz={row['cohen_dz']:.2f} | "
                f"CI95=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            )

        plot_policy_comparison(
            {k: v for k, v in stats.items() if not k.startswith("_")},
            self.out_dir / "policy_comparison.png",
        )
        with open(self.out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            self.out_dir / "eval_per_episode.npz",
            seeds=seeds,
            **{f"saved__{k}": v for k, v in saved_series.items()},
            **{f"return__{k}": v for k, v in return_series.items()},
        )
        return stats


def evaluate(**kwargs) -> dict:
    return PolicyEvaluator(**kwargs).evaluate()


def main():
    parser = argparse.ArgumentParser(description="Evaluate policies on Forest Fire CA")
    parser.add_argument("--q-path", type=str, default="results/q_table.npz")
    parser.add_argument("--ppo-path", type=str, default="results/ppo_forest_fire")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--no-demos", action="store_true")
    args = parser.parse_args()
    evaluate(
        q_path=args.q_path,
        ppo_path=args.ppo_path,
        n_episodes=args.episodes,
        seed=args.seed,
        out_dir=args.out,
        save_demos=not args.no_demos,
    )


if __name__ == "__main__":
    main()
