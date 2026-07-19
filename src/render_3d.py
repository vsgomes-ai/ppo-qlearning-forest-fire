"""3D rendering CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .agents import QAgent, heuristic_policy, make_ppo_policy, q_policy, random_policy
from .config import make_default_env
from .env import NOOP_ACTION, ForestFireEnv
from .evaluation import run_episode
from .rendering import (
    save_episode_gif_3d,
    save_episode_gif_3d_compare,
    save_episode_gif_3d_grid4,
    save_grid_figure_3d,
)


def make_env(dramatic: bool, seed: int) -> ForestFireEnv:
    return make_default_env(seed=seed)


def build_policy(name: str, env: ForestFireEnv, q_path: str, seed: int, ppo_path: str = "results/ppo_forest_fire"):
    rng = np.random.default_rng(seed)
    if name == "Q-learning":
        ns = env.n_regions * env.n_fire_bins * env.n_tree_bins
        agent = QAgent(n_actions=env.action_space.n, n_states_=ns)
        agent.load(q_path)
        agent.epsilon = 0.0
        return (lambda obs: q_policy(agent, obs, env)), "Q-learning (obs compacta)"
    if name == "PPO":
        return make_ppo_policy(ppo_path, env), "PPO (obs estruturada)"
    if name == "heurística":
        return (lambda obs: heuristic_policy(env)), "Heuristica (centroide)"
    if name == "aleatório":
        return (lambda obs: random_policy(env, rng)), "Aleatorio"
    return (lambda obs: NOOP_ACTION), "Sem controle"


def main():
    parser = argparse.ArgumentParser(description="3D GIF of Forest Fire CA + RL")
    parser.add_argument("--q-path", type=str, default="results/q_table.npz")
    parser.add_argument("--ppo-path", type=str, default="results/ppo_forest_fire")
    parser.add_argument("--seed", type=int, default=340)
    parser.add_argument(
        "--policy",
        choices=["Q-learning", "heurística", "aleatório", "ninguém", "PPO"],
        default="heurística",
    )
    parser.add_argument("--out", type=str, default="results/forest_fire_3d.gif")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--dramatic", action="store_true", default=True)
    parser.add_argument("--no-dramatic", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare4", action="store_true")
    parser.add_argument(
        "--compare-right",
        choices=["Q-learning", "heurística", "aleatório", "PPO"],
        default="Q-learning",
    )
    args = parser.parse_args()
    dramatic = args.dramatic and not args.no_dramatic
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def pol(name, env):
        return build_policy(name, env, args.q_path, args.seed, args.ppo_path)

    if args.compare4:
        names = ["aleatório", "heurística", "Q-learning", "PPO"]
        panels, saved = [], []
        for name in names:
            env = make_env(dramatic, args.seed)
            policy, title = pol(name, env)
            result = run_episode(env, policy, seed=args.seed, collect_frames=True)
            panels.append((title, result["frames"]))
            saved.append(float(result["trees_saved_frac"]))
            print(f"  {title}: saved={result['trees_saved_frac']:.1%}")
        out4 = Path(args.out)
        if "4" not in out4.stem and "algo" not in out4.stem:
            out4 = Path("results/forest_fire_3d_4algos.gif")
        save_episode_gif_3d_grid4(
            panels, out4, fps=args.fps, rotate=not args.no_rotate, saved_fracs=saved
        )
        print(f"GIF 2x2 salvo em {out4}")
        return

    if args.compare:
        env_l = make_env(dramatic, args.seed)
        env_r = make_env(dramatic, args.seed)
        pol_l, title_l = pol("ninguém", env_l)
        pol_r, title_r = pol(args.compare_right, env_r)
        left = run_episode(env_l, pol_l, seed=args.seed, collect_frames=True)
        right = run_episode(env_r, pol_r, seed=args.seed, collect_frames=True)
        compare_path = Path(args.out)
        if "compare" not in compare_path.stem:
            compare_path = Path("results/forest_fire_3d_compare.gif")
        save_episode_gif_3d_compare(
            left["frames"],
            right["frames"],
            compare_path,
            title_left=title_l,
            title_right=title_r,
            fps=args.fps,
            rotate=not args.no_rotate,
        )
        print(f"GIF compare salvo em {compare_path}")
        return

    env = make_env(dramatic, args.seed)
    policy, title = pol(args.policy, env)
    result = run_episode(env, policy, seed=args.seed, collect_frames=True)
    frames = result["frames"]
    save_grid_figure_3d(frames[0], out.with_name(out.stem + "_t0.png"), title=f"{title} inicio", t=0)
    save_grid_figure_3d(
        frames[-1],
        out.with_name(out.stem + "_tf.png"),
        title=f"{title} fim",
        t=max(0, len(frames) - 1),
    )
    save_episode_gif_3d(
        frames, out, fps=args.fps, rotate=not args.no_rotate, title_prefix=title
    )
    print(f"GIF 3D salvo em {out} | saved={result['trees_saved_frac']:.1%}")


if __name__ == "__main__":
    main()
