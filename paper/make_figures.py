"""Figuras de publicação exclusivamente com Plotly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# Paleta sóbria (sem neon/roxo genérico)
COLOURS = {
    "Aleatório": "#5c6b73",
    "Heurística": "#c4703a",
    "Q-learning": "#2f5d8a",
    "PPO": "#1f7a4d",
}
ORDER = ["Aleatório", "Heurística", "Q-learning", "PPO"]
KEY = {
    "Aleatório": "aleatório",
    "Heurística": "heurística",
    "Q-learning": "Q-learning",
    "PPO": "PPO",
}

LAYOUT = dict(
    font=dict(family="Liberation Serif, Times New Roman, serif", size=13, color="#1a1a1a"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=56, r=24, t=48, b=48),
)


def _write(fig: go.Figure, stem: str, width: int = 900, height: int = 360):
    fig.update_layout(**LAYOUT)
    pdf = FIG / f"{stem}.pdf"
    png = FIG / f"{stem}.png"
    fig.write_image(str(pdf), width=width, height=height, scale=2)
    fig.write_image(str(png), width=width, height=height, scale=2)
    print("OK", stem)


def _smooth(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(arr) < window:
        return np.arange(len(arr)), arr.astype(float)
    ker = np.ones(window) / window
    sm = np.convolve(arr, ker, mode="valid")
    return np.arange(window - 1, len(arr)), sm


def fig_policy_bars():
    with open(RESULTS / "eval_metrics.json", encoding="utf-8") as f:
        stats = json.load(f)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Biomassa preservada",
            "Retorno acumulado",
        ),
        horizontal_spacing=0.12,
    )

    saved = [stats[KEY[n]]["trees_saved_frac"] for n in ORDER]
    saved_e = [stats[KEY[n]]["trees_saved_std"] for n in ORDER]
    rets = [stats[KEY[n]]["return"] for n in ORDER]
    ret_e = [stats[KEY[n]]["return_std"] for n in ORDER]
    cols = [COLOURS[n] for n in ORDER]

    fig.add_trace(
        go.Bar(
            x=ORDER,
            y=saved,
            error_y=dict(type="data", array=saved_e, thickness=1.2, width=6, color="#333"),
            marker=dict(color=cols, line=dict(width=0)),
            width=0.62,
            showlegend=False,
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ORDER,
            y=rets,
            error_y=dict(type="data", array=ret_e, thickness=1.2, width=6, color="#333"),
            marker=dict(color=cols, line=dict(width=0)),
            width=0.62,
            showlegend=False,
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_width=1, line_color="#444", row=1, col=2)

    fig.update_yaxes(
        title_text="Fração de árvores preservadas",
        range=[0, 1],
        gridcolor="#e6e6e6",
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Retorno médio",
        gridcolor="#e6e6e6",
        zeroline=False,
        row=1,
        col=2,
    )
    fig.update_xaxes(tickangle=-18, row=1, col=1)
    fig.update_xaxes(tickangle=-18, row=1, col=2)
    fig.update_layout(
        title=dict(text="Comparação das políticas (100 episódios, mesmas sementes)", x=0.02),
        bargap=0.28,
    )
    _write(fig, "fig_policy_comparison", width=960, height=340)


def fig_violin():
    data = np.load(FIG / "boxplot_data.npz")
    fig = go.Figure()
    for name in ORDER:
        arr = data[KEY[name]]
        fig.add_trace(
            go.Violin(
                y=arr,
                name=name,
                line_color=COLOURS[name],
                fillcolor=COLOURS[name],
                opacity=0.55,
                meanline_visible=True,
                box_visible=True,
                points="outliers",
                pointpos=0,
                jitter=0.25,
                marker=dict(size=4, opacity=0.55, color=COLOURS[name]),
                bandwidth=0.045,
            )
        )
    fig.update_layout(
        title=dict(
            text="Distribuição da biomassa preservada (100 episódios)",
            x=0.02,
        ),
        yaxis=dict(
            title="Fração de árvores preservadas",
            range=[-0.02, 1.05],
            gridcolor="#e6e6e6",
            zeroline=False,
        ),
        xaxis=dict(title=""),
        showlegend=False,
        violingap=0.18,
        violinmode="group",
    )
    _write(fig, "fig_violin_saved", width=820, height=340)


def fig_learning():
    q_ret = np.load(RESULTS / "episode_returns.npy")
    ppo_ret = np.load(RESULTS / "ppo_episode_returns.npy")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Q-learning (3000 episódios)", "PPO (~200000 timesteps)"),
        horizontal_spacing=0.10,
        shared_yaxes=True,
    )

    # Q
    fig.add_trace(
        go.Scatter(
            y=q_ret,
            mode="lines",
            line=dict(width=0.6, color="rgba(47,93,138,0.22)"),
            name="episódio",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    xq, sq = _smooth(q_ret, 50)
    fig.add_trace(
        go.Scatter(
            x=xq,
            y=sq,
            mode="lines",
            line=dict(width=2.4, color=COLOURS["Q-learning"]),
            name="média móvel 50",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # PPO: subsample raw for file size / clarity
    step = max(1, len(ppo_ret) // 2500)
    fig.add_trace(
        go.Scatter(
            y=ppo_ret[::step],
            mode="lines",
            line=dict(width=0.5, color="rgba(31,122,77,0.18)"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    xp, sp = _smooth(ppo_ret, 100)
    fig.add_trace(
        go.Scatter(
            x=xp,
            y=sp,
            mode="lines",
            line=dict(width=2.4, color=COLOURS["PPO"]),
            name="média móvel 100",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Episódio", gridcolor="#eaeaea", row=1, col=1)
    fig.update_xaxes(title_text="Episódio (log do treino)", gridcolor="#eaeaea", row=1, col=2)
    fig.update_yaxes(title_text="Retorno acumulado", gridcolor="#eaeaea", row=1, col=1)
    fig.update_yaxes(gridcolor="#eaeaea", row=1, col=2)
    fig.update_layout(
        title=dict(text="Curvas de aprendizado", x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0.55),
    )
    _write(fig, "fig_learning_curves", width=960, height=340)


def fig_paired_diff():
    """Paired Δ = PPO - baseline for trees_saved, one cloud per contrast."""
    data = np.load(RESULTS / "eval_per_episode.npz")
    ppo = data["saved__PPO"]
    contrasts = [
        ("vs. aleatório", "saved__aleatório", COLOURS["Aleatório"]),
        ("vs. heurística", "saved__heurística", COLOURS["Heurística"]),
        ("vs. Q-learning", "saved__Q-learning", COLOURS["Q-learning"]),
    ]

    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, (label, key, colour) in enumerate(contrasts):
        delta = ppo - data[key]
        # strip / jitter points
        x = np.full(delta.shape, i) + rng.uniform(-0.12, 0.12, size=delta.shape)
        fig.add_trace(
            go.Box(
                y=delta,
                x=[label] * len(delta),
                name=label,
                marker_color=colour,
                boxpoints=False,
                line=dict(color=colour, width=1.6),
                fillcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=delta,
                mode="markers",
                marker=dict(size=6, color=colour, opacity=0.55, line=dict(width=0)),
                name=label,
                showlegend=False,
                hovertemplate=f"{label}<br>Δ=%{{y:.3f}}<extra></extra>",
            )
        )
        # mean marker
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[float(np.mean(delta))],
                mode="markers",
                marker=dict(symbol="diamond", size=11, color="#1a1a1a"),
                showlegend=False,
                hovertemplate=f"média Δ={float(np.mean(delta)):.3f}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line=dict(color="#444444", width=1.4, dash="dash"))
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(contrasts))),
        ticktext=[c[0] for c in contrasts],
        gridcolor="#eaeaea",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Δ biomassa (PPO − baseline)",
        gridcolor="#eaeaea",
        zeroline=False,
    )
    fig.update_layout(
        title=dict(
            text="Diferenças pareadas por semente (n = 100)",
            x=0.02,
            xanchor="left",
        ),
    )
    _write(fig, "fig_paired_diff", width=820, height=400)


def fig_3d_stills():
    from src.evaluate import run_episode
    from src.render_3d import build_policy, make_env
    from src.viz import save_grid_figure_3d

    seed = 340
    q_path = str(RESULTS / "q_table.npz")

    env = make_env(False, seed)
    pol, _ = build_policy("ninguém", env, q_path, seed)
    res0 = run_episode(env, pol, seed=seed, collect_frames=True)
    save_grid_figure_3d(
        res0["frames"][0],
        FIG / "fig3d_t0.png",
        title="Estado inicial",
        t=0,
    )

    specs = [
        ("ninguém", "fig3d_wild_tf", "Sem controle (fim)"),
        ("heurística", "fig3d_heu_tf", "Heuristica (fim)"),
        ("Q-learning", "fig3d_ql_tf", "Q-learning (fim)"),
        ("PPO", "fig3d_ppo_tf", "PPO (fim)"),
    ]
    for name, stem, label in specs:
        env = make_env(False, seed)
        pol, _ = build_policy(name, env, q_path, seed)
        res = run_episode(env, pol, seed=seed, collect_frames=True)
        t_final = max(0, len(res["frames"]) - 1)
        save_grid_figure_3d(
            res["frames"][-1],
            FIG / f"{stem}.png",
            title=f"{label} (salvas = {res['trees_saved_frac']:.0%})",
            t=t_final,
        )
        print(f"  {label}: saved={res['trees_saved_frac']:.1%} t={t_final}")
    print("OK fig_3d_stills")


if __name__ == "__main__":
    # garantir dados do violino
    if not (FIG / "boxplot_data.npz").exists():
        print("Aviso: falta boxplot_data.npz; pulando fig_violin.")
        run_violin = False
    else:
        run_violin = True
    fig_policy_bars()
    if run_violin:
        fig_violin()
    fig_learning()
    fig_paired_diff()
    # stills 3D são caros; rode explicitamente se precisar regenerar
    # fig_3d_stills()
    print("Figuras em", FIG)
