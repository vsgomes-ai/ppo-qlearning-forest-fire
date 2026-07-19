"""Visualization helpers for the forest-fire grid and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter

from ..ca import BURNED, BURNING, EMPTY, TREE

# empty, tree, burning, burned
CMAP = colors.ListedColormap(["#f5f0e6", "#2d6a4f", "#e63946", "#6c757d"])
BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5]
NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Map discrete states to an RGB uint8 image."""
    palette = {
        EMPTY: (245, 240, 230),
        TREE: (45, 106, 79),
        BURNING: (230, 57, 70),
        BURNED: (108, 117, 125),
    }
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for state, colour in palette.items():
        rgb[grid == state] = colour
    return rgb


def plot_grid(grid: np.ndarray, ax=None, title: str | None = None):
    ax = ax or plt.gca()
    ax.imshow(grid, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def save_grid_figure(grid: np.ndarray, path: str | Path, title: str = ""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_grid(grid, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_episode_gif(frames: list[np.ndarray], path: str | Path, fps: int = 4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(frames[0], cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("t=0")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"t={i}")
        return (im, title)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_learning_curve(rewards: list[float], path: str | Path, window: int = 50):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rewards, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(arr, alpha=0.25, color="#457b9d", label="episódio")
    if len(arr) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(arr, kernel, mode="valid")
        ax.plot(
            np.arange(window - 1, len(arr)),
            smooth,
            color="#1d3557",
            lw=2,
            label=f"média móvel ({window})",
        )
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Retorno acumulado")
    ax.set_title("Curva de aprendizado (Q-learning)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy_comparison(stats: dict[str, dict[str, float]], path: str | Path):
    """Bar chart of mean trees_saved_frac and mean return per policy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(stats.keys())
    saved = [stats[n]["trees_saved_frac"] for n in names]
    returns = [stats[n]["return"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    colours = ["#adb5bd", "#f4a261", "#2a9d8f", "#e76f51", "#457b9d"]
    axes[0].bar(names, saved, color=colours[: len(names)])
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Fração de árvores salvas")
    axes[0].set_title("Biomassa preservada")
    axes[0].set_ylim(0, 1)

    axes[1].bar(names, returns, color=colours[: len(names)])
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylabel("Retorno médio")
    axes[1].set_title("Retorno acumulado")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- 3D cinematic forest (árvores com tronco + copa) ---

_TRUNK = (0.42, 0.28, 0.16, 0.98)
_TRUNK_BURN = (0.22, 0.18, 0.14, 0.95)
_STUMP = (0.30, 0.28, 0.26, 0.90)
_GROUND = (0.16, 0.20, 0.13, 1.0)
_CANOPY = [
    (0.10, 0.38, 0.22, 0.96),
    (0.14, 0.48, 0.28, 0.95),
    (0.18, 0.55, 0.32, 0.93),
]
_FLAME = [
    (0.90, 0.22, 0.08, 1.00),  # base — vermelho
    (1.00, 0.55, 0.10, 0.98),  # meio — laranja
    (1.00, 0.88, 0.30, 0.95),  # ponta — amarelo
]


def _batch_bar3d(ax, xs, ys, zs, dx, dy, dz, color, edge=None):
    if len(xs) == 0:
        return
    ax.bar3d(
        xs,
        ys,
        zs,
        dx,
        dy,
        dz,
        color=color,
        shade=True,
        edgecolor=edge if edge is not None else (0.05, 0.07, 0.05, 0.2),
        linewidth=0.12,
        zsort="average",
    )


def _draw_forest_scene(ax, grid: np.ndarray, elev: float, azim: float, title: str, t: int):
    """Draw a stylised 3D forest: ground + trunks + canopies / flames / stumps."""
    ax.cla()
    ax.set_facecolor("#071018")
    n = grid.shape[0]

    # Solo contínuo
    xx, yy = np.meshgrid(np.linspace(0, n, n + 1), np.linspace(0, n, n + 1))
    zz = np.zeros_like(xx, dtype=float) - 0.02
    ax.plot_surface(
        xx,
        yy,
        zz,
        color=_GROUND,
        shade=True,
        linewidth=0,
        antialiased=False,
        zorder=0,
    )

    # Lotes por tipo de peça
    trunks_x, trunks_y, trunks_z, trunks_h = [], [], [], []
    burn_trunks_x, burn_trunks_y, burn_trunks_z, burn_trunks_h = [], [], [], []
    canopy_layers = [([], [], [], [], [], []) for _ in range(3)]  # x,y,z,dx,dy,dz per layer
    canopy_cols = [[] for _ in range(3)]
    flame_layers = [([], [], [], [], [], []) for _ in range(3)]
    flame_cols = [[] for _ in range(3)]
    stumps_x, stumps_y, stumps_z, stumps_h = [], [], [], []
    ash_x, ash_y = [], []

    # Pequena variação de altura para não parecer “tijolo”
    rng_local = np.random.default_rng((int(grid.sum()) * 10007 + t * 17) % (2**32))

    for i in range(n):
        for j in range(n):
            state = int(grid[i, j])
            cx, cy = j + 0.5, i + 0.5
            jitter = 0.04 * float(rng_local.normal())

            if state == EMPTY:
                ash_x.append(j + 0.15)
                ash_y.append(i + 0.15)
                continue

            if state == TREE:
                th = 0.42 + 0.08 * ((i + j) % 3) + jitter * 0.3
                trunks_x.append(cx - 0.08)
                trunks_y.append(cy - 0.08)
                trunks_z.append(0.0)
                trunks_h.append(th)
                # 3 camadas de copa (estreitando para cima = “árvore”)
                widths = (0.72, 0.56, 0.38)
                z0 = th - 0.05
                for li, w in enumerate(widths):
                    h = 0.28 - 0.04 * li
                    canopy_layers[li][0].append(cx - w / 2)
                    canopy_layers[li][1].append(cy - w / 2)
                    canopy_layers[li][2].append(z0)
                    canopy_layers[li][3].append(w)
                    canopy_layers[li][4].append(w)
                    canopy_layers[li][5].append(h)
                    canopy_cols[li].append(_CANOPY[(i + j + li) % len(_CANOPY)])
                    z0 += h * 0.85
                continue

            if state == BURNING:
                th = 0.28
                burn_trunks_x.append(cx - 0.07)
                burn_trunks_y.append(cy - 0.07)
                burn_trunks_z.append(0.0)
                burn_trunks_h.append(th)
                # Chama em 3 andares (larga em baixo → ponta em cima)
                widths = (0.55, 0.38, 0.22)
                heights = (0.35, 0.40, 0.45)
                z0 = th
                flick = 0.06 * np.sin(0.9 * t + 0.4 * i + 0.3 * j)
                for li, (w, h) in enumerate(zip(widths, heights)):
                    flame_layers[li][0].append(cx - w / 2)
                    flame_layers[li][1].append(cy - w / 2)
                    flame_layers[li][2].append(z0)
                    flame_layers[li][3].append(w)
                    flame_layers[li][4].append(w)
                    flame_layers[li][5].append(h + flick * (1 - li * 0.3))
                    flame_cols[li].append(_FLAME[li])
                    z0 += h * 0.75
                continue

            if state == BURNED:
                stumps_x.append(cx - 0.12)
                stumps_y.append(cy - 0.12)
                stumps_z.append(0.0)
                stumps_h.append(0.18 + 0.04 * ((i * 3 + j) % 3))
                ash_x.append(j + 0.1)
                ash_y.append(i + 0.1)

    # Cinzas / solo marcado (blocos baixos)
    if ash_x:
        _batch_bar3d(
            ax,
            ash_x,
            ash_y,
            [0.0] * len(ash_x),
            0.7,
            0.7,
            0.06,
            color=(0.45, 0.40, 0.32, 0.55),
            edge=(0.2, 0.18, 0.12, 0.15),
        )

    if trunks_x:
        _batch_bar3d(
            ax,
            trunks_x,
            trunks_y,
            trunks_z,
            0.16,
            0.16,
            trunks_h,
            color=_TRUNK,
            edge=(0.2, 0.12, 0.08, 0.35),
        )

    if burn_trunks_x:
        _batch_bar3d(
            ax,
            burn_trunks_x,
            burn_trunks_y,
            burn_trunks_z,
            0.14,
            0.14,
            burn_trunks_h,
            color=_TRUNK_BURN,
            edge=(0.1, 0.08, 0.06, 0.35),
        )

    for li in range(3):
        xs, ys, zs, dxs, dys, dzs = canopy_layers[li]
        if xs:
            # bar3d aceita escalares ou arrays para dx/dy/dz
            ax.bar3d(
                xs, ys, zs, dxs, dys, dzs,
                color=canopy_cols[li],
                shade=True,
                edgecolor=(0.05, 0.15, 0.08, 0.25),
                linewidth=0.1,
                zsort="average",
            )

    for li in range(3):
        xs, ys, zs, dxs, dys, dzs = flame_layers[li]
        if xs:
            ax.bar3d(
                xs, ys, zs, dxs, dys, dzs,
                color=flame_cols[li],
                shade=True,
                edgecolor=(0.4, 0.15, 0.05, 0.2),
                linewidth=0.08,
                zsort="average",
            )

    if stumps_x:
        _batch_bar3d(
            ax,
            stumps_x,
            stumps_y,
            stumps_z,
            0.24,
            0.24,
            stumps_h,
            color=_STUMP,
            edge=(0.1, 0.1, 0.1, 0.3),
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(-0.05, 2.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#152033")
        axis.line.set_color("#152033")
    ax.grid(False)
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    burning = int(np.sum(grid == BURNING))
    trees = int(np.sum(grid == TREE))
    ax.set_title(
        f"{title}\nt={t}  |  fogo={burning}  |  arvores={trees}",
        color="#e8eef8",
        fontsize=10,
        pad=6,
    )


def _pad_frames(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    if len(frames) >= length:
        return frames[:length]
    if not frames:
        raise ValueError("empty frames")
    return frames + [frames[-1]] * (length - len(frames))


def save_episode_gif_3d(
    frames: list[np.ndarray],
    path: str | Path,
    fps: int = 5,
    dpi: int = 110,
    rotate: bool = True,
    title_prefix: str = "Forest Fire CA + RL",
):
    """Cinematic 3D forest animation (trunk + canopy + flames)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("frames must be non-empty")

    fig = plt.figure(figsize=(7.0, 6.0), facecolor="#071018")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90)

    base_elev, base_azim = 26, -58

    def update(t):
        if rotate:
            azim = base_azim + (t / max(1, len(frames) - 1)) * 50
            elev = base_elev + 2.5 * np.sin(2 * np.pi * t / max(1, len(frames)))
        else:
            azim, elev = base_azim, base_elev
        _draw_forest_scene(ax, frames[t], elev, azim, title_prefix, t)
        return []

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=max(1, 1000 // fps), blit=False
    )
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_episode_gif_3d_compare(
    frames_left: list[np.ndarray],
    frames_right: list[np.ndarray],
    path: str | Path,
    title_left: str = "Sem controle",
    title_right: str = "Com RL / controle",
    fps: int = 5,
    dpi: int = 100,
    rotate: bool = True,
):
    """Side-by-side 3D GIF: uncontrolled fire vs controlled policy."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(len(frames_left), len(frames_right))
    left = _pad_frames(frames_left, n_frames)
    right = _pad_frames(frames_right, n_frames)

    fig = plt.figure(figsize=(12.5, 6.0), facecolor="#071018")
    ax_l = fig.add_subplot(121, projection="3d", computed_zorder=False)
    ax_r = fig.add_subplot(122, projection="3d", computed_zorder=False)
    fig.suptitle(
        "Autômato celular de incêndio  ·  sem controle  vs  com intervenção",
        color="#e8eef8",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88, wspace=0.08)

    base_elev, base_azim = 26, -58

    def update(t):
        if rotate:
            azim = base_azim + (t / max(1, n_frames - 1)) * 50
            elev = base_elev + 2.5 * np.sin(2 * np.pi * t / max(1, n_frames))
        else:
            azim, elev = base_azim, base_elev
        _draw_forest_scene(ax_l, left[t], elev, azim, title_left, t)
        _draw_forest_scene(ax_r, right[t], elev, azim, title_right, t)
        return []

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=max(1, 1000 // fps), blit=False
    )
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_episode_gif_3d_grid4(
    panels: list[tuple[str, list[np.ndarray]]],
    path: str | Path,
    fps: int = 5,
    dpi: int = 85,
    rotate: bool = True,
    saved_fracs: list[float] | None = None,
):
    """
    GIF 2×2 com quatro políticas no mesmo mapa inicial.
    panels: [(titulo, frames), ...] comprimento 4, ordem:
    aleatório, heurística, Q-learning, PPO.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if len(panels) != 4:
        raise ValueError("panels must contain exactly 4 (title, frames) pairs")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(len(f) for _, f in panels)
    padded = [(title, _pad_frames(frames, n_frames)) for title, frames in panels]

    fig = plt.figure(figsize=(12.0, 11.0), facecolor="#071018")
    axes = [
        fig.add_subplot(221, projection="3d", computed_zorder=False),
        fig.add_subplot(222, projection="3d", computed_zorder=False),
        fig.add_subplot(223, projection="3d", computed_zorder=False),
        fig.add_subplot(224, projection="3d", computed_zorder=False),
    ]
    fig.suptitle(
        "Autômato celular de incêndio  ·  4 políticas (mesmo mapa inicial)",
        color="#e8eef8",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.06, hspace=0.12)

    base_elev, base_azim = 26, -58

    def update(t):
        if rotate:
            azim = base_azim + (t / max(1, n_frames - 1)) * 50
            elev = base_elev + 2.5 * np.sin(2 * np.pi * t / max(1, n_frames))
        else:
            azim, elev = base_azim, base_elev
        for i, (title, frames) in enumerate(padded):
            label = title
            if saved_fracs is not None:
                label = f"{title}  ·  salvas={saved_fracs[i]:.0%}"
            _draw_forest_scene(axes[i], frames[t], elev, azim, label, t)
        return []

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=max(1, 1000 // fps), blit=False
    )
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_grid_figure_3d(
    grid: np.ndarray,
    path: str | Path,
    title: str = "",
    t: int = 0,
):
    """Single 3D still frame (PNG) with tree geometry."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6), facecolor="#071018")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    _draw_forest_scene(
        ax,
        grid,
        elev=26,
        azim=-58,
        title=title or "Forest Fire 3D",
        t=int(t),
    )
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
