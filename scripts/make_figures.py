"""Render the README figures into assets/. Run from the repo root: python -m scripts.make_figures"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from dirac_gui import DEFAULT_PARAMS, draw_field, draw_trajectory, make_gans, simulate

OUT = Path("assets")
SETTINGS = dict(theta0=1.0, psi0=1.0, n_steps=500, h_g=0.2, h_d=0.2)


def grid(alternating=False):
    gans = make_gans(DEFAULT_PARAMS)
    fig, axs = plt.subplots(2, 5, figsize=(17, 7.4), sharex=True, sharey=True, layout="constrained")
    trajs = []
    for ax, (title, gan) in zip(axs.flat, gans.items()):
        traj = simulate(gan, **SETTINGS, alternating=alternating)
        draw_field(ax, title, gan)
        trajs.append((ax, traj))
    return fig, trajs


def static(name, alternating):
    fig, trajs = grid(alternating)
    for ax, traj in trajs:
        draw_trajectory(ax, traj)
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)


def animation(name, every=5):
    fig, trajs = grid()
    lines = [(draw_trajectory(ax, traj, 1), traj) for ax, traj in trajs]
    frames = range(1, SETTINGS["n_steps"] + 2, every)

    def update(n):
        for line, (thetas, psis) in lines:
            line.set_data(thetas[:n], psis[:n])

    FuncAnimation(fig, update, frames=frames).save(OUT / name, writer=PillowWriter(fps=20), dpi=50)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    np.random.seed(0)  # instance noise field is stochastic
    static("simultaneous.png", alternating=False)
    static("alternating.png", alternating=True)
    animation("animation.gif")
