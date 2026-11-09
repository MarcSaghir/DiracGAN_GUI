"""Interactive DiracGAN explorer: vector fields and gradient-descent trajectories
for a range of GAN objectives and regularizers."""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from diracgan.gans import (
    GAN,
    NSGAN,
    WGAN,
    WGAN_GP,
    GAN_Consensus,
    GAN_GradPenalty,
    GAN_InstNoise,
    NSGAN_GradPenalty,
)
from diracgan.simulate import trajectory_altgd, trajectory_simgd
from new_gans import LeCamGAN, LSGAN

GRID = np.linspace(-2, 2, 10)
LIM = 2.25

ARROW_COLOR = "#3b4252"
TRAJ_COLOR = "#5e81ac"
START_COLOR = "#bf616a"

# key: (label, default)
PARAMS = {
    "wgan_clip": ("WGAN clip", 1.0),
    "wgan_gp_reg": ("WGAN-GP reg", 0.7),
    "wgan_gp_target": ("WGAN-GP target", 1.0),
    "instnoise_std": ("Inst. noise std", 0.7),
    "gp_reg": ("GAN-GP reg", 0.3),
    "consensus_reg": ("Consensus reg", 1.0),
    "nsgan_gp_reg": ("NSGAN-GP reg", 0.3),
    "lecam_reg": ("LeCam reg", -0.3),
    "lecam_anchor": ("LeCam anchor", 0.1),
}
DEFAULT_PARAMS = {k: v for k, (_, v) in PARAMS.items()}


def make_gans(p):
    return {
        "GAN": GAN(),
        "NSGAN": NSGAN(),
        "WGAN": WGAN(p["wgan_clip"]),
        "WGAN-GP": WGAN_GP(p["wgan_gp_reg"], p["wgan_gp_target"]),
        "Instance noise": GAN_InstNoise(p["instnoise_std"]),
        "GAN + grad. penalty": GAN_GradPenalty(p["gp_reg"]),
        "Consensus opt.": GAN_Consensus(p["consensus_reg"]),
        "NSGAN + grad. penalty": NSGAN_GradPenalty(p["nsgan_gp_reg"]),
        "LSGAN": LSGAN(),
        "LeCam": LeCamGAN(p["lecam_reg"], p["lecam_anchor"]),
    }


def simulate(gan, theta0, psi0, n_steps, h_g, h_d, alternating=False, g_steps=1, d_steps=1):
    # diverging objectives overflow; matplotlib just skips inf/nan points
    with np.errstate(over="ignore", invalid="ignore"):
        if alternating:
            return trajectory_altgd(gan, theta0, psi0, nsteps=n_steps, hs_g=h_g, hs_d=h_d,
                                    gsteps=g_steps, dsteps=d_steps)
        return trajectory_simgd(gan, theta0, psi0, nsteps=n_steps, hs_g=h_g, hs_d=h_d)


def draw_field(ax, title, gan):
    ax.clear()
    T, P = np.meshgrid(GRID, GRID)
    ax.quiver(T, P, *gan(T, P), color=ARROW_COLOR)
    if isinstance(gan, WGAN):
        ax.axhspan(-LIM, -gan.clip, color="0.85", zorder=0)
        ax.axhspan(gan.clip, LIM, color="0.85", zorder=0)
    ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), xticks=range(-2, 3), yticks=range(-2, 3),
           xlabel=r"$\theta$", ylabel=r"$\psi$", title=title, aspect="equal")
    ax.label_outer()


def draw_trajectory(ax, trajectory, n=None):
    """Plot start point and first n steps; returns the line for later updates."""
    thetas, psis = trajectory
    ax.plot(thetas[0], psis[0], "o", color=START_COLOR, zorder=3)
    (line,) = ax.plot(thetas[:n], psis[:n], "^", mfc="none", mec=TRAJ_COLOR, alpha=0.8)
    return line


class DiracGANApp:
    def __init__(self, root):
        self.root = root
        root.title("DiracGAN Convergence Explorer")
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")

        self.theta0 = tk.DoubleVar(value=1.0)
        self.psi0 = tk.DoubleVar(value=1.0)
        self.n_steps = tk.IntVar(value=500)
        self.h_g = tk.DoubleVar(value=0.2)
        self.h_d = tk.DoubleVar(value=0.2)
        self.g_steps = tk.IntVar(value=1)
        self.d_steps = tk.IntVar(value=1)
        self.alternating = tk.BooleanVar(value=False)
        self.params = dict(DEFAULT_PARAMS)
        self.param_vars = {k: tk.StringVar(value=str(v)) for k, v in self.params.items()}

        self.animating = False
        self.index = 0
        self.lines = []
        self.trajectories = []

        # controls are packed first so the canvas can't squeeze them out
        self._build_controls()
        self.fig = Figure(figsize=(20, 8.5), layout="constrained")
        self.axs = self.fig.subplots(2, 5, sharex=True, sharey=True).flatten()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.apply()

    def _build_controls(self):
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side=tk.BOTTOM, fill=tk.X)
        controls.columnconfigure(0, weight=1)

        sliders = ttk.LabelFrame(controls, text="Training", padding=6)
        sliders.grid(row=0, column=0, sticky="nsew", padx=4)
        slider_specs = [
            ("Start θ", self.theta0, -2, 2, 0.01),
            ("Start ψ", self.psi0, -2, 2, 0.01),
            ("Generator learning rate", self.h_g, 0.01, 1, 0.01),
            ("Discriminator learning rate", self.h_d, 0.01, 1, 0.01),
            ("Number of steps", self.n_steps, 100, 1000, 10),
            ("Generator steps (alternating)", self.g_steps, 1, 10, 1),
            ("Discriminator steps (alternating)", self.d_steps, 1, 10, 1),
        ]
        for i, (label, var, lo, hi, res) in enumerate(slider_specs):
            tk.Scale(sliders, label=label, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL).grid(row=i // 4, column=i % 4, sticky="ew", padx=4)
        sliders.columnconfigure((0, 1, 2, 3), weight=1)

        hyper = ttk.LabelFrame(controls, text="Hyperparameters", padding=6)
        hyper.grid(row=0, column=1, sticky="nsew", padx=4)
        for i, (key, (label, _)) in enumerate(PARAMS.items()):
            row, col = i // 3, (i % 3) * 2
            ttk.Label(hyper, text=label).grid(row=row, column=col, sticky="e", padx=(8, 2), pady=3)
            ttk.Entry(hyper, textvariable=self.param_vars[key], width=7).grid(row=row, column=col + 1)

        actions = ttk.LabelFrame(controls, text="Run", padding=6)
        actions.grid(row=0, column=2, sticky="nsew", padx=4)
        ttk.Checkbutton(actions, text="Alternating GD", variable=self.alternating).pack(anchor="w", pady=3)
        ttk.Button(actions, text="Apply", command=self.apply).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Animate", command=self.start_animation).pack(fill=tk.X, pady=2)
        self.pause_button = ttk.Button(actions, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(fill=tk.X, pady=2)

    def _read_params(self):
        for key, var in self.param_vars.items():
            try:
                self.params[key] = float(var.get())
            except ValueError:
                var.set(str(self.params[key]))  # keep last valid value

    def _redraw(self, n=None):
        self._read_params()
        gans = make_gans(self.params)
        self.trajectories, self.lines = [], []
        for ax, (title, gan) in zip(self.axs, gans.items()):
            traj = simulate(gan, self.theta0.get(), self.psi0.get(), self.n_steps.get(),
                            self.h_g.get(), self.h_d.get(), self.alternating.get(),
                            self.g_steps.get(), self.d_steps.get())
            draw_field(ax, title, gan)
            self.lines.append(draw_trajectory(ax, traj, n))
            self.trajectories.append(traj)
        self.canvas.draw()

    def apply(self):
        self.animating = False
        self._redraw()

    def start_animation(self):
        self._redraw(n=1)
        self.index = 1
        self.animating = True
        self.pause_button.config(text="Pause")
        self._step()

    def toggle_pause(self):
        self.animating = not self.animating
        self.pause_button.config(text="Pause" if self.animating else "Resume")
        if self.animating:
            self._step()

    def _step(self):
        if self.index > len(self.trajectories[0][0]):
            self.animating = False
        if not self.animating:
            return
        for line, (thetas, psis) in zip(self.lines, self.trajectories):
            line.set_data(thetas[: self.index], psis[: self.index])
        self.canvas.draw()
        self.index += 1
        self.root.after(5, self._step)


if __name__ == "__main__":
    root = tk.Tk()
    DiracGANApp(root)
    root.mainloop()
