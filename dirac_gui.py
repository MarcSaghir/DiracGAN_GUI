import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

from new_gans import LSGAN, LeCamGAN

from diracgan.simulate import trajectory_altgd, trajectory_simgd

# ablauf:
"""
-> Trajectories und vektoren für alle GANs berechnen
-> Ersten Punkt und Vektoren plotten und warten
"""


class DiracGANPlot:
    def __init__(self, root):
        self.running = False
        self.animating = False
        # initial
        # plot configs
        # gan configs

        self.GAN_params = {
            "WGAN_clip": 1.0,
            "WGAN_GP_reg": 0.7,
            "WGAN_GP_target": 1.0,
            "GAN_InstNoise_std": 0.7,
            "GAN_GradPenalty_reg": 0.3,
            "GAN_Consensus_reg": 1.0,
            "NSGAN_GradPenalty_reg": 0.3,
            "LeCamGAN_lambda": -0.3,  # default value for LeCamGAN
            "LeCamGAN_alpha":0.1
        }
        self.GANS = [
            GAN(),
            NSGAN(),
            WGAN(self.GAN_params["WGAN_clip"]),
            WGAN_GP(self.GAN_params["WGAN_GP_reg"], self.GAN_params["WGAN_GP_target"]),
            GAN_InstNoise(self.GAN_params["GAN_InstNoise_std"]),
            GAN_GradPenalty(self.GAN_params["GAN_GradPenalty_reg"]),
            GAN_Consensus(self.GAN_params["GAN_Consensus_reg"]),
            NSGAN_GradPenalty(self.GAN_params["NSGAN_GradPenalty_reg"]),
            LSGAN(),
            LeCamGAN(self.GAN_params["LeCamGAN_lambda"], self.GAN_params["LeCamGAN_alpha"])
        ]

        # learning rate
        self.h_d = tk.DoubleVar(value=0.2)
        self.h_g = tk.DoubleVar(value=0.2)
        # starting points
        self.theta0 = tk.DoubleVar(value=1.0)
        self.psi0 = tk.DoubleVar(value=1.0)
        # iterations
        self.n_steps = tk.IntVar(value=500)
        # steps per update
        self.gsteps = tk.IntVar(value=1)
        self.dsteps = tk.IntVar(value=1)

        self.theta_s = np.linspace(-2, 2.0, 10)
        self.psi_s = np.linspace(-2, 2, 10)

        # what kind of gradient descent (simultaneous or alternating)
        self.grad_descent = "simultaneous"

        self.root = root

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set window size to screen size
        root.geometry(f"{screen_width}x{screen_height}")

        self.root.title("DiracGAN Convergence Plotter")

        # Create a frame for the plot
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a matplotlib figure
        n_cols = len(self.GANS) // 2
        self.fig, self.axs = plt.subplots(
            2,
            n_cols,
            figsize=(4 * n_cols, 5),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        self.axs = self.axs.flatten()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---------- SLIDER ----------------
        self.start_theta_slider = tk.Scale(
            root,
            from_=self.theta_s.min(),
            to=self.theta_s.max(),
            orient=tk.HORIZONTAL,
            label="Start Theta",
            resolution=0.01,
            variable=self.theta0,
            length=100
        )
        self.start_theta_slider.pack(fill="x", expand=False)

        self.start_psi_slider = tk.Scale(
            root,
            from_=self.psi_s.min(),
            to=self.psi_s.max(),
            orient=tk.HORIZONTAL,
            label="Start Psi",
            variable=self.psi0,
            resolution=0.01,
        )
        self.start_psi_slider.pack(fill="x", expand=True)

        self.num_steps_slider = tk.Scale(
            root,
            from_=100,
            to=1000,
            orient=tk.HORIZONTAL,
            label="Number of Steps",
            variable=self.n_steps,
        )
        self.num_steps_slider.pack(fill="x", expand=True)

        self.h_g_slider = tk.Scale(
            root,
            from_=0.01,
            to=1,
            orient=tk.HORIZONTAL,
            label="Generator Learning Rate",
            variable=self.h_g,
            resolution=0.01,
        )
        self.h_g_slider.pack(fill="x", expand=True)

        self.h_d_slider = tk.Scale(
            root,
            from_=0.01,
            to=1,
            orient=tk.HORIZONTAL,
            label="Discriminator Learning Rate",
            variable=self.h_d,
            resolution=0.01,
        )
        self.h_d_slider.pack(fill="x", expand=True)

        self.dsteps_slider = tk.Scale(
            root,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            label="Discriminator Updates per Generator Update",
            variable=self.dsteps,
        )
        self.dsteps_slider.pack(fill="x", expand=True)

        self.gsteps_slider = tk.Scale(
            root,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            label="Generator Updates per Discriminator Update",
            variable=self.gsteps,
        )
        self.gsteps_slider.pack(fill="x", expand=True)

        # ---------- BUTTONS ----------------

        # animation buttons
        self.animate_button = ttk.Button(
            root, text="Start Animation", command=self.start_animation
        )
        self.animate_button.pack(side=tk.LEFT, padx=6, pady=5)
        self.end_animation_button = ttk.Button(
            root, text="Stop Animation", command=self.stop_animation
        )
        self.end_animation_button.pack(side=tk.LEFT, padx=6, pady=5)
        self.continue_animation_button = ttk.Button(
            root, text="Resume Animation", command=self.continue_animation
        )
        self.continue_animation_button.pack(side=tk.LEFT, padx=6, pady=5)

        # Button to update the plot
        self.apply_button = ttk.Button(root, text="Apply", command=self.apply_changes)
        self.apply_button.pack(side=tk.LEFT, padx=6, pady=5)
        self.toggle_button = ttk.Button(
            root,
            text="Start Trajectory",
            command=self.toggle_animation,
            state="disabled",
        )
        
        # -------- number fields --------

        self.lecam_lambda_entry = ttk.Entry(root, width=6)
        self.lecam_lambda_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.lecam_lambda_button = ttk.Button(root, text="LeCam Reg", command=self.get_lecam_lambda)
        self.lecam_lambda_button.pack(side=tk.LEFT, padx=6, pady=6)

        self.lecam_alpha_entry = ttk.Entry(root, width=6)
        self.lecam_alpha_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.lecam_alpha_button = ttk.Button(root, text="LeCam Anchor", command=self.get_lecam_alpha)
        self.lecam_alpha_button.pack(side=tk.LEFT, padx=6, pady=6)


        self.wgan_clip_entry = ttk.Entry(root, width=6)
        self.wgan_clip_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.wgan_clip_button = ttk.Button(root, text="WGAN Clip", command=self.get_wgan_clip)
        self.wgan_clip_button.pack(side=tk.LEFT, padx=6, pady=6)


        self.wgan_gp_reg_entry = ttk.Entry(root, width=6)
        self.wgan_gp_reg_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.wgan_gp_reg_button = ttk.Button(root, text="WGANGP Reg", command=self.get_wgan_gp_reg)
        self.wgan_gp_reg_button.pack(side=tk.LEFT, padx=6, pady=6)

        self.wgan_gp_target_entry = ttk.Entry(root, width=6)
        self.wgan_gp_target_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.wgan_gp_target_button = ttk.Button(root, text="WGANGP Target", command=self.get_wgan_gp_target)
        self.wgan_gp_target_button.pack(side=tk.LEFT, padx=6, pady=6)

        self.gan_instnoise_std_entry = ttk.Entry(root, width=6)
        self.gan_instnoise_std_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.gan_instnoise_std_button = ttk.Button(root, text="InstNoise Std", command=self.get_gan_instnoise_std)
        self.gan_instnoise_std_button.pack(side=tk.LEFT, padx=6, pady=6)

        self.gan_gradpenalty_reg_entry = ttk.Entry(root, width=6)
        self.gan_gradpenalty_reg_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.gan_gradpenalty_reg_button = ttk.Button(root, text="GradPen Reg", command=self.get_gan_gradpenalty_reg)
        self.gan_gradpenalty_reg_button.pack(side=tk.LEFT, padx=6, pady=6)

        self.gan_consensus_reg_entry = ttk.Entry(root, width=6)
        self.gan_consensus_reg_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.gan_consensus_reg_button = ttk.Button(root, text="Consensus Reg", command=self.get_gan_consensus_reg)
        self.gan_consensus_reg_button.pack(side=tk.LEFT, padx=6, pady=6)

        self.nsgan_gradpenalty_reg_entry = ttk.Entry(root, width=6)
        self.nsgan_gradpenalty_reg_entry.pack(side=tk.LEFT, padx=6, pady=6)
        self.nsgan_gradpenalty_reg_button = ttk.Button(root, text="NSGAN GradPen Reg", command=self.get_nsgan_gradpenalty_reg)
        self.nsgan_gradpenalty_reg_button.pack(side=tk.LEFT, padx=6, pady=6)

        self.check_gd = tk.BooleanVar()
        self.gd_checkbox = ttk.Checkbutton(root, text="Enable feature", variable=self.check_gd)
        self.gd_checkbox.pack(padx=5, pady=5)

        # Button to read checkbox state
        self.button = ttk.Button(root, text="Alternating", command=self.set_gd)
        self.button.pack(padx=5, pady=5)


        # self.grad_descent_combobox =
        # ttk.Combobox(root, values=["simultaneous", "alternating"],
        # state="readonly", command=self.set_gradient_descent)
        # self.grad_descent_combobox.set("simultaneous")

        # Initial plot
        self.refresh_plot()
        self.update_plot()

    def set_gd(self):
        if self.check_gd.get():
            self.set_gradient_descent("alternating")
        else:
            self.set_gradient_descent("simultaneous")
        #self.apply_changes()


    def get_lecam_lambda(self):
        try:
            value = float(self.lecam_lambda_entry.get())
            self.update_GAN_param("LeCamGAN_lambda", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def get_lecam_alpha(self):
        try:
            value = float(self.lecam_alpha_entry.get())
            self.update_GAN_param("LeCamGAN_alpha", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def get_wgan_clip(self):
        try:
            value = float(self.wgan_clip_entry.get())
            self.update_GAN_param("WGAN_clip", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def get_wgan_gp_reg(self):
        try:
            value = float(self.wgan_gp_reg_entry.get())
            self.update_GAN_param("WGAN_GP_reg", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def get_wgan_gp_target(self):
        try:
            value = float(self.wgan_gp_target_entry.get())
            self.update_GAN_param("WGAN_GP_target", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


    def get_gan_instnoise_std(self):
        try:
            value = float(self.gan_instnoise_std_entry.get())
            self.update_GAN_param("GAN_InstNoise_std", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def get_gan_gradpenalty_reg(self):
        try:
            value = float(self.gan_gradpenalty_reg_entry.get())
            self.update_GAN_param("GAN_GradPenalty_reg", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


    def get_gan_consensus_reg(self):
        try:
            value = float(self.gan_consensus_reg_entry.get())
            self.update_GAN_param("GAN_Consensus_reg", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


    def get_nsgan_gradpenalty_reg(self):
        try:
            value = float(self.nsgan_gradpenalty_reg_entry.get())
            self.update_GAN_param("NSGAN_GradPenalty_reg", value)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    


    def toggle_animation(self):
        if not self.running:
            self.running = True
            self.toggle_button.config(text="Stop Trajectory")
            self.init_plot_values()
            self.animate()
        else:
            self.running = False
            self.toggle_button.config(text="Start Trajectory")

    def activate_buttons(self):
        self.toggle_button.config(state="normal")
        #self.next_step_button.config(state="normal")
        #self.refresh_button.config(state="normal")

    def apply_changes(self):
        self.activate_buttons()
        self.refresh_plot()
        self.update_plot()

    def refresh_plot(self):
        self.GANS = [
            GAN(),
            NSGAN(),
            WGAN(self.GAN_params["WGAN_clip"]),
            WGAN_GP(self.GAN_params["WGAN_GP_reg"], self.GAN_params["WGAN_GP_target"]),
            GAN_InstNoise(self.GAN_params["GAN_InstNoise_std"]),
            GAN_GradPenalty(self.GAN_params["GAN_GradPenalty_reg"]),
            GAN_Consensus(self.GAN_params["GAN_Consensus_reg"]),
            NSGAN_GradPenalty(self.GAN_params["NSGAN_GradPenalty_reg"]),
            LSGAN(),
            LeCamGAN(self.GAN_params["LeCamGAN_lambda"], self.GAN_params["LeCamGAN_alpha"])
        ]
        self.init_plot_values()
        self.step = 0
        for ax in self.axs:
            ax.clear()

    def init_plot_values(self):
        self.trajectories = []
        self.arrows = []

        for gan in self.GANS:
            if self.grad_descent == "simultaneous":
                self.trajectories.append(
                    trajectory_simgd(
                        gan,
                        self.theta0.get(),
                        self.psi0.get(),
                        nsteps=self.n_steps.get(),
                        hs_d=self.h_d.get(),
                        hs_g=self.h_g.get(),
                    )
                )
            else:  # alternating mode
                self.trajectories.append(
                    trajectory_altgd(
                        gan,
                        self.theta0.get(),
                        self.psi0.get(),
                        nsteps=self.n_steps.get(),
                        hs_d=self.h_d.get(),
                        hs_g=self.h_g.get(),
                        gsteps=self.gsteps.get(),
                        dsteps=self.dsteps.get(),
                    )
                )

            theta_mesh, psi_mesh = np.meshgrid(self.theta_s, self.psi_s)
            # directions for the arrows
            v1, v2 = gan(theta_mesh, psi_mesh)
            self.arrows.append((v1, v2))

    def update_plot(self):
        # Generate grid and vector field data
        for i, ax in enumerate(self.axs):
            ax.set_xlim(self.theta_s.min() - 0.25, self.theta_s.max() + 0.25)
            ax.set_ylim(self.psi_s.min() - 0.25, self.psi_s.max() + 0.25)
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            x = self.theta_s
            y = self.psi_s
            X, Y = np.meshgrid(x, y)
            U = self.arrows[i][0]
            V = self.arrows[i][1]

            # Plot the quiver
            ax.quiver(X, Y, U, V, color="#3b4252")
            ax.set_title(self.GANS[i].__class__.__name__)
            ax.set_aspect("equal")

            psis, thetas = self.trajectories[i]
            ax.scatter(
                psis,
                thetas,
                marker="^",
                facecolor="None",
                edgecolor="#5e81ac",
                alpha=0.8,
            )
            ax.scatter(psis[0], thetas[0], color="#bf616a")

            self.canvas.draw()

    def make_initial_plot(self):
        for i, ax in enumerate(self.axs):
            ax.set_xlim(self.theta_s.min() - 0.25, self.theta_s.max() + 0.25)
            ax.set_ylim(self.psi_s.min() - 0.25, self.psi_s.max() + 0.25)
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])
            x = self.theta_s
            y = self.psi_s
            X, Y = np.meshgrid(x, y)
            U = self.arrows[i][0]
            V = self.arrows[i][1]

            # Plot the quiver
            ax.quiver(X, Y, U, V, color="#3b4252")
            ax.set_title(self.GANS[i].__class__.__name__)
            ax.set_aspect("equal")

            psis, thetas = self.trajectories[i]
            ax.scatter(psis[0], thetas[0], color="#bf616a")

            self.canvas.draw()

    def start_animation(self):
        if not self.animating:
            self.animating = True
            self.index = 0

            self.refresh_plot()
            self.make_initial_plot()
            self.animate_step()

    def continue_animation(self):
        if not self.animating:
            self.animating = True
            # self.index = 0

            # self.refresh_plot()
            # self.make_initial_plot()
            self.animate_step()

    def stop_animation(self):
        self.animating = False
        # if self.after_id is not None:
        #    self.root.after_cancel(self.after_id)
        #    self.after_id = None

    def animate_step(self):
        if not self.animating or self.index >= len(self.trajectories[0][0]):
            return
        for i, ax in enumerate(self.axs):
            psis, thetas = self.trajectories[i]
            ax.scatter(
                psis[self.index],
                thetas[self.index],
                marker="^",
                facecolors="none",
                edgecolors="#5e81ac",
                alpha=0.8,
            )

        self.canvas.draw()

        self.index += 1
        # self.after_id = self.root.after(2000, self.animate_step)
        self.root.after(10, self.animate_step)

    def set_gradient_descent(self, descent: str):
        assert descent in ["simultaneous", "alternating"]
        self.grad_descent = descent

    def update_GAN_param(self, gan_name, value):
        assert gan_name in self.GAN_params.keys()
        self.GAN_params[gan_name] = value


if __name__ == "__main__":
    root = tk.Tk()
    app = DiracGANPlot(root)
    root.mainloop()
