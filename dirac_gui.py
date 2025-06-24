import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from diracgan.simulate import trajectory_simgd, trajectory_altgd
from diracgan.gans import GAN, NSGAN, WGAN, WGAN_GP, GAN_InstNoise, GAN_GradPenalty, GAN_Consensus, NSGAN_GradPenalty

#ablauf:
"""
-> Trajectories und vektoren für alle GANs berechnen
-> Ersten Punkt und Vektoren plotten und warten
"""


class DiracGANPlot:
    def __init__(self, root):
        self.running = False
        #initial plot configs
        #gan configs


        self.GAN_params = {
            "WGAN_clip": 1., 
            "WGAN_GP_reg": 0.7,
            "WGAN_GP_target": 1.,
            "GAN_InstNoise_std": 0.7,
            "GAN_GradPenalty_reg": 0.3,
            "GAN_Consensus_reg": 1.,
            "NSGAN_GradPenalty_reg": 0.3
        }
        self.GANS = [GAN(),
                     NSGAN(),
                     WGAN(self.GAN_params["WGAN_clip"]),
                     WGAN_GP(self.GAN_params["WGAN_GP_reg"], self.GAN_params["WGAN_GP_target"]),
                     GAN_InstNoise(self.GAN_params["GAN_InstNoise_std"]),
                     GAN_GradPenalty(self.GAN_params["GAN_GradPenalty_reg"]),
                     GAN_Consensus(self.GAN_params["GAN_Consensus_reg"]),
                     NSGAN_GradPenalty(self.GAN_params["NSGAN_GradPenalty_reg"])
        ]

        #learning rate
        self.h_d = 0.2
        self.h_g = 0.2
        #starting points
        self.theta0 = 1.
        self.psi0 = 1.
        #iterations
        self.n_steps = 500
        #steps per update
        self.gsteps = 1
        self.dsteps = 1

        self.theta_s = np.linspace(-2, 2., 10)
        self.psi_s = np.linspace(-2, 2, 10)

        #what kind of gradient descent (simultaneous or alternating)
        self.grad_descent = "simultaneous"

        self.root = root
        self.root.title("Interactive Plot")

        # Create a frame for the plot
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a matplotlib figure
        n_cols = len(self.GANS) // 2
        self.fig, self.axs = plt.subplots(2, n_cols, figsize=(4*n_cols, 6), sharex=True, sharey=True, layout="constrained")
        self.axs = self.axs.flatten()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button to update the plot 
        self.apply_button = ttk.Button(root, text="Apply", command=...)
        self.apply_button.pack(side=tk.BOTTOM, pady=10)       
        self.toggle_button = ttk.Button(root, text="Start Trajectory", command=self.toggle_animation, state="disabled")
        self.toggle_button.pack(side=tk.BOTTOM, pady=10)
        self.next_step_button = ttk.Button(root, text="Next Step", command=..., state="disabled")
        self.next_step_button.pack(side=tk.BOTTOM, pady=10)
        self.refresh_button = ttk.Button(root, text="Previous Step", command=..., state="disabled")
        self.refresh_button.pack(side=tk.BOTTOM, pady=10)

        self.start_theta_slider = ttk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, command=self.set_start_theta)
        self.start_theta_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.start_psi_slider = ttk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, command=self.set_start_psi)
        self.start_psi_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.num_steps_slider = ttk.Scale(root, from_=100, to=1000, orient=tk.HORIZONTAL, command=self.set_num_steps)
        self.num_steps_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.h_g_slider = ttk.Scale(root, from_=0.01, to=1, orient=tk.HORIZONTAL, command=self.set_generator_lr)
        self.h_g_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.h_d_slider = ttk.Scale(root, from_=0.01, to=1, orient=tk.HORIZONTAL, command=self.set_discriminator_lr)
        self.h_d_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.dsteps_slider = ttk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, command=self.set_discriminator_updates_per_generator_update)
        self.dsteps_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.gsteps_slider = ttk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, command=self.set_generator_updates_per_discriminator_update)
        self.gsteps_slider.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        # self.grad_descent_combobox = ttk.Combobox(root, values=["simultaneous", "alternating"], state="readonly", command=self.set_gradient_descent)
        # self.grad_descent_combobox.set("simultaneous")

        # Initial plot
        self.refresh_plot()
        self.update_plot()

    def toggle_animation(self):
        if not self.running:
            self.running = True
            self.toggle_button.config(text="Stop Trajectory")
            self.init_plot_values()
            self.animate()
        else:
            self.running = False
            self.toggle_button.config(text="Start Trajectory")
        
    def refresh_plot(self):
        self.GANS = [GAN(),
                     NSGAN(),
                     WGAN(self.GAN_params["WGAN_clip"]),
                     WGAN_GP(self.GAN_params["WGAN_GP_reg"], self.GAN_params["WGAN_GP_target"]),
                     GAN_InstNoise(self.GAN_params["GAN_InstNoise_std"]),
                     GAN_GradPenalty(self.GAN_params["GAN_GradPenalty_reg"]),
                     GAN_Consensus(self.GAN_params["GAN_Consensus_reg"]),
                     NSGAN_GradPenalty(self.GAN_params["NSGAN_GradPenalty_reg"])
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
                        self.theta0, 
                        self.psi0, 
                        nsteps=self.n_steps, 
                        hs_d=self.h_d,
                        hs_g=self.h_g
                    )
                )
            else: #alternating mode
                self.trajectories.append(
                    trajectory_altgd(
                        gan, 
                        self.theta0, 
                        self.psi0, 
                        nsteps=self.n_steps, 
                        hs_d=self.h_d, 
                        hs_g=self.h_g, 
                        gsteps=self.gsteps, 
                        dsteps=self.dsteps
                    )
                )
                
                theta_mesh, psi_mesh = np.meshgrid(self.theta_s, self.psi_s)
                #directions for the arrows
                v1, v2 = gan(theta_mesh, psi_mesh)
                self.arrows.append((v1, v2))
        
    def update_plot(self):
        # Generate grid and vector field data
        for ax in self.axs:
            x = self.theta_s
            y = self.psi_s
            X, Y = np.meshgrid(x, y)
            U = -Y + np.random.randn(*X.shape) * 0.2
            V = X + np.random.randn(*Y.shape) * 0.2

            # Plot the quiver
            ax.quiver(X, Y, U, V, color='black')
            ax.set_title("Quiver Plot")
            ax.set_aspect('equal')
            self.canvas.draw()

    #button functionalities
    def set_start_theta(self, theta):
        self.theta0 = theta
        
    def set_start_psi(self, psi):
        self.psi0 = psi

    def set_num_steps(self, steps):
        self.n_steps = steps

    def set_generator_lr(self, lr):
        self.h_g = lr

    def set_discriminator_lr(self, lr):
        self.h_d = lr
        
    def set_discriminator_updates_per_generator_update(self, nd):
        self.dsteps = nd

    def set_generator_updates_per_discriminator_update(self, ng):
        self.gsteps = ng

    def set_gradient_descent(self, descent:str):
        assert descent in ["simultaneous", "alternating"]
        self.grad_descent = descent
    
    def update_GAN_param(self, gan_name, value):
        assert gan_name in self.GAN_params.keys()
        self.GAN_params[gan_name] = value
    

if __name__ == "__main__":
    root = tk.Tk()
    app = DiracGANPlot(root)
    root.mainloop()