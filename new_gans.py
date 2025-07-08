from diracgan.gans import VectorField, fp, fp2
import numpy as np

def f(x):
    return -np.log(1 + np.exp(-x))

class LeCamGAN(VectorField):
    def __init__(self, reg=-0.3, anchor_real=0.3):
        super().__init__()
        self.reg = reg
        self.anchor_real = anchor_real

    def _get_vector(self, theta, psi):
        v1 = (-psi * fp(psi*theta)) + self.reg * ((2 * (psi ** 2) * theta -2 * psi * self.anchor_real))
        
        v2 = theta * fp(psi * theta)
        return v1, v2


class LSGAN(VectorField):
    def _get_vector(self, theta, psi):
        v1 = ((psi * (theta**2)) - theta)
        v2 = -1 * (psi**2) * theta
        return v1, v2




class LeCam_GAN(VectorField):
    def __init__(self):
        super().__init__()


    def _get_vector(self, theta, psi):
        v1 = -psi * fp(psi*theta)
        v2 = theta * fp(psi*theta)
        return v1, v2





