"""GAN variants added on top of the original DiracGAN code."""

from diracgan.gans import VectorField, fp


class LSGAN(VectorField):
    def _get_vector(self, theta, psi):
        v1 = psi * theta**2 - theta
        v2 = -(psi**2) * theta
        return v1, v2


class LeCamGAN(VectorField):
    """Standard GAN loss plus the LeCam regularizer (Tseng et al., 2021)."""

    def __init__(self, reg=-0.3, anchor_real=0.3):
        super().__init__()
        self.reg = reg
        self.anchor_real = anchor_real

    def _get_vector(self, theta, psi):
        # reg term: d/dtheta of (psi * theta - anchor)^2
        lecam = 2 * psi**2 * theta - 2 * psi * self.anchor_real
        v1 = -psi * fp(psi * theta) + self.reg * lecam
        v2 = theta * fp(psi * theta)
        return v1, v2
