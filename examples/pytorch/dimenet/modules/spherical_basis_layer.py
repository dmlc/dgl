import sympy as sym
import torch
import torch.nn as nn
from modules.basis_utils import bessel_basis, real_sph_harm
from modules.envelope import Envelope


class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(
            num_spherical, num_radial
        )  # x, [num_spherical, num_radial] sympy functions
        self.sph_harm_formulas = real_sph_harm(
            num_spherical
        )  # theta, [num_spherical, ] sympy functions
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to torch functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify(
                    [theta], self.sph_harm_formulas[i][0], modules
                )(0)
                self.sph_funcs.append(
                    lambda tensor: torch.zeros_like(tensor) + first_sph
                )
            else:
                self.sph_funcs.append(
                    sym.lambdify([theta], self.sph_harm_formulas[i][0], modules)
                )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], self.bessel_formulas[i][j], modules)
                )

    def get_bessel_funcs(self):
        return self.bessel_funcs

    def get_sph_funcs(self):
        return self.sph_funcs
