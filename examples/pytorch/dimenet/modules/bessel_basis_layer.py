import numpy as np
import torch
import torch.nn as nn
from modules.envelope import Envelope


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()

        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        self.reset_params()

    def reset_params(self):
        with torch.no_grad():
            torch.arange(
                1, self.frequencies.numel() + 1, out=self.frequencies
            ).mul_(np.pi)
        self.frequencies.requires_grad_()

    def forward(self, g):
        d_scaled = g.edata["d"] / self.cutoff
        # Necessary for proper broadcasting behaviour
        d_scaled = torch.unsqueeze(d_scaled, -1)
        d_cutoff = self.envelope(d_scaled)
        g.edata["rbf"] = d_cutoff * torch.sin(self.frequencies * d_scaled)
        return g
