import numpy as np
import torch
import torch.nn as nn
from modules.envelope import Envelope
from modules.initializers import GlorotOrthogonal


class EmbeddingBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_radial,
        bessel_funcs,
        cutoff,
        envelope_exponent,
        num_atom_types=95,
        activation=None,
    ):
        super(EmbeddingBlock, self).__init__()

        self.bessel_funcs = bessel_funcs
        self.cutoff = cutoff
        self.activation = activation
        self.envelope = Envelope(envelope_exponent)
        self.embedding = nn.Embedding(num_atom_types, emb_size)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        self.dense = nn.Linear(emb_size * 3, emb_size)
        self.reset_params()

    def reset_params(self):
        nn.init.uniform_(self.embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense.weight)

    def edge_init(self, edges):
        """msg emb init"""
        # m init
        rbf = self.dense_rbf(edges.data["rbf"])
        if self.activation is not None:
            rbf = self.activation(rbf)

        m = torch.cat([edges.src["h"], edges.dst["h"], rbf], dim=-1)
        m = self.dense(m)
        if self.activation is not None:
            m = self.activation(m)

        # rbf_env init
        d_scaled = edges.data["d"] / self.cutoff
        rbf_env = [f(d_scaled) for f in self.bessel_funcs]
        rbf_env = torch.stack(rbf_env, dim=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf_env

        return {"m": m, "rbf_env": rbf_env}

    def forward(self, g):
        g.ndata["h"] = self.embedding(g.ndata["Z"])
        g.apply_edges(self.edge_init)
        return g
