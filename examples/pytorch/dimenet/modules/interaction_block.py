import dgl.function as fn
import torch
import torch.nn as nn
from modules.initializers import GlorotOrthogonal
from modules.residual_layer import ResidualLayer


class InteractionBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_radial,
        num_spherical,
        num_bilinear,
        num_before_skip,
        num_after_skip,
        activation=None,
    ):
        super(InteractionBlock, self).__init__()

        self.activation = activation
        # Transformations of Bessel and spherical basis representations
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_sbf = nn.Linear(
            num_radial * num_spherical, num_bilinear, bias=False
        )
        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size)
        self.dense_kj = nn.Linear(emb_size, emb_size)
        # Bilinear layer
        bilin_initializer = torch.empty(
            (emb_size, num_bilinear, emb_size)
        ).normal_(mean=0, std=2 / emb_size)
        self.W_bilin = nn.Parameter(bilin_initializer)
        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList(
            [
                ResidualLayer(emb_size, activation=activation)
                for _ in range(num_before_skip)
            ]
        )
        self.final_before_skip = nn.Linear(emb_size, emb_size)
        # Residual layers after skip connection
        self.layers_after_skip = nn.ModuleList(
            [
                ResidualLayer(emb_size, activation=activation)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_params()

    def reset_params(self):
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense_sbf.weight)
        GlorotOrthogonal(self.dense_ji.weight)
        GlorotOrthogonal(self.dense_kj.weight)
        GlorotOrthogonal(self.final_before_skip.weight)

    def edge_transfer(self, edges):
        # Transform from Bessel basis to dence vector
        rbf = self.dense_rbf(edges.data["rbf"])
        # Initial transformation
        x_ji = self.dense_ji(edges.data["m"])
        x_kj = self.dense_kj(edges.data["m"])
        if self.activation is not None:
            x_ji = self.activation(x_ji)
            x_kj = self.activation(x_kj)

        # w: W * e_RBF \bigodot \sigma(W * m + b)
        return {"x_kj": x_kj * rbf, "x_ji": x_ji}

    def msg_func(self, edges):
        sbf = self.dense_sbf(edges.data["sbf"])
        # Apply bilinear layer to interactions and basis function activation
        # [None, 8] * [128, 8, 128] * [None, 128] -> [None, 128]
        x_kj = torch.einsum(
            "wj,wl,ijl->wi", sbf, edges.src["x_kj"], self.W_bilin
        )
        return {"x_kj": x_kj}

    def forward(self, g, l_g):
        g.apply_edges(self.edge_transfer)

        # nodes correspond to edges and edges correspond to nodes in the original graphs
        # node: d, rbf, o, rbf_env, x_kj, x_ji
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.update_all(self.msg_func, fn.sum("x_kj", "m_update"))

        for k, v in l_g.ndata.items():
            g.edata[k] = v

        # Transformations before skip connection
        g.edata["m_update"] = g.edata["m_update"] + g.edata["x_ji"]
        for layer in self.layers_before_skip:
            g.edata["m_update"] = layer(g.edata["m_update"])
        g.edata["m_update"] = self.final_before_skip(g.edata["m_update"])
        if self.activation is not None:
            g.edata["m_update"] = self.activation(g.edata["m_update"])

        # Skip connection
        g.edata["m"] = g.edata["m"] + g.edata["m_update"]

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            g.edata["m"] = layer(g.edata["m"])

        return g
