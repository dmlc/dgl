import dgl
import dgl.function as fn
import torch.nn as nn
from modules.initializers import GlorotOrthogonal
from modules.residual_layer import ResidualLayer


class InteractionPPBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        int_emb_size,
        basis_emb_size,
        num_radial,
        num_spherical,
        num_before_skip,
        num_after_skip,
        activation=None,
    ):
        super(InteractionPPBlock, self).__init__()

        self.activation = activation
        # Transformations of Bessel and spherical basis representations
        self.dense_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.dense_rbf2 = nn.Linear(basis_emb_size, emb_size, bias=False)
        self.dense_sbf1 = nn.Linear(
            num_radial * num_spherical, basis_emb_size, bias=False
        )
        self.dense_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size)
        self.dense_kj = nn.Linear(emb_size, emb_size)
        # Embedding projections for interaction triplets
        self.down_projection = nn.Linear(emb_size, int_emb_size, bias=False)
        self.up_projection = nn.Linear(int_emb_size, emb_size, bias=False)
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
        GlorotOrthogonal(self.dense_rbf1.weight)
        GlorotOrthogonal(self.dense_rbf2.weight)
        GlorotOrthogonal(self.dense_sbf1.weight)
        GlorotOrthogonal(self.dense_sbf2.weight)
        GlorotOrthogonal(self.dense_ji.weight)
        nn.init.zeros_(self.dense_ji.bias)
        GlorotOrthogonal(self.dense_kj.weight)
        nn.init.zeros_(self.dense_kj.bias)
        GlorotOrthogonal(self.down_projection.weight)
        GlorotOrthogonal(self.up_projection.weight)

    def edge_transfer(self, edges):
        # Transform from Bessel basis to dense vector
        rbf = self.dense_rbf1(edges.data["rbf"])
        rbf = self.dense_rbf2(rbf)
        # Initial transformation
        x_ji = self.dense_ji(edges.data["m"])
        x_kj = self.dense_kj(edges.data["m"])
        if self.activation is not None:
            x_ji = self.activation(x_ji)
            x_kj = self.activation(x_kj)

        x_kj = self.down_projection(x_kj * rbf)
        if self.activation is not None:
            x_kj = self.activation(x_kj)
        return {"x_kj": x_kj, "x_ji": x_ji}

    def msg_func(self, edges):
        sbf = self.dense_sbf1(edges.data["sbf"])
        sbf = self.dense_sbf2(sbf)
        x_kj = edges.src["x_kj"] * sbf
        return {"x_kj": x_kj}

    def forward(self, g, l_g):
        g.apply_edges(self.edge_transfer)

        # nodes correspond to edges and edges correspond to nodes in the original graphs
        # node: d, rbf, o, rbf_env, x_kj, x_ji
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g_reverse = dgl.reverse(l_g, copy_edata=True)
        l_g_reverse.update_all(self.msg_func, fn.sum("x_kj", "m_update"))

        g.edata["m_update"] = self.up_projection(l_g_reverse.ndata["m_update"])
        if self.activation is not None:
            g.edata["m_update"] = self.activation(g.edata["m_update"])
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
