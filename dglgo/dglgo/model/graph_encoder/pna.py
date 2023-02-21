from typing import List

import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, SumPooling
from ogb.graphproppred.mol_encoder import AtomEncoder


def aggregate_mean(h):
    """mean aggregation"""
    return torch.mean(h, dim=1)


def aggregate_max(h):
    """max aggregation"""
    return torch.max(h, dim=1)[0]


def aggregate_min(h):
    """min aggregation"""
    return torch.min(h, dim=1)[0]


def aggregate_sum(h):
    """sum aggregation"""
    return torch.sum(h, dim=1)


def aggregate_var(h):
    """variance aggregation"""
    h_mean_squares = torch.mean(h * h, dim=1)
    h_mean = torch.mean(h, dim=1)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_std(h):
    """standard deviation aggregation"""
    return torch.sqrt(aggregate_var(h) + 1e-5)


AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
}


def scale_identity(h, D, delta):
    """identity scaling (no scaling operation)"""
    return h


def scale_amplification(h, D, delta):
    """amplification scaling"""
    return h * (np.log(D + 1) / delta)


def scale_attenuation(h, D, delta):
    """attenuation scaling"""
    return h * (delta / np.log(D + 1))


SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class MLP(nn.Module):
    def __init__(
        self,
        in_feat_size: int,
        out_feat_size: int,
        num_layers: int = 3,
        decreasing_hidden_size=False,
    ):
        """Multilayer Perceptron (MLP)"""
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        if decreasing_hidden_size:
            for i in range(num_layers - 1):
                self.layers.append(
                    nn.Linear(
                        in_feat_size // 2**i, in_feat_size // 2 ** (i + 1)
                    )
                )
            self.layers.append(
                nn.Linear(in_feat_size // 2 ** (num_layers - 1), out_feat_size)
            )
        else:
            self.layers.append(nn.Linear(in_feat_size, out_feat_size))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(out_feat_size, out_feat_size))
        self.num_layers = num_layers

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != self.num_layers - 1:
                h = F.relu(h)
        return h


class SimplePNAConv(nn.Module):
    r"""A simplified PNAConv variant used in OGB submissions"""

    def __init__(
        self,
        feat_size: int,
        aggregators: List[str],
        scalers: List[str],
        delta: float,
        dropout: float,
        batch_norm: bool,
        residual: bool,
        num_mlp_layers: int,
    ):
        super(SimplePNAConv, self).__init__()

        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.delta = delta
        self.mlp = MLP(
            in_feat_size=(len(aggregators) * len(scalers)) * feat_size,
            out_feat_size=feat_size,
            num_layers=num_mlp_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

        if batch_norm:
            self.bn = nn.BatchNorm1d(feat_size)
        else:
            self.bn = None

    def reduce(self, nodes):
        h = nodes.mailbox["m"]
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat(
            [scale(h, D=D, delta=self.delta) for scale in self.scalers], dim=1
        )
        return {"h": h}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.update_all(fn.copy_u("h", "m"), self.reduce)
            h_new = g.ndata["h"]
        h_new = self.mlp(h_new)

        if self.bn is not None:
            h_new = self.bn(h_new)
        h_new = F.relu(h_new)

        if self.residual:
            h_new = h_new + h
        h_new = self.dropout(h_new)

        return h_new


class PNA(nn.Module):
    def __init__(
        self,
        data_info: dict,
        embed_size: int = 80,
        aggregators: str = "mean max min std",
        scalers: str = "identity amplification attenuation",
        dropout: float = 0.3,
        batch_norm: bool = True,
        residual: bool = True,
        num_mlp_layers: int = 1,
        num_layers: int = 4,
        readout: str = "mean",
    ):
        """Principal Neighbourhood Aggregation

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        embed_size : int
            Embedding size.
        aggregators : str
            Aggregation function names separated by space, can include mean, max, min, std, sum
        scalers : str
            Scaler function names separated by space, can include identity, amplification, and attenuation
        dropout : float
            Dropout rate.
        batch_norm : bool
            Whether to use batch normalization.
        residual : bool
            Whether to use residual connection.
        num_mlp_layers : int
            Number of MLP layers to use after message aggregation in each PNA layer.
        num_layers : int
            Number of PNA layers.
        readout : str
            Readout for computing graph-level representations, can be 'sum' or 'mean'.
        """
        super(PNA, self).__init__()
        self.data_info = data_info
        self.embed_size = embed_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.num_mlp_layers = num_mlp_layers
        self.num_layers = num_layers
        self.readout = readout

        if aggregators is None:
            aggregators = ["mean", "max", "min", "std"]
        else:
            aggregators = [agg.strip() for agg in aggregators.split(" ")]
            assert set(aggregators).issubset(
                {"mean", "max", "min", "std", "sum"}
            ), "Expect aggregators to be a subset of ['mean', 'max', 'min', 'std', 'sum'], \
                    got {}".format(
                aggregators
            )
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]
        else:
            scalers = [scl.strip() for scl in scalers.split(" ")]
            assert set(scalers).issubset(
                {"identity", "amplification", "attenuation"}
            ), "Expect scalers to be a subset of ['identity', 'amplification', 'attenuation'], \
                    got {}".format(
                scalers
            )
        self.aggregators = aggregators
        self.scalers = scalers

        if data_info["name"] in ["ogbg-molhiv", "ogbg-molpcba"]:
            self.node_encoder = AtomEncoder(embed_size)
        else:
            # Handle other datasets
            self.node_encoder = nn.Linear(
                data_info["node_feat_size"], embed_size
            )
        self.conv_layers = nn.ModuleList(
            [
                SimplePNAConv(
                    feat_size=embed_size,
                    aggregators=aggregators,
                    scalers=scalers,
                    delta=data_info["delta"],
                    dropout=dropout,
                    batch_norm=batch_norm,
                    residual=residual,
                    num_mlp_layers=num_mlp_layers,
                )
                for _ in range(num_layers)
            ]
        )

        if readout == "sum":
            self.pool = SumPooling()
        elif readout == "mean":
            self.pool = AvgPooling()
        else:
            raise ValueError(
                "Expect readout to be 'sum' or 'mean', got {}".format(readout)
            )
        self.pred = MLP(
            embed_size, data_info["out_size"], decreasing_hidden_size=True
        )

    def forward(self, graph, node_feat, edge_feat=None):
        hn = self.node_encoder(node_feat)
        for conv in self.conv_layers:
            hn = conv(graph, hn)
        hg = self.pool(graph, hn)

        return self.pred(hg)
