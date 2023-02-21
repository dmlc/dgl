"""
Graph Representation Learning via Hard Attention Networks in DGL using Adam optimization.
References
----------
Paper: https://arxiv.org/abs/1907.04652
"""

from functools import partial

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.sampling import select_topk


class HardGAO(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=8,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=F.elu,
        k=8,
    ):
        super(HardGAO, self).__init__()
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k
        self.residual = residual
        # Initialize Parameters for Additive Attention
        self.fc = nn.Linear(
            self.in_feats, self.out_feats * self.num_heads, bias=False
        )
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, self.num_heads, self.out_feats))
        )
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, self.num_heads, self.out_feats))
        )
        # Initialize Parameters for Hard Projection
        self.p = nn.Parameter(torch.FloatTensor(size=(1, in_feats)))
        # Initialize Dropouts
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if self.residual:
            if self.in_feats == self.out_feats:
                self.residual_module = Identity()
            else:
                self.residual_module = nn.Linear(
                    self.in_feats, self.out_feats * num_heads, bias=False
                )

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.p, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.residual:
            nn.init.xavier_normal_(self.residual_module.weight, gain=gain)

    def forward(self, graph, feat, get_attention=False):
        # Check in degree and generate error
        if (graph.in_degrees() == 0).any():
            raise DGLError(
                "There are 0-in-degree nodes in the graph, "
                "output for those nodes will be invalid. "
                "This is harmful for some applications, "
                "causing silent performance regression. "
                "Adding self-loop on the input graph by "
                "calling `g = dgl.add_self_loop(g)` will resolve "
                "the issue. Setting ``allow_zero_in_degree`` "
                "to be `True` when constructing this module will "
                "suppress the check and let the code run."
            )
        # projection process to get importance vector y
        graph.ndata["y"] = torch.abs(
            torch.matmul(self.p, feat.T).view(-1)
        ) / torch.norm(self.p, p=2)
        # Use edge message passing function to get the weight from src node
        graph.apply_edges(fn.copy_u("y", "y"))
        # Select Top k neighbors
        subgraph = select_topk(graph.cpu(), self.k, "y").to(graph.device)
        # Sigmoid as information threshold
        subgraph.ndata["y"] = torch.sigmoid(subgraph.ndata["y"])
        # Using vector matrix elementwise mul for acceleration
        feat = subgraph.ndata["y"].view(-1, 1) * feat
        feat = self.feat_drop(feat)
        h = self.fc(feat).view(-1, self.num_heads, self.out_feats)
        el = (h * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (h * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # Assign the value on the subgraph
        subgraph.srcdata.update({"ft": h, "el": el})
        subgraph.dstdata.update({"er": er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        subgraph.apply_edges(fn.u_add_v("el", "er", "e"))
        e = self.leaky_relu(subgraph.edata.pop("e"))
        # compute softmax
        subgraph.edata["a"] = self.attn_drop(edge_softmax(subgraph, e))
        # message passing
        subgraph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
        rst = subgraph.dstdata["ft"]
        # activation
        if self.activation:
            rst = self.activation(rst)
        # Residual
        if self.residual:
            rst = rst + self.residual_module(feat).view(
                feat.shape[0], -1, self.out_feats
            )

        if get_attention:
            return rst, subgraph.edata["a"]
        else:
            return rst


class HardGAT(nn.Module):
    def __init__(
        self,
        g,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        k,
    ):
        super(HardGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        gat_layer = partial(HardGAO, k=k)
        muls = heads
        # input projection (no residual)
        self.gat_layers.append(
            gat_layer(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                gat_layer(
                    num_hidden * muls[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            gat_layer(
                num_hidden * muls[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                None,
            )
        )

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
