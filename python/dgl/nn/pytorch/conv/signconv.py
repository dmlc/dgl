"""Torch Module for Signed Graph Convolutional layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F

from .... import function as fn


class SIGNConv(nn.Module):
    def __init__(self, in_channels,
                 out_channels, first_aggr,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(SIGNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr
        self.norm = norm
        self.norm_embed = norm_embed
        self.linear_layer = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, graph, feature):
        if self.first_aggr:
            self.forward_base(graph, feature)
        else:
            self.forward_deep(graph, feature)

    def forward_base(self, graph, feature):
        with graph.local_scope():
            graph.ndata["feat"] = feature
            if self.norm:
                graph.update_all(fn.copy_u("feat", "m"), fn.mean("m", "out"))
            else:
                graph.update_all(fn.copy_u("feat", "m"), fn.sum("m", "out"))

            out = th.cat([graph.ndata["out"], feature], 1)
            out = self.linear_layer(out)

            if self.norm_embed:
                out = F.normalize(out, p=2, dim=-1)

        return out

    def forward_deep(self, graphs, features):
        pos_graph, neg_graph = graphs
        pos_feat, neg_feat = features

        with pos_graph.local_scope():
            pos_graph.ndata["feat"] = pos_feat
            if self.norm:
                pos_graph.update_all(
                    fn.copy_u("feat", "m"), fn.mean("m", "out"))
            else:
                pos_graph.update_all(
                    fn.copy_u("feat", "m"), fn.sum("m", "out"))

            pos_out = pos_graph.ndata["out"]

        with neg_graph.local_scope():
            neg_graph.ndata["feat"] = neg_feat
            if self.norm:
                neg_graph.update_all(
                    fn.copy_u("feat", "m"), fn.mean("m", "out"))
            else:
                neg_graph.update_all(
                    fn.copy_u("feat", "m"), fn.sum("m", "out"))

            neg_out = neg_graph.ndata["out"]

        out = th.cat([pos_out, neg_out, pos_feat], 1)
        out = self.linear_layer(out)

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out
