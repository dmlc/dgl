import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

from dgl.nn.pytorch import RelGraphConv


class RGCN(nn.Module):
    def __init__(
        self,
        num_nodes,
        h_dim,
        out_dim,
        num_rels,
        regularizer="basis",
        num_bases=-1,
        dropout=0.0,
        self_loop=False,
        ns_mode=False,
    ):
        super(RGCN, self).__init__()

        if num_bases == -1:
            num_bases = num_rels
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.conv1 = RelGraphConv(
            h_dim, h_dim, num_rels, regularizer, num_bases, self_loop=self_loop
        )
        self.conv2 = RelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            regularizer,
            num_bases,
            self_loop=self_loop,
        )
        self.dropout = nn.Dropout(dropout)
        self.ns_mode = ns_mode

    def forward(self, g, nids=None):
        if self.ns_mode:
            # forward for neighbor sampling
            x = self.emb(g[0].srcdata[dgl.NID])
            h = self.conv1(g[0], x, g[0].edata[dgl.ETYPE], g[0].edata["norm"])
            h = self.dropout(F.relu(h))
            h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], g[1].edata["norm"])
            return h
        else:
            x = self.emb.weight if nids is None else self.emb(nids)
            h = self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"])
            h = self.dropout(F.relu(h))
            h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
            return h
