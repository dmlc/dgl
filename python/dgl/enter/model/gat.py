

import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn import GATConv


class GAT(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 num_layers : int = 2,
                 num_hidden: int = 8,
                 heads = [8, 8],
                 activation = "elu",
                 feat_drop : float = 0.6,
                 attn_drop: float = 0.6,
                 negative_slope: float = 0.2,
                 residual: bool = False):
        super(GAT, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_layers = num_layers-1
        self.gat_layers = nn.ModuleList()
        self.activation = getattr(torch.nn.functional, activation)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_size, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, self.num_layers):
            # due to multi-head, the in_size = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], out_size, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, graph, node_feat, edge_feat = None):
        h = node_feat
        for l in range(self.num_layers):
            h = self.gat_layers[l](graph, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](graph, h).mean(1)
        return logits

    def forward_block(self,  blocks, node_feat, edge_feat = None):
        h = node_feat
        for l in range(self.num_layers):
            h = self.gat_layers[l](blocks[l], h).flatten(1)        
        logits = self.gat_layers[-1](blocks[-1], h).mean(1)
        return logits