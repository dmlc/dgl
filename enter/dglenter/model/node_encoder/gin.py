
import torch.nn as nn
from dgl.nn import GINConv


class GIN(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim=64, num_layers=3, aggregator_type='mean'):
        super().__init__()
        self.conv_list = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                input_dim = in_size
            else:
                input_dim = hidden_dim
            mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

            self.conv_list.append(GINConv(mlp, aggregator_type, 1e-5, True))
        self.out_mlp = nn.Linear(hidden_dim, out_size)

    def forward(self, graph, node_feat, edge_feat=None):
        h = node_feat
        for i in range(self.num_layers):
            h = self.conv_list[i](graph, h)
        h = self.out_mlp(h)
        return h
