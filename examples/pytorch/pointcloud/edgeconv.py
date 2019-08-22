import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class EdgeConv(nn.Module):
    def __init__(self, in_features, out_features):
        self.theta = nn.Linear(in_features, out_features)
        self.phi = nn.Linear(in_features, out_features)

    def apply_edges(self, edges):
        theta_x = self.theta @ (edges.dst['x'] - edges.src['x'])
        phi_x = self.phi @ edges.src['x']
        return {'e': F.relu(theta_x + phi_x)}

    def forward(self, g):
        g.update_all(self.apply_edges, fn.max('e', 'x'))
        return g.ndata['x']
