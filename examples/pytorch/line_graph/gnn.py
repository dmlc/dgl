"""
Supervised Community Detection with Hierarchical Graph Neural Networks
https://arxiv.org/abs/1705.08415

Deviations from paper:
- Pm Pd
"""


import copy
import itertools
import dgl
import dgl.function as fn
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, radius):
        super().__init__()
        self.out_feats = out_feats
        self.radius = radius

        new_linear = lambda: nn.Linear(in_feats, out_feats * 2)
        new_module_list = lambda: nn.ModuleList([new_linear() for i in range(radius)])

        self.theta_x, self.theta_deg, self.theta_y = \
            new_linear(), new_linear(), new_linear()
        self.theta_list = new_module_list()

        self.gamma_y, self.gamma_deg, self.gamma_x = \
            new_linear(), new_linear(), new_linear()
        self.gamma_list = new_module_list()

        self.bn_x = nn.BatchNorm1d(out_feats)
        self.bn_y = nn.BatchNorm1d(out_feats)

    def aggregate(self, g, z):
        z_list = []
        g.set_n_repr(z)
        g.update_all(fn.copy_src(), fn.sum(), batchable=True)
        z_list.append(g.get_n_repr())
        for i in range(self.radius - 1):
            for j in range(2 ** i):
                g.update_all(fn.copy_src(), fn.sum(), batchable=True)
            z_list.append(g.get_n_repr())
        return z_list

    def forward(self, g, lg, x, y, deg_g, deg_lg, eid2nid):
        xy = F.embedding(eid2nid, x)

        x_list = [theta(z) for theta, z in zip(self.theta_list, self.aggregate(g, x))]

        g.set_e_repr(y)
        g.update_all(fn.copy_edge(), fn.sum(), batchable=True)
        yx = g.get_n_repr()

        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum(x_list) + self.theta_y(yx)
        x = self.bn_x(x[:, :self.out_feats] + F.relu(x[:, self.out_feats:]))

        y_list = [gamma(z) for gamma, z in zip(self.gamma_list, self.aggregate(lg, y))]
        lg.set_n_repr(xy)
        lg.update_all(fn.copy_src(), fn.sum(), batchable=True)
        xy = lg.get_n_repr()
        y = self.gamma_y(y) + self.gamma_deg(deg_lg * y) + sum(y_list) + self.gamma_x(xy)
        y = self.bn_y(y[:, :self.out_feats] + F.relu(y[:, self.out_feats:]))

        return x, y


class GNN(nn.Module):
    def __init__(self, feats, radius, n_classes):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        super(GNN, self).__init__()
        self.linear = nn.Linear(feats[-1], n_classes)
        self.module_list = nn.ModuleList([GNNModule(m, n, radius)
                                          for m, n in zip(feats[:-1], feats[1:])])

    def forward(self, g, lg, deg_g, deg_lg, eid2nid):
        def normalize(x):
            x = x - th.mean(x, 0)
            x = x / th.sqrt(th.mean(x * x, 0))
            return x

        x = normalize(deg_g)
        y = normalize(deg_lg)
        for module in self.module_list:
            x, y = module(g, lg, x, y, deg_g, deg_lg, eid2nid)
        return self.linear(x)
