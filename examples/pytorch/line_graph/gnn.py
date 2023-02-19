import copy
import itertools

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, radius):
        super().__init__()
        self.out_feats = out_feats
        self.radius = radius

        new_linear = lambda: nn.Linear(in_feats, out_feats)
        new_linear_list = lambda: nn.ModuleList(
            [new_linear() for i in range(radius)]
        )

        self.theta_x, self.theta_deg, self.theta_y = (
            new_linear(),
            new_linear(),
            new_linear(),
        )
        self.theta_list = new_linear_list()

        self.gamma_y, self.gamma_deg, self.gamma_x = (
            new_linear(),
            new_linear(),
            new_linear(),
        )
        self.gamma_list = new_linear_list()

        self.bn_x = nn.BatchNorm1d(out_feats)
        self.bn_y = nn.BatchNorm1d(out_feats)

    def aggregate(self, g, z):
        z_list = []
        g.ndata["z"] = z
        g.update_all(fn.copy_u(u="z", out="m"), fn.sum(msg="m", out="z"))
        z_list.append(g.ndata["z"])
        for i in range(self.radius - 1):
            for j in range(2**i):
                g.update_all(
                    fn.copy_u(u="z", out="m"), fn.sum(msg="m", out="z")
                )
            z_list.append(g.ndata["z"])
        return z_list

    def forward(self, g, lg, x, y, deg_g, deg_lg, pm_pd):
        pmpd_x = F.embedding(pm_pd, x)

        sum_x = sum(
            theta(z) for theta, z in zip(self.theta_list, self.aggregate(g, x))
        )

        g.edata["y"] = y
        g.update_all(fn.copy_e(e="y", out="m"), fn.sum("m", "pmpd_y"))
        pmpd_y = g.ndata.pop("pmpd_y")

        x = (
            self.theta_x(x)
            + self.theta_deg(deg_g * x)
            + sum_x
            + self.theta_y(pmpd_y)
        )
        n = self.out_feats // 2
        x = th.cat([x[:, :n], F.relu(x[:, n:])], 1)
        x = self.bn_x(x)

        sum_y = sum(
            gamma(z) for gamma, z in zip(self.gamma_list, self.aggregate(lg, y))
        )

        y = (
            self.gamma_y(y)
            + self.gamma_deg(deg_lg * y)
            + sum_y
            + self.gamma_x(pmpd_x)
        )
        y = th.cat([y[:, :n], F.relu(y[:, n:])], 1)
        y = self.bn_y(y)

        return x, y


class GNN(nn.Module):
    def __init__(self, feats, radius, n_classes):
        super(GNN, self).__init__()
        self.linear = nn.Linear(feats[-1], n_classes)
        self.module_list = nn.ModuleList(
            [GNNModule(m, n, radius) for m, n in zip(feats[:-1], feats[1:])]
        )

    def forward(self, g, lg, deg_g, deg_lg, pm_pd):
        x, y = deg_g, deg_lg
        for module in self.module_list:
            x, y = module(g, lg, x, y, deg_g, deg_lg, pm_pd)
        return self.linear(x)
