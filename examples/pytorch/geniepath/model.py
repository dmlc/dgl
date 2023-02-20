import torch as th
import torch.nn as nn

from dgl.nn import GATConv
from torch.nn import LSTM


class GeniePathConv(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads=1, residual=False):
        super(GeniePathConv, self).__init__()
        self.breadth_func = GATConv(
            in_dim, hid_dim, num_heads=num_heads, residual=residual
        )
        self.depth_func = LSTM(hid_dim, out_dim)

    def forward(self, graph, x, h, c):
        x = self.breadth_func(graph, x)
        x = th.tanh(x)
        x = th.mean(x, dim=1)
        x, (h, c) = self.depth_func(x.unsqueeze(0), (h, c))
        x = x[0]
        return x, (h, c)


class GeniePath(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hid_dim=16,
        num_layers=2,
        num_heads=1,
        residual=False,
    ):
        super(GeniePath, self).__init__()
        self.hid_dim = hid_dim
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GeniePathConv(
                    hid_dim,
                    hid_dim,
                    hid_dim,
                    num_heads=num_heads,
                    residual=residual,
                )
            )

    def forward(self, graph, x):
        h = th.zeros(1, x.shape[0], self.hid_dim).to(x.device)
        c = th.zeros(1, x.shape[0], self.hid_dim).to(x.device)

        x = self.linear1(x)
        for layer in self.layers:
            x, (h, c) = layer(graph, x, h, c)
        x = self.linear2(x)

        return x


class GeniePathLazy(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hid_dim=16,
        num_layers=2,
        num_heads=1,
        residual=False,
    ):
        super(GeniePathLazy, self).__init__()
        self.hid_dim = hid_dim
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = th.nn.Linear(hid_dim, out_dim)
        self.breaths = nn.ModuleList()
        self.depths = nn.ModuleList()
        for i in range(num_layers):
            self.breaths.append(
                GATConv(
                    hid_dim, hid_dim, num_heads=num_heads, residual=residual
                )
            )
            self.depths.append(LSTM(hid_dim * 2, hid_dim))

    def forward(self, graph, x):
        h = th.zeros(1, x.shape[0], self.hid_dim).to(x.device)
        c = th.zeros(1, x.shape[0], self.hid_dim).to(x.device)

        x = self.linear1(x)
        h_tmps = []
        for layer in self.breaths:
            h_tmps.append(th.mean(th.tanh(layer(graph, x)), dim=1))
        x = x.unsqueeze(0)
        for h_tmp, layer in zip(h_tmps, self.depths):
            in_cat = th.cat((h_tmp.unsqueeze(0), x), -1)
            x, (h, c) = layer(in_cat, (h, c))
        x = self.linear2(x[0])

        return x
