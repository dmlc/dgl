import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, order=1,
                 activation=None, dropout=0, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.order = order
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.batch_norm:
            self.bn.reset_parameters()

    def compute_degree(self, g):
        if 'D_in' in g.ndata:
            D_in = g.ndata['D_in']
        else:
            D_in = 1. / g.in_degrees().float().clamp(min=1).unsqueeze(1)
        return D_in

    def forward(self, graph, features):
        g = graph.local_var()
        h = self.dropout(features)

        if self.order == 1:
            D_in = self.compute_degree(g)
            g.ndata['h'] = h
            if 'w' not in g.edata:
                g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
            g.update_all(fn.u_mul_e('h', 'w', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            h = h * D_in

        h = self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.batch_norm:
            h = self.bn(h)
        return h


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, arch="1-1-0", activation=F.relu, dropout=0, batch_norm=False):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()

        orders = list(map(int, arch.split('-')))
        for i in range(len(orders)):
            if i == 0:
                self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, order=orders[i],
                                         activation=activation, dropout=dropout, batch_norm=batch_norm))
            elif i == len(orders) - 1:
                self.gcn.append(GCNLayer(in_dim=hid_dim, out_dim=out_dim, order=orders[i],
                                         activation=None, dropout=dropout, batch_norm=batch_norm))
            else:
                self.gcn.append(GCNLayer(in_dim=hid_dim, out_dim=hid_dim, order=orders[i],
                                         activation=activation, dropout=dropout, batch_norm=batch_norm))

    def forward(self, graph):
        h = graph.ndata['feat']
        for layer in self.gcn:
            h = layer(graph, h)
        return h

