import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True,
                 activation=None, dropout=0):
        super(GCNLayer, self).__init__()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, graph, features):
        g = graph.local_var()

        # I saved the training graph's degree norm during sampling into ndata["D_in"] and ["D_out"]
        if "D_in" in g.ndata:
            D_in = g.ndata["D_in"]
        else:
            D_in = 1. / g.in_degrees().float().sqrt().unsqueeze(1)

        if "D_out" in g.ndata:
            D_out = g.ndata["D_out"]
        else:
            D_out = 1. / g.out_degrees().float().sqrt().unsqueeze(1)

        h = features * D_out
        g.ndata['h'] = h
        # w is the weights of edges
        # I saved the aggregation norm computed during sampling into ndata["w"]
        if 'w' not in g.edata:
            g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
        g.update_all(fn.u_mul_e('h', 'w', 'm'),
                     fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        h = h * D_in

        h = self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, activation=F.relu):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()
        self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, activation=activation))
        for _ in range(n_layers - 2):
            self.gcn.append(GCNLayer(in_dim=hid_dim, out_dim=hid_dim, activation=activation))
        self.gcn.append(GCNLayer(in_dim=hid_dim, out_dim=out_dim, activation=None))

    def forward(self, graph):
        h = graph.ndata['feat']
        for layer in self.gcn:
            h = layer(graph, h)
        return h
