import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, order=1, act=None,
                 dropout=0, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.lins = nn.ModuleList()
        for _ in range(order + 1):
            self.lins.append(nn.Linear(in_dim, out_dim, bias=True))

        self.order = order
        self.act = act
        self.dropout = nn.Dropout(dropout)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim)
        else:
            self.batch_norm = None

    def feat_trans(self, features, idx):  # linear transformation + activation + batch normalization
        h = self.lins[idx](features)

        if self.act is not None:
            h = self.act(h)

        if self.batch_norm:
            h = self.batch_norm(h)

        return h

    def forward(self, graph, features):
        g = graph.local_var()
        h_in = self.dropout(features)
        h_hop = [h_in]

        D_norm = g.ndata['D_norm']
        for _ in range(self.order):  # forward propagation
            g.ndata['h'] = h_hop[-1]
            if 'w' in g.edata:
                g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            else:
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = g.ndata['h'] * D_norm
            h_hop.append(h)

        h_part = [self.feat_trans(ft, idx) for idx, ft in enumerate(h_hop)]
        h_out = th.cat(h_part, 1)

        return h_out

class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, arch="1-1-0",
                 act=F.relu, dropout=0, batch_norm=True):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()

        orders = list(map(int, arch.split('-')))
        self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, order=orders[0],
                                 act=act, dropout=dropout, batch_norm=batch_norm))
        pre_out = (orders[0] + 1) * hid_dim

        for i in range(1, len(orders)-1):
            self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[i],
                                     act=act, dropout=dropout, batch_norm=batch_norm))
            pre_out = (orders[i] + 1) * hid_dim

        self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, order=orders[-1],
                                 act=act, dropout=dropout, batch_norm=batch_norm))
        pre_out = (orders[-1] + 1) * hid_dim

        self.out_layer = GCNLayer(in_dim=pre_out, out_dim=out_dim, order=0,
                                  act=None, dropout=dropout, batch_norm=False)

    def forward(self, graph):
        h = graph.ndata['feat']

        for layer in self.gcn:
            h = layer(graph, h)

        h = F.normalize(h, p=2, dim=1)
        h = self.out_layer(graph, h)

        return h
