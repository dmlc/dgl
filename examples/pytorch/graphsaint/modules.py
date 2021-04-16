import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, order=1,
                 act=None, dropout=0, batch_norm=False, aggr="concat"):
        super(GCNLayer, self).__init__()
        self.lins = nn.ModuleList()
        for _ in range(order + 1):
            self.lins.append(nn.Linear(in_dim, out_dim, bias))
        self.order = order
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bns = nn.ModuleList()
            for _ in range(order + 1):
                self.bns.append(nn.BatchNorm1d(out_dim))
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def compute_degree(self, g):
        if 'D_in' in g.ndata:
            D_in = g.ndata['D_in']
        else:
            D_in = 1. / g.in_degrees().float().clamp(min=1).unsqueeze(1)
        return D_in

    def feat_trans(self, features, idx):
        h = self.lins[idx](features)
        if self.act is not None:
            h = self.act(h)
        if self.batch_norm:
            h = self.bns[idx](h)
        return h

    def forward(self, graph, features):
        g = graph.local_var()
        h_in = self.dropout(features)
        h_hop = [h_in]

        for _ in range(self.order):
            D_in = self.compute_degree(g)
            g.ndata['h'] = h_hop[-1]
            if 'w' not in g.edata:
                g.edata['w'] = th.ones((g.num_edges(), )).to(features.device)
            g.update_all(fn.u_mul_e('h', 'w', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            h = h * D_in
            h_hop.append(h)

        h_part = [self.feat_trans(ft, idx) for idx, ft in enumerate(h_hop)]
        if self.aggr == "mean":
            h_out = h_part[0]
            for i in range(len(h_part) - 1):
                h_out = h_out + h_part[i + 1]
        elif self.aggr == "concat":
            h_out = th.cat(h_part, 1)
        else:
            raise NotImplementedError
        return h_out


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, bias=True, arch="1-1-0",
                 act=F.relu, dropout=0, batch_norm=False, aggr="concat"):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()

        orders = list(map(int, arch.split('-')))
        self.gcn.append(GCNLayer(in_dim=in_dim, out_dim=hid_dim, bias=bias, order=orders[0],
                                 act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        pre_out = ((aggr == "concat") * orders[0] + 1) * hid_dim

        for i in range(1, len(orders)-1):
            self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, bias=bias, order=orders[i],
                                     act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
            pre_out = ((aggr == "concat") * orders[i] + 1) * hid_dim

        self.gcn.append(GCNLayer(in_dim=pre_out, out_dim=hid_dim, bias=bias, order=orders[-1],
                                 act=act, dropout=dropout, batch_norm=batch_norm, aggr=aggr))
        pre_out = ((aggr == "concat") * orders[-1] + 1) * hid_dim

        self.out_layer = GCNLayer(in_dim=pre_out, out_dim=out_dim, bias=bias, order=0,
                                  act=None, dropout=dropout, batch_norm=False, aggr=aggr)

    def forward(self, graph):
        h = graph.ndata['feat']
        for layer in self.gcn:
            h = layer(graph, h)
        h = F.normalize(h, p=2, dim=1)
        h = self.out_layer(graph, h)

        return h

