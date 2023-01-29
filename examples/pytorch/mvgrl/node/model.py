import torch as th
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import AvgPooling


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits


class MVGRL(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MVGRL, self).__init__()

        self.encoder1 = GraphConv(
            in_dim, out_dim, norm="both", bias=True, activation=nn.PReLU()
        )
        self.encoder2 = GraphConv(
            in_dim, out_dim, norm="none", bias=True, activation=nn.PReLU()
        )
        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, graph, diff_graph, feat, edge_weight):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        return (h1 + h2).detach()

    def forward(self, graph, diff_graph, feat, shuf_feat, edge_weight):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        h3 = self.encoder1(graph, shuf_feat)
        h4 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)

        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        return out
