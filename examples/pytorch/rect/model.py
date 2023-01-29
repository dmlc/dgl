import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.gcn_1 = GraphConv(in_feats, n_hidden, activation=activation)
        self.gcn_2 = GraphConv(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = self.gcn_1(self.g, features)
        h = self.dropout(h)
        preds = self.gcn_2(self.g, h)
        return preds

    def embed(self, inputs):
        h_1 = self.gcn_1(self.g, inputs)
        return h_1.detach()


class RECT_L(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation, dropout=0.0):
        super(RECT_L, self).__init__()
        self.g = g
        self.gcn_1 = GraphConv(in_feats, n_hidden, activation=activation)
        self.fc = nn.Linear(n_hidden, in_feats)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, inputs):
        h_1 = self.gcn_1(self.g, inputs)
        h_1 = F.dropout(h_1, p=self.dropout, training=self.training)
        preds = self.fc(h_1)
        return preds

    # Detach the return variables
    def embed(self, inputs):
        h_1 = self.gcn_1(self.g, inputs)
        return h_1.detach()
