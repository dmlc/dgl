import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, JumpingKnowledge


class JKNet(nn.Module):
    def __init__(
        self, in_dim, hid_dim, out_dim, num_layers=1, mode="cat", dropout=0.0
    ):
        super(JKNet, self).__init__()

        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(num_layers):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))

        if self.mode == "lstm":
            self.jump = JumpingKnowledge(mode, hid_dim, num_layers)
        else:
            self.jump = JumpingKnowledge(mode)

        if self.mode == "cat":
            hid_dim = hid_dim * (num_layers + 1)

        self.output = nn.Linear(hid_dim, out_dim)
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, g, feats):
        feat_lst = []
        for layer in self.layers:
            feats = self.dropout(layer(g, feats))
            feat_lst.append(feats)

        if self.mode == "lstm":
            self.jump.lstm.flatten_parameters()

        g.ndata["h"] = self.jump(feat_lst)
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))

        return self.output(g.ndata["h"])
