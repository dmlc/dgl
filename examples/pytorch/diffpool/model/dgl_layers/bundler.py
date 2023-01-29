import torch
import torch.nn as nn
import torch.nn.functional as F


class Bundler(nn.Module):
    """
    Bundler, which will be the node_apply function in DGL paradigm
    """

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(Bundler, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)
        self.activation = activation

        nn.init.xavier_uniform_(
            self.linear.weight, gain=nn.init.calculate_gain("relu")
        )

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data["h"]
        c = node.data["c"]
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}
