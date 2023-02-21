import torch.nn as nn
import torch.nn.functional as function

from dgl.nn import GraphConv, SumPooling
from torch.nn import BatchNorm1d


class EEGGraphConvNet(nn.Module):
    """EEGGraph Convolution Net
    Parameters
    ----------
    num_feats: the number of features per node. In our case, it is 6.
    """

    def __init__(self, num_feats):
        super(EEGGraphConvNet, self).__init__()

        self.conv1 = GraphConv(num_feats, 16)
        self.conv2 = GraphConv(16, 32)
        self.conv3 = GraphConv(32, 64)
        self.conv4 = GraphConv(64, 50)
        self.conv4_bn = BatchNorm1d(
            50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.fc_block1 = nn.Linear(50, 30)
        self.fc_block2 = nn.Linear(30, 10)
        self.fc_block3 = nn.Linear(10, 2)

        # Xavier initializations
        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))
        self.fc_block3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))

        self.sumpool = SumPooling()

    def forward(self, g, return_graph_embedding=False):
        x = g.ndata["x"]
        edge_weight = g.edata["edge_weights"]

        x = self.conv1(g, x, edge_weight=edge_weight)
        x = function.leaky_relu(x, negative_slope=0.01)
        x = function.dropout(x, p=0.2, training=self.training)

        x = self.conv2(g, x, edge_weight=edge_weight)
        x = function.leaky_relu(x, negative_slope=0.01)
        x = function.dropout(x, p=0.2, training=self.training)

        x = self.conv3(g, x, edge_weight=edge_weight)
        x = function.leaky_relu(x, negative_slope=0.01)
        x = function.dropout(x, p=0.2, training=self.training)

        x = self.conv4(g, x, edge_weight=edge_weight)
        x = self.conv4_bn(x)
        x = function.leaky_relu(x, negative_slope=0.01)
        x = function.dropout(x, p=0.2, training=self.training)
        # NOTE: this takes node-level features/"embeddings"
        # and aggregates to graph-level - use for graph-level classification

        out = self.sumpool(g, x)
        if return_graph_embedding:
            return out

        out = function.leaky_relu(self.fc_block1(out), negative_slope=0.1)
        out = function.dropout(out, p=0.2, training=self.training)

        out = function.leaky_relu(self.fc_block2(out), negative_slope=0.1)
        out = function.dropout(out, p=0.2, training=self.training)

        out = self.fc_block3(out)
        return out
