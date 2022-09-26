import torch.nn as nn
import torch.nn.functional as function

from dgl.nn import GraphConv, SumPooling


class EEGGraphConvNet(nn.Module):
    """EEGGraph Convolution Net
    Parameters
    ----------
    num_feats: the number of features per node. In our case, it is 6.
    """

    def __init__(self, num_feats):
        super(EEGGraphConvNet, self).__init__()

        self.conv1 = GraphConv(num_feats, 32)
        self.conv2 = GraphConv(32, 20)
        self.conv2_bn = nn.BatchNorm1d(
            20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.fc_block1 = nn.Linear(20, 10)
        self.fc_block2 = nn.Linear(10, 2)

        # Xavier initializations
        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1))

    def forward(self, g, return_graph_embedding=False):
        x = g.ndata["x"]
        edge_weight = g.edata["edge_weights"]

        x = function.leaky_relu(self.conv1(g, x, edge_weight=edge_weight))
        x = function.leaky_relu(
            self.conv2_bn(self.conv2(g, x, edge_weight=edge_weight))
        )

        # NOTE: this takes node-level features/"embeddings"
        # and aggregates to graph-level - use for graph-level classification
        sumpool = SumPooling()
        out = sumpool(g, x)
        if return_graph_embedding:
            return out

        out = function.dropout(out, p=0.2, training=self.training)
        out = self.fc_block1(out)
        out = function.leaky_relu(out)
        out = self.fc_block2(out)

        return out
