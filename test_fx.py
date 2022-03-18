from dgl.fx import dgl_symbolic_trace
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv3 = dgl.nn.GraphConv(hidden_features, out_features)
        self.n_layers = 3

    def forward(self, blocks, x0):
        x_dst0 = x0[:blocks[0].number_of_dst_nodes()]
        x1 = F.relu(self.conv1(blocks[0], (x0, x_dst0)))
        x_dst1 = x1[:blocks[1].number_of_dst_nodes()]
        x2 = F.relu(self.conv2(blocks[1], (x1, x_dst1)))
        x_dst2 = x2[:blocks[2].number_of_dst_nodes()]
        x3 = F.relu(self.conv3(blocks[2], (x2, x_dst2)))
        return x3


if __name__ == "__main__":
    model = StochasticTwoLayerGCN(5, 4, 3)
    feat = torch.ones((5, 5))
    g = dgl.graph((torch.tensor([0, 1, 2, 3, 4]), torch.tensor([4, 3, 2, 1, 0])))
    traced = dgl_symbolic_trace(model)
    print(traced.graph)
    print(traced.code)
    res = traced([g, g, g], feat)
    print(res)
