import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils.split import FunctionGenerator


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv3 = dgl.nn.GraphConv(hidden_features, out_features)
        self.n_layers = 3

    def forward(self, graph, x0):
        x1 = F.relu(self.conv1(graph, x0))
        x2 = F.relu(self.conv2(graph, x1))
        return x2

class GCN_2(GCN):
    def forward(self, blocks, x0):
        x_dst0 = x0[:blocks[0].number_of_dst_nodes()]
        x1 = F.relu(self.conv1(blocks[0], (x0, x_dst0)))
        x_dst1 = x1[:blocks[1].number_of_dst_nodes()]
        x2 = F.relu(self.conv2(blocks[1], (x1, x_dst1)))
        return x2

def eval(_class):
    model = _class(5, 5, 5)
    generator = FunctionGenerator(True)
    generator.module_split(model)

def test_GCN():
    eval(GCN)

def test_GCN_2():
    eval(GCN_2)

if __name__ == "__main__":
    test_GCN()
    test_GCN_2()