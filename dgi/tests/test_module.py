import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import pytest
from dgl.data import CiteseerGraphDataset
from dgi import InferenceHelper


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
        x3 = F.relu(self.conv3(graph, x2))
        return x3

class GCN_2(GCN):
    def forward(self, blocks, x0):
        x_dst0 = x0[:blocks[0].number_of_dst_nodes()]
        x1 = F.relu(self.conv1(blocks[0], (x0, x_dst0)))
        x_dst1 = x1[:blocks[1].number_of_dst_nodes()]
        x2 = F.relu(self.conv2(blocks[1], (x1, x_dst1)))
        x_dst2 = x2[:blocks[2].number_of_dst_nodes()]
        x3 = F.relu(self.conv3(blocks[2], (x2, x_dst2)))
        return x3

def eval(_class):
    dataset = CiteseerGraphDataset(verbose=False)
    g : dgl.DGLHeteroGraph = dataset[0]
    feat = g.ndata['feat']
    labels = g.ndata['label']
    num_classes = dataset.num_classes
    in_feats = feat.shape[1]
    hidden_feature = 256

    model = _class(in_feats, hidden_feature, num_classes)

    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    loss_fcn = nn.CrossEntropyLoss()

    with torch.no_grad():
        helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
        helper_pred = helper.inference(g, feat)
        helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)

def test_GCN():
    eval(GCN)

def test_GCN_2():
    eval(GCN_2)

if __name__ == "__main__":
    test_GCN()
    test_GCN_2()
