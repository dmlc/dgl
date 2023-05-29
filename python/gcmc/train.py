import sys
sys.path.append('..')
import dgl
from dgl import edge_subgraph
from dgl.data import MovieLensDataset
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
import torch as th
import torch.nn as nn

class RateConv(nn.Module):
    def __init__(self, rate_list, in_dim, out_dim):
        super(RateConv, self).__init__()
        self.rate_list = rate_list
        self.convs = nn.ModuleList()
        for _ in rate_list:
            self.convs.append(GraphConv(in_dim, out_dim))

    def forward(self, g, h):
        h_list = []
        for rate, conv in zip(self.rate_list, self.convs):
            _g = edge_subgraph(g, g.edges()[g.edata['rate'] == rate])
            h = conv(_g, h)
            h_list.append(h)
        h = th.concat(h_list, dim=1)
        return h

class Model(nn.Module):
    def __init__(self, rate_list, in_dim:dict, hid_dim:int, out_dim:int):
        super(Model, self).__init__()
        self.conv = HeteroGraphConv({
            'user-movie': RateConv(rate_list, in_dim['user'], hid_dim),
            'movie-user': RateConv(rate_list, in_dim['movie'], hid_dim),
        })
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, g, h):
        h = self.conv(g, h)
        return self.fc(h)
    
dataset = MovieLensDataset('ml-100k', valid_ratio=0.2)
graph = dataset[0].to('cuda')
in_dim = {'user': graph.nodes['user'].data['feat'].shape[1], 'movie': graph.nodes['movie'].data['feat'].shape[1]}
hid_dim, out_dim = 500, 75
rate_list = graph.edges['user-movie'].data['rate'].unique(sorted=True).tolist()
model = Model(rate_list, in_dim, hid_dim, out_dim).to('cuda')
pass







