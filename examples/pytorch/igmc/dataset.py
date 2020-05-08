import os
import dgl
import random
import numpy as np
from collections import namedtuple
import torch as th
from torch.utils.data import Dataset

MovieLensDataTuple = namedtuple('MovieLensDataTuple', ['g', 'g_label', 'user_feature', 'movie_feature', 'device'])

class MovieLensDataset(Dataset):
    def __init__(self, subgraphs_info, device):
        self.subgraphs = subgraphs_info
        self.device = device
    
    def __len__(self):
        return len(self.subgraphs)

    def create_dgl_graph(self, subgraph):
        g_label = subgraph.pop('g_label')
        user_feature = subgraph.pop('u')
        movie_feature = subgraph.pop('v')
        g = dgl.heterograph(subgraph)
        return MovieLensDataTuple(g=g, g_label=g_label, user_feature=user_feature, movie_feature=movie_feature, device=self.device)

    def __getitem__(self, idx):
        return self.create_dgl_graph(self.subgraphs[idx])

def collate_movielens(data):
    g_list, g_label, user_feature, movie_feature, device = map(list, zip(*data))
    g = dgl.batch_hetero(g_list)
    # TODO: will this earase all the feature?
    g.set_n_initializer(dgl.init.zero_initializer, ntype='user')
    g.set_n_initializer(dgl.init.zero_initializer, ntype='movie')
    for et in g.canonical_etypes:
        g.set_e_initializer(dgl.init.zero_initializer, etype=et)
    user_feature = th.tensor(np.concatenate(user_feature, axis=0), dtype=th.float, device=device[0])
    movie_feature = th.tensor(np.concatenate(movie_feature, axis=0), dtype=th.float, device=device[0])
    g.nodes['user'].data['x'] = user_feature
    g.nodes['movie'].data['x'] = movie_feature
    g_label = th.tensor(np.array(g_label), dtype=th.float, device=device[0]) 
    return g, g_label
