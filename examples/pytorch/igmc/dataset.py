import os
import dgl
import random
import numpy as np
from collections import namedtuple
import torch as th
from torch.utils.data import Dataset

MovieLensDataTuple = namedtuple('MovieLensDataTuple', ['g', 'g_label', 'x', 'etype', 'device'])

class MovieLensDataset(Dataset):
    def __init__(self, subgraphs_info, device):
        self.subgraphs = subgraphs_info
        self.device = device
    
    def __len__(self):
        return len(self.subgraphs)

    def create_dgl_graph(self, subgraph):
        g = dgl.DGLGraph((subgraph['src'], subgraph['dst']))
        return MovieLensDataTuple(g=g, g_label=subgraph['g_label'], x=subgraph['x'], etype=subgraph['etype'], device=self.device)

    def __getitem__(self, idx):
        return self.create_dgl_graph(self.subgraphs[idx])

def collate_movielens(data):
    g_list, g_label, x, etype, device = map(list, zip(*data))
    g = dgl.batch(g_list)
    # TODO: will this earase all the feature?
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    x = th.tensor(np.concatenate(x, axis=0), dtype=th.float, device=device[0])
    g.ndata['x'] = x 
    etype = th.tensor(np.concatenate(etype, axis=0), dtype=th.long, device=device[0])
    g.edata['etype'] = etype
    g_label = th.tensor(np.array(g_label), dtype=th.float, device=device[0]) 
    return g, g_label
