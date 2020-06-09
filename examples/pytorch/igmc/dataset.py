import os
import dgl
import random
import numpy as np
from collections import namedtuple
import torch as th
from torch.utils.data import Dataset

MovieLensDataTuple = namedtuple('MovieLensDataTuple', ['g', 'g_label', 'x', 'etype', 'device'])

class MovieLensDataset(Dataset):
    def __init__(self, subgraphs_info, device, mode, link_dropout=0.0, force_undirected=False):
        self.subgraphs = subgraphs_info
        self.device = device
        self.mode = mode
        self.link_dropout = link_dropout
        self.force_undirected = force_undirected
    
    def __len__(self):
        return len(self.subgraphs)

    def dropout_link(self, subgraph):
        assert self.link_dropout >= 0.0 and self.link_dropout <= 1.0, 'Invalid dropout rate.'
        n_edges = subgraph['etype'].shape[0] // 2 if self.force_undirected else subgraph['etype'].shape[0]
        
        drop_mask = np.random.binomial([np.ones(n_edges)],1-self.link_dropout)[0].astype(np.bool)

        # DGL graph created with edge has to start with node 0, thus we force the first link to be valid
        drop_mask[0] = True
        if self.force_undirected:
            drop_mask = np.concatenate([drop_mask, drop_mask], axis=0)
        
        src, dst, etype = subgraph['src'][drop_mask], subgraph['dst'][drop_mask], subgraph['etype'][drop_mask]

        max_node_idx = np.max(np.concatenate([src, dst])) + 1
        x = subgraph['x'][:max_node_idx]
        new_subgraph = {'g_label': subgraph['g_label'], 'src': src, 'dst': dst, 'etype': etype, 'x': x}
        return new_subgraph

    def create_dgl_graph(self, subgraph):
        g = dgl.DGLGraph((subgraph['src'], subgraph['dst']))
        return MovieLensDataTuple(g=g, g_label=subgraph['g_label'], x=subgraph['x'], etype=subgraph['etype'], device=self.device)

    def __getitem__(self, idx):
        subgraph = self.subgraphs[idx]
        if self.mode == 'train' and self.link_dropout > 0:
            subgraph = self.dropout_link(subgraph)
        try:
            return self.create_dgl_graph(subgraph)
        except:
            to_try = np.random.randint(0, len(self.subgraphs))
            return self.__getitem__(to_try)

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
