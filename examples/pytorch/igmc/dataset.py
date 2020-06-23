import random
from collections import namedtuple

import numpy as np
import torch as th
from torch.utils.data import Dataset
import dgl

from utils import subgraph_extraction_labeling

MovieLensDataTuple = namedtuple('MovieLensDataTuple', ['g', 'g_label', 'x', 'etype'])

class MovieLensDatasetStatic(Dataset):
    def __init__(self, subgraphs_info, mode, edge_dropout=0.2, force_undirected=False):
        self.subgraphs = subgraphs_info
        # self.device = device
        self.mode = mode
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
    
    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        subgraph = self.subgraphs[idx]
        # if self.mode == 'train' and self.edge_dropout > 0:
        #     subgraph = dropout_link(subgraph, self.edge_dropout, self.force_undirected)
        try:
            return create_dgl_graph(subgraph)
        except:
            to_try = np.random.randint(0, len(self.subgraphs))
            return self.__getitem__(to_try)

class MovieLensDatasetDynamic(Dataset):
    def __init__(self, links, g_labels, adj, 
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1, 
                mode='train', edge_dropout=0.2, force_undirected=False):
        self.links = links
        self.g_labels = g_labels
        self.adj = adj 

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_node_label = max_node_label

        # self.device = device
        self.mode = mode
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
    
    def __len__(self):
        return len(self.links[0])

    def __getitem__(self, idx):
        # for u, v, g_label in zip(links[0], links[1], g_labels):
        u, v = self.links[0][idx], self.links[1][idx]
        g_label = self.g_labels[idx]

        subgraph = subgraph_extraction_labeling(
            g_label, (u, v), self.adj, 
            self.hop, self.sample_ratio, self.max_node_label, self.max_nodes_per_hop)
                    
        # if self.mode == 'train' and self.edge_dropout > 0:
        #     subgraph = dropout_link(subgraph, self.edge_dropout, self.force_undirected)
        try:
            return create_dgl_graph(subgraph)
        except:
            to_try = np.random.randint(0, len(self.subgraphs))
            return self.__getitem__(to_try)

# def dropout_link(subgraph, edge_dropout=0.2, force_undirected=False):
#     assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'
#     n_edges = subgraph['etype'].shape[0] // 2 if force_undirected else subgraph['etype'].shape[0]
    
#     drop_mask = np.random.binomial([np.ones(n_edges)],1-edge_dropout)[0].astype(np.bool)

#     # DGL graph created with edge has to start with node 0, thus we force the first link to be valid
#     drop_mask[0] = True
#     if force_undirected:
#         drop_mask = np.concatenate([drop_mask, drop_mask], axis=0)
    
#     src, dst, etype = subgraph['src'][drop_mask], subgraph['dst'][drop_mask], subgraph['etype'][drop_mask]

#     max_node_idx = np.max(np.concatenate([src, dst])) + 1
#     x = subgraph['x'][:max_node_idx]
#     new_subgraph = {'g_label': subgraph['g_label'], 'src': src, 'dst': dst, 'etype': etype, 'x': x}
#     return new_subgraph

def create_dgl_graph(subgraph):
    g = dgl.DGLGraph((subgraph['src'], subgraph['dst']))
    return MovieLensDataTuple(g=g, g_label=subgraph['g_label'], 
                            x=subgraph['x'], etype=subgraph['etype'])

def collate_movielens(data):
    g_list, g_label, x, etype = map(list, zip(*data))
    g = dgl.batch(g_list)
    # TODO [tianjun] will this earase all the feature?
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    x = th.tensor(np.concatenate(x, axis=0), dtype=th.float)
    g.ndata['x'] = x
    etype = th.tensor(np.concatenate(etype, axis=0), dtype=th.long)
    g.edata['etype'] = etype
    g_label = th.tensor(np.array(g_label), dtype=th.float)
    return g, g_label

if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    from data import MovieLens
    from utils import links2subgraphs

    train_val = True
    dynamic = True

    dataset = MovieLens("ml-100k", train_val=train_val)
    if not dynamic:
        train_graphs, val_graphs, test_graphs = links2subgraphs(
                dataset.rating_mx_train, # dataset.rating_values, pool,
                dataset.train_rating_pairs, dataset.valid_rating_pairs, dataset.test_rating_pairs,
                dataset.train_rating_values, dataset.valid_rating_values, dataset.test_rating_values,
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1,
                train_val=train_val, parallel=False)
        
        train_dataset = MovieLensDatasetStatic(train_graphs, mode='train', edge_dropout=0.2, force_undirected=False)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=collate_movielens)
        val_dataset = MovieLensDatasetStatic(val_graphs, mode='test', edge_dropout=0.2, force_undirected=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_movielens)
        test_dataset = MovieLensDatasetStatic(test_graphs, mode='test', edge_dropout=0.2, force_undirected=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_movielens)
    else:
        train_dataset = MovieLensDatasetDynamic(dataset.train_rating_pairs, dataset.train_rating_values, dataset.rating_mx_train, 
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1, 
                                    mode='train', edge_dropout=0.2)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=collate_movielens)
        test_dataset = MovieLensDatasetDynamic(dataset.test_rating_pairs, dataset.test_rating_values, dataset.rating_mx_train, 
                                    hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1, 
                                    mode='test', edge_dropout=0.2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_movielens)
        if train_val:
                val_dataset = MovieLensDatasetDynamic(dataset.valid_rating_pairs, dataset.valid_rating_values, dataset.rating_mx_train, 
                                            hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1, 
                                            mode='valid', edge_dropout=0.2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_movielens)
        
        with tqdm(total=len(train_loader)) as pbar:
            for _ in train_loader:
                pass
                pbar.update(1)
