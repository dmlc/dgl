import os
import random

class MovieLensDataset(object):
    def __init__(self, subgraphs_info, device):
        self.subgraphs = subgraphs_info
        order = list(range(len(self.subgraphs)))
    
    def __len__(self):
        return len(self.subgraphs)

    def create_dgl_graph(self, subgraph):
        return dgl.heterograph()

    def __getitem__(self, idx):
        return self.create_dgl_graph(self.subgraphs[idx])

def collate_vertexgraphs(data):
    g = dgl.batch(g_list)
    # TODO: will this earase all the feature?
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

