

import torch as th
import dgl
from scipy.sparse import csc_matrix
import numpy as np
import os
import contextlib

class SSDGraph:
    def __init__(self, graph_path, idtype=th.int64, device='cpu'):
        self.idtype = idtype
        self.device = device
        self.ndata = {}
        self.edata = {}

        # Set properties to pass some property tests in dgl.NodeDataLoader
        self.indptr = th.as_tensor(np.load(os.path.join(graph_path, 'indptr.npy'), mmap_mode='r'), device=self.device)
        self.n_nodes = self.indptr.shape[0] - 1
        self.indices = th.as_tensor(np.load(os.path.join(graph_path, 'indices.npy'), mmap_mode='r'), device=self.device)
        self.n_edges = self.indices.shape[0]
        self._in_degrees = th.as_tensor(np.load(os.path.join(graph_path, 'in_degrees.npy'), mmap_mode='r'), device=self.device)
        assert self.indptr[0] == 0, "indptr[0] {}".format(self.indptr[0])

    def number_of_nodes(self):
        return self.n_nodes

    def in_degrees(self, nids):
        return self._in_degrees[nids]
    
    def get_node_storage(self):
        raise NotImplementedError 

    def get_edge_storage(self):
        raise NotImplementedError 


class SSDBlockSampler(dgl.dataloading.NeighborSampler):
    def __init__(self):
        super(SSDBlockSampler, self).__init__([-1])

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        seed_nodes = seed_nodes.cpu()
        # To subcsc
        start = seed_nodes[0].item()
        end = seed_nodes[-1].item() + 1
        sub_indptr = g.indptr[start:end+1]
        sub_indices = g.indices[sub_indptr[0]:sub_indptr[-1]]

        # Construct a dgl graph
        ind_diff = np.diff(sub_indptr)
        lhs = sub_indices
        rhs = seed_nodes.cpu().numpy().repeat(ind_diff)
        dgl_graph = dgl.graph((lhs, rhs))

        # To block
        block = dgl.transforms.to_block(dgl_graph, seed_nodes)
        input_nodes = block.srcdata[dgl.NID]
        output_nodes = block.dstdata[dgl.NID]
        return input_nodes, output_nodes, [block]
