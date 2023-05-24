import unittest

import backend as F

import dgl.graphbolt as gb

import pytest
import torch

#from .utils import random_hetero_graph, csc_to_coo
from utils import random_hetero_graph, csc_to_coo

def test_sample_etype_with_replace(seed_num, num_nodes, num_edges, fanouts):
    data = random_hetero_graph(num_nodes, num_edges, num_ntypes=3, num_etypes=2)
    graph = gb.from_csc(*data)
    coo = csc_to_coo(graph.csc_indptr, graph.indices)
    coo = torch.cat([coo, graph.type_per_edge.unsqueeze(dim=0)])
    seed_nodes = torch.randint(0, num_nodes, (seed_num,))
    sub_graph = graph.sample_etype_neighbors(seed_nodes, fanouts, replace=True, return_eids=True)
    # expected_coo = torch.index_select(coo, 1, eid)
    # assert torch.equal(induced_coo, expected_coo)
    
if __name__ == '__main__':
    fanouts = torch.tensor([2, 3], dtype=int)
    test_sample_etype_with_replace(3, 10, 50, fanouts)