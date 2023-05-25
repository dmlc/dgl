import unittest

import backend as F

import dgl.graphbolt as gb

import pytest
import torch

#from .utils import random_hetero_graph, csc_to_coo
from utils import random_hetero_graph, random_graph_with_fixed_neighbors

def test_sample_etype_without_replace(seed_num, num_nodes, neighbors_per_etype, num_ntypes, num_etypes, fanouts):
    graph = random_graph_with_fixed_neighbors(num_nodes, neighbors_per_etype, num_ntypes, num_etypes)
    seed_nodes = torch.randint(0, num_nodes, (seed_num,))
    sub_graph = graph.sample_etype_neighbors(seed_nodes, fanouts, replace=False, return_eids=True)
    
    eid = sub_graph.reverse_edge_ids

    assert torch.equal(seed_nodes, sub_graph.reverse_row_node_ids)
    assert sub_graph.indptr.is_contiguous()

    expected_indices = torch.index_select(graph.indices, 0, eid)
    assert torch.equal(sub_graph.indices, expected_indices)
    expected_etypes = torch.index_select(graph.type_per_edge, 0, eid)
    assert torch.equal(sub_graph.type_per_edge, expected_etypes)
    # assert indices
    num_per_node = sub_graph.indptr[1:] - sub_graph.indptr[:-1]
    sampled_etypes = torch.stack(torch.split(sub_graph.type_per_edge, num_per_node.tolist()))
    for etype in range(num_etypes):
        mask = sampled_etypes == etype
        etype_neighbor_num = torch.sum(mask, dim=1) 
        num_pick = fanouts[etype].item()
        if num_pick == -1 or num_pick >= neighbors_per_etype:
            assert torch.all(etype_neighbor_num == neighbors_per_etype)
        else:
            assert torch.all(etype_neighbor_num == num_pick)
    
    
if __name__ == '__main__':
    fanouts = torch.tensor([5, 5, 5, 5, 5], dtype=int)
    test_sample_etype_without_replace(30, 50, 5, 3, 5, fanouts)