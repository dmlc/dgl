import dgl
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import itertools
import backend as F
import networkx as nx
import unittest, pytest
from dgl import DGLError
from utils import parametrize_dtype

def create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype):
    if isinstance(num_adj, int):
        num_adj = [num_adj, num_adj+1]
    num_adj_list = list(np.random.choice(np.arange(num_adj[0], num_adj[1]), num_nodes))
    src = np.concatenate([[i] * num_adj_list[i] for i in range(num_nodes)])
    dst = [np.random.choice(num_nodes, nadj, replace=False) for nadj in num_adj_list]
    dst = np.concatenate(dst)
    return dgl.graph((src, dst), index_dtype=index_dtype)

def check_sort(spm, tag_arr = None, split = None):
    if tag_arr is None:
        tag_arr = np.arange(spm.shape[0])
    for i in range(spm.shape[0]):
        row = spm.getrow(i)
        dst = row.nonzero()[1]
        if split is not None:
            split_row = split[i]
            split_ptr = tag_arr[dst[0]]
        for j in range(len(dst) - 1):
            if split is not None and tag_arr[dst[j]] != split_ptr:
                return False
            if tag_arr[dst[j]] > tag_arr[dst[j+1]]:
                return False
            if split is not None and tag_arr[dst[j]] < tag_arr[dst[j+1]]:
                if j+1 != int(split_row[split_ptr+1]):
                    return False
                split_ptr = tag_arr[dst[j+1]]
    return True

@parametrize_dtype
def test_sort_inplace(index_dtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype=index_dtype)
    g.ndata['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes()))

    dgl.sort_csr_(g, 'tag')
    csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(csr, g.ndata['tag'], g.ndata['_SPLIT']))

    dgl.sort_csc_(g, 'tag')
    csc = g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(csc, g.ndata['tag'], g.ndata['_SPLIT']))

@parametrize_dtype
def test_sort_inplace_bipartite(index_dtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype=index_dtype)
    g = dgl.bipartite(g.edges(), index_dtype=index_dtype)
    g.nodes['_U'].data['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_U')))
    g.nodes['_V'].data['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_V')))
    
    dgl.sort_csr_(g, 'tag')
    csr = g.adjacency_matrix(etype='_E', transpose=True, scipy_fmt='csr')
    assert(check_sort(csr, g.nodes['_V'].data['tag'], g.nodes['_U'].data['_SPLIT']))

    dgl.sort_csc_(g, 'tag')
    csc = g.adjacency_matrix(etype='_E', scipy_fmt='csr')
    assert(check_sort(csc, g.nodes['_U'].data['tag'],  g.nodes['_V'].data['_SPLIT']))

@parametrize_dtype
def test_sort_outplace(index_dtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype=index_dtype)
    g.ndata['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes()))
    g.ndata['tag'][0] = 1
    g.ndata['tag'][1] = 0
    new_g = dgl.sort_csr(g, 'tag')
    old_csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(new_csr, new_g.ndata['tag'], new_g.ndata["_SPLIT"]))
    assert(not check_sort(old_csr, g.ndata['tag']))

    new_g = dgl.sort_csc(g, 'tag')
    old_csc = g.adjacency_matrix(scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(new_csc, new_g.ndata['tag'], new_g.ndata["_SPLIT"]))
    assert(not check_sort(old_csc, g.ndata['tag']))

@parametrize_dtype
def test_sort_outplace_bipartite(index_dtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype=index_dtype)
    g = dgl.bipartite(g.edges(), index_dtype=index_dtype)
    utag = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_U')))
    utag[0] = 1
    utag[1] = 0

    vtag = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_U')))
    vtag[0] = 1
    vtag[1] = 0

    g.nodes['_V'].data['tag'] = vtag
    g.nodes['_U'].data['tag'] = utag

    new_g = dgl.sort_csr(g, 'tag')
    old_csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(new_csr, new_g.nodes['_V'].data['tag'], new_g.nodes['_U'].data['_SPLIT']))
    assert(not check_sort(old_csr, g.nodes['_V'].data['tag']))

    new_g = dgl.sort_csc(g, 'tag')
    old_csc = g.adjacency_matrix(scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(new_csc, new_g.nodes['_U'].data['tag'], new_g.nodes['_V'].data['_SPLIT']))
    assert(not check_sort(old_csc, g.nodes['_U'].data['tag']))

if __name__ == "__main__":
    test_sort_inplace("int32")
    test_sort_inplace_bipartite("int32")
    test_sort_outplace("int32")
    test_sort_outplace_bipartite("int32")

    # test_biased_sampling()