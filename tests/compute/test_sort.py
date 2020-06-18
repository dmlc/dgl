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
    # print(num_adj_list)
    src = np.concatenate([[i] * num_adj_list[i] for i in range(num_nodes)])
    # print(src)
    # print(type(num_adj_list[0]))
    dst = [np.random.choice(num_nodes, nadj, replace=False) for nadj in num_adj_list]
    dst = np.concatenate(dst)
    # print(src, dst)
    return dgl.graph((src, dst), index_dtype=index_dtype)

def check_sort(spm, tag_arr, split):
    for i in range(spm.shape[0]):
        row = spm.getrow(i)
        split_row = split[i]
        dst = row.nonzero()[1]
        split_ptr = tag_arr[dst[0]]
        for j in range(len(dst) - 1):
            if tag_arr[dst[j]] != split_ptr:
                return False
            if  tag_arr[dst[j]] > tag_arr[dst[j+1]]:
                return False
            if tag_arr[dst[j]] < tag_arr[dst[j+1]]:
                if j+1 != int(split_row[split_ptr+1]):
                    return False
                split_ptr = tag_arr[dst[j+1]]
    return True

@parametrize_dtype
def test_sort_inplace(index_dtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype=index_dtype)
    g.ndata['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes()))
    split = dgl.sort_csr_(g, 'tag')
    csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(csr, g.ndata['tag'], split))

    split = dgl.sort_csc_(g, 'tag')
    csc = g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(csc, g.ndata['tag'], split))

@parametrize_dtype
def test_sort_inplace_bipartite(index_dtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype=index_dtype)
    g = dgl.bipartite(g.edges(), index_dtype=index_dtype)
    g.nodes['_U'].data['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_U')))
    g.nodes['_V'].data['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_V')))
    split = dgl.sort_csr_(g, 'tag')
    csr = g.adjacency_matrix(etype='_E', transpose=True, scipy_fmt='csr')
    assert(check_sort(csr, g.nodes['_V'].data['tag'], split))

    split = dgl.sort_csc_(g, 'tag')
    csc = g.adjacency_matrix(etype='_E', scipy_fmt='csr')
    assert(check_sort(csc, g.nodes['_U'].data['tag'], split))

@parametrize_dtype
def test_sort_outplace(index_dtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype=index_dtype)
    g.ndata['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes()))
    g.ndata['tag'][0] = 1
    g.ndata['tag'][1] = 0
    new_g, split = dgl.sort_csr(g, 'tag')
    old_csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(new_csr, g.ndata['tag'], split))
    assert(not check_sort(old_csr, g.ndata['tag'], split))

    new_g, split = dgl.sort_csc(g, 'tag')
    old_csc = g.adjacency_matrix(scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(new_csc, new_g.ndata['tag'], split))
    assert(not check_sort(old_csc, g.ndata['tag'], split))

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

    new_g, split = dgl.sort_csr(g, 'tag')
    old_csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(new_csr, new_g.nodes['_V'].data['tag'], split))
    assert(not check_sort(old_csr, g.nodes['_V'].data['tag'], split))

    new_g, split = dgl.sort_csc(g, 'tag')
    old_csc = g.adjacency_matrix(scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(new_csc, new_g.nodes['_U'].data['tag'], split))
    assert(not check_sort(old_csc, g.nodes['_U'].data['tag'], split))

if __name__ == "__main__":
    # test_sort_inplace("int32")
    test_sort_outplace_bipartite("int32")
    # test_sort_outplace("int32")
    # test_biased_sampling()