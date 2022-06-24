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
from test_utils import parametrize_idtype

def create_test_heterograph(num_nodes, num_adj, idtype):
    if isinstance(num_adj, int):
        num_adj = [num_adj, num_adj+1]
    num_adj_list = list(np.random.choice(np.arange(num_adj[0], num_adj[1]), num_nodes))
    src = np.concatenate([[i] * num_adj_list[i] for i in range(num_nodes)])
    dst = [np.random.choice(num_nodes, nadj, replace=False) for nadj in num_adj_list]
    dst = np.concatenate(dst)
    return dgl.graph((src, dst), idtype=idtype)

def check_sort(spm, tag_arr=None, tag_pos=None):
    if tag_arr is None:
        tag_arr = np.arange(spm.shape[0])
    else:
        tag_arr = F.asnumpy(tag_arr)
    if tag_pos is not None:
        tag_pos = F.asnumpy(tag_pos)
    for i in range(spm.shape[0]):
        row = spm.getrow(i)
        dst = row.nonzero()[1]
        if tag_pos is not None:
            tag_pos_row = tag_pos[i]
            tag_pos_ptr = tag_arr[dst[0]] if len(dst) > 0 else 0
        for j in range(len(dst) - 1):
            if tag_pos is not None and tag_arr[dst[j]] != tag_pos_ptr:
                # `tag_pos_ptr` is the expected tag value. Here we check whether the
                # tag value is equal to `tag_pos_ptr`
                return False
            if tag_arr[dst[j]] > tag_arr[dst[j+1]]:
                # The tag should be in descending order after sorting
                return False
            if tag_pos is not None and tag_arr[dst[j]] < tag_arr[dst[j+1]]:
                if j+1 != int(tag_pos_row[tag_pos_ptr+1]):
                    # The boundary of tag should be consistent with `tag_pos`
                    return False
                tag_pos_ptr = tag_arr[dst[j+1]]
    return True


def check_sort_edge(old_spm, new_spm, tag, fmt):
    assert fmt in ['csr', 'csc']
    if fmt == 'csr':
        old_src, old_dst = old_spm.nonzero()
        new_src, new_dst = new_spm.nonzero()
    else:
        old_dst, old_src = old_spm.nonzero()
        new_dst, new_src = new_spm.nonzero()
    for nid in np.unique(old_dst):
        src = old_src[old_dst == nid]
        src_tag = tag[src]
        expected_src = src[np.argsort(F.asnumpy(src_tag))]
        actual_src = new_src[new_dst == nid]
        np.array_equal(expected_src, actual_src)


@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sorting by tag not implemented")
@parametrize_idtype
@pytest.mark.parametrize("tag_type", ['node', 'edge'])
def test_sort_with_tag(idtype, tag_type):
    num_nodes, num_adj, num_tags = 200, [20, 50], 5
    g = create_test_heterograph(num_nodes, num_adj, idtype=idtype)
    assert tag_type in ['node', 'edge']
    if tag_type == 'node':
        tag = F.tensor(np.random.choice(num_tags, g.number_of_nodes()))
    else:
        tag = F.tensor(np.random.choice(num_tags, g.num_edges()))

    # sort_csr_by_tag
    new_g = dgl.sort_csr_by_tag(g, tag, tag_type=tag_type)
    old_csr = g.adjacency_matrix(scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(scipy_fmt='csr')
    if tag_type == 'node':
        assert(check_sort(new_csr, tag, new_g.ndata["_TAG_OFFSET"]))
        assert(not check_sort(old_csr, tag))  # Check the original csr is not modified.
    else:
        check_sort_edge(old_csr, new_csr, tag, 'csr')

    # sort_csc_by_tag
    new_g = dgl.sort_csc_by_tag(g, tag, tag_type=tag_type)
    old_csc = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    if tag_type == 'node':
        assert(check_sort(new_csc, tag, new_g.ndata["_TAG_OFFSET"]))
        assert(not check_sort(old_csc, tag))
    else:
        check_sort_edge(old_csc, new_csc, tag, 'csc')


@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sorting by tag not implemented")
@parametrize_idtype
def test_sort_with_tag_bipartite(idtype):
    num_nodes, num_adj, num_tags = 200, [20, 50], 5
    g = create_test_heterograph(num_nodes, num_adj, idtype=idtype)
    g = dgl.heterograph({('_U', '_E', '_V') : g.edges()})
    utag = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_U')))
    vtag = F.tensor(np.random.choice(num_tags, g.number_of_nodes('_V')))

    new_g = dgl.sort_csr_by_tag(g, vtag)
    old_csr = g.adjacency_matrix(scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(new_csr, vtag, new_g.nodes['_U'].data['_TAG_OFFSET']))
    assert(not check_sort(old_csr, vtag))

    new_g = dgl.sort_csc_by_tag(g, utag)
    old_csc = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(new_csc, utag, new_g.nodes['_V'].data['_TAG_OFFSET']))
    assert(not check_sort(old_csc, utag))

if __name__ == "__main__":
    test_sort_with_tag(F.int32)
    test_sort_with_tag_bipartite(F.int32)
