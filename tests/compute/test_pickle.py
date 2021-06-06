import networkx as nx
import scipy.sparse as ssp
import dgl
import dgl.contrib as contrib
from dgl.graph_index import create_graph_index
from dgl.utils import toindex
import backend as F
import dgl.function as fn
import pickle
import io
import unittest, pytest
import test_utils
from test_utils import parametrize_dtype, get_cases

def _assert_is_identical(g, g2):
    assert g.is_readonly == g2.is_readonly
    assert g.number_of_nodes() == g2.number_of_nodes()
    src, dst = g.all_edges(order='eid')
    src2, dst2 = g2.all_edges(order='eid')
    assert F.array_equal(src, src2)
    assert F.array_equal(dst, dst2)

    assert len(g.ndata) == len(g2.ndata)
    assert len(g.edata) == len(g2.edata)
    for k in g.ndata:
        assert F.allclose(g.ndata[k], g2.ndata[k])
    for k in g.edata:
        assert F.allclose(g.edata[k], g2.edata[k])

def _assert_is_identical_hetero(g, g2):
    assert g.is_readonly == g2.is_readonly
    assert g.ntypes == g2.ntypes
    assert g.canonical_etypes == g2.canonical_etypes

    # check if two metagraphs are identical
    for edges, features in g.metagraph().edges(keys=True).items():
        assert g2.metagraph().edges(keys=True)[edges] == features

    # check if node ID spaces and feature spaces are equal
    for ntype in g.ntypes:
        assert g.number_of_nodes(ntype) == g2.number_of_nodes(ntype)
        assert len(g.nodes[ntype].data) == len(g2.nodes[ntype].data)
        for k in g.nodes[ntype].data:
            assert F.allclose(g.nodes[ntype].data[k], g2.nodes[ntype].data[k])

    # check if edge ID spaces and feature spaces are equal
    for etype in g.canonical_etypes:
        src, dst = g.all_edges(etype=etype, order='eid')
        src2, dst2 = g2.all_edges(etype=etype, order='eid')
        assert F.array_equal(src, src2)
        assert F.array_equal(dst, dst2)
        for k in g.edges[etype].data:
            assert F.allclose(g.edges[etype].data[k], g2.edges[etype].data[k])

def _assert_is_identical_nodeflow(nf1, nf2):
    assert nf1.is_readonly == nf2.is_readonly
    assert nf1.number_of_nodes() == nf2.number_of_nodes()
    src, dst = nf1.all_edges()
    src2, dst2 = nf2.all_edges()
    assert F.array_equal(src, src2)
    assert F.array_equal(dst, dst2)

    assert nf1.num_layers == nf2.num_layers
    for i in range(nf1.num_layers):
        assert nf1.layer_size(i) == nf2.layer_size(i)
        assert nf1.layers[i].data.keys() == nf2.layers[i].data.keys()
        for k in nf1.layers[i].data:
            assert F.allclose(nf1.layers[i].data[k], nf2.layers[i].data[k])
    assert nf1.num_blocks == nf2.num_blocks
    for i in range(nf1.num_blocks):
        assert nf1.block_size(i) == nf2.block_size(i)
        assert nf1.blocks[i].data.keys() == nf2.blocks[i].data.keys()
        for k in nf1.blocks[i].data:
            assert F.allclose(nf1.blocks[i].data[k], nf2.blocks[i].data[k])

def _assert_is_identical_batchedgraph(bg1, bg2):
    _assert_is_identical(bg1, bg2)
    assert bg1.batch_size == bg2.batch_size
    assert bg1.batch_num_nodes == bg2.batch_num_nodes
    assert bg1.batch_num_edges == bg2.batch_num_edges

def _assert_is_identical_batchedhetero(bg1, bg2):
    _assert_is_identical_hetero(bg1, bg2)
    for ntype in bg1.ntypes:
        assert bg1.batch_num_nodes(ntype) == bg2.batch_num_nodes(ntype)
    for canonical_etype in bg1.canonical_etypes:
        assert bg1.batch_num_edges(canonical_etype) == bg2.batch_num_edges(canonical_etype)

def _assert_is_identical_index(i1, i2):
    assert i1.slice_data() == i2.slice_data()
    assert F.array_equal(i1.tousertensor(), i2.tousertensor())

def _reconstruct_pickle(obj):
    f = io.BytesIO()
    pickle.dump(obj, f)
    f.seek(0)
    obj = pickle.load(f)
    f.close()

    return obj

def test_pickling_index():
    # normal index
    i = toindex([1, 2, 3])
    i.tousertensor()
    i.todgltensor() # construct a dgl tensor which is unpicklable
    i2 = _reconstruct_pickle(i)
    _assert_is_identical_index(i, i2)

    # slice index
    i = toindex(slice(5, 10))
    i2 = _reconstruct_pickle(i)
    _assert_is_identical_index(i, i2)

def test_pickling_graph_index():
    gi = create_graph_index(None, False)
    gi.add_nodes(3)
    src_idx = toindex([0, 0])
    dst_idx = toindex([1, 2])
    gi.add_edges(src_idx, dst_idx)

    gi2 = _reconstruct_pickle(gi)

    assert gi2.number_of_nodes() == gi.number_of_nodes()
    src_idx2, dst_idx2, _ = gi2.edges()
    assert F.array_equal(src_idx.tousertensor(), src_idx2.tousertensor())
    assert F.array_equal(dst_idx.tousertensor(), dst_idx2.tousertensor())


def _global_message_func(nodes):
    return {'x': nodes.data['x']}

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU not implemented")
@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(exclude=['dglgraph', 'two_hetero_batch']))
def test_pickling_graph(g, idtype):
    g = g.astype(idtype)
    new_g = _reconstruct_pickle(g)
    test_utils.check_graph_equal(g, new_g, check_feature=True)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU not implemented")
def test_pickling_batched_heterograph():
    # copied from test_heterograph.create_test_heterograph()
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    })
    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    })

    g.nodes['user'].data['u_h'] = F.randn((3, 4))
    g.nodes['game'].data['g_h'] = F.randn((2, 5))
    g.edges['plays'].data['p_h'] = F.randn((4, 6))
    g2.nodes['user'].data['u_h'] = F.randn((3, 4))
    g2.nodes['game'].data['g_h'] = F.randn((2, 5))
    g2.edges['plays'].data['p_h'] = F.randn((4, 6))

    bg = dgl.batch_hetero([g, g2])
    new_bg = _reconstruct_pickle(bg)
    test_utils.check_graph_equal(bg, new_bg)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU edge_subgraph w/ relabeling not implemented")
def test_pickling_subgraph():
    f1 = io.BytesIO()
    f2 = io.BytesIO()
    g = dgl.rand_graph(10000, 100000)
    g.ndata['x'] = F.randn((10000, 4))
    g.edata['x'] = F.randn((100000, 5))
    pickle.dump(g, f1)
    sg = g.subgraph([0, 1])
    sgx = sg.ndata['x'] # materialize
    pickle.dump(sg, f2)
    # TODO(BarclayII): How should I test that the size of the subgraph pickle file should not
    # be as large as the size of the original pickle file?
    assert f1.tell() > f2.tell() * 50

    f2.seek(0)
    f2.truncate()
    sgx = sg.edata['x'] # materialize
    pickle.dump(sg, f2)
    assert f1.tell() > f2.tell() * 50

    f2.seek(0)
    f2.truncate()
    sg = g.edge_subgraph([0])
    sgx = sg.edata['x'] # materialize
    pickle.dump(sg, f2)
    assert f1.tell() > f2.tell() * 50

    f2.seek(0)
    f2.truncate()
    sgx = sg.ndata['x'] # materialize
    pickle.dump(sg, f2)
    assert f1.tell() > f2.tell() * 50

    f1.close()
    f2.close()

if __name__ == '__main__':
    test_pickling_index()
    test_pickling_graph_index()
    test_pickling_frame()
    test_pickling_graph()
    test_pickling_nodeflow()
    test_pickling_batched_graph()
    test_pickling_heterograph()
    test_pickling_batched_heterograph()
