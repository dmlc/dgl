import networkx as nx
import dgl
import dgl.contrib as contrib
from dgl.frame import Frame, FrameRef, Column
from dgl.graph_index import create_graph_index
from dgl.utils import toindex
import backend as F
import dgl.function as fn
import pickle
import io

def _assert_is_identical(g, g2):
    assert g.is_multigraph == g2.is_multigraph
    assert g.is_readonly == g2.is_readonly
    assert g.number_of_nodes() == g2.number_of_nodes()
    src, dst = g.all_edges()
    src2, dst2 = g2.all_edges()
    assert F.array_equal(src, src2)
    assert F.array_equal(dst, dst2)

    assert len(g.ndata) == len(g2.ndata)
    assert len(g.edata) == len(g2.edata)
    for k in g.ndata:
        assert F.allclose(g.ndata[k], g2.ndata[k])
    for k in g.edata:
        assert F.allclose(g.edata[k], g2.edata[k])

def _assert_is_identical_nodeflow(nf1, nf2):
    assert nf1.is_multigraph == nf2.is_multigraph
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
    gi = create_graph_index(None, False, False)
    gi.add_nodes(3)
    src_idx = toindex([0, 0])
    dst_idx = toindex([1, 2])
    gi.add_edges(src_idx, dst_idx)

    gi2 = _reconstruct_pickle(gi)

    assert gi2.number_of_nodes() == gi.number_of_nodes()
    src_idx2, dst_idx2, _ = gi2.edges()
    assert F.array_equal(src_idx.tousertensor(), src_idx2.tousertensor())
    assert F.array_equal(dst_idx.tousertensor(), dst_idx2.tousertensor())


def test_pickling_frame():
    x = F.randn((3, 7))
    y = F.randn((3, 5))

    c = Column(x)

    c2 = _reconstruct_pickle(c)
    assert F.allclose(c.data, c2.data)

    fr = Frame({'x': x, 'y': y})

    fr2 = _reconstruct_pickle(fr)
    assert F.allclose(fr2['x'].data, x)
    assert F.allclose(fr2['y'].data, y)

    fr = Frame()


def _global_message_func(nodes):
    return {'x': nodes.data['x']}

def test_pickling_graph():
    # graph structures and frames are pickled
    g = dgl.DGLGraph()
    g.add_nodes(3)
    src = F.tensor([0, 0])
    dst = F.tensor([1, 2])
    g.add_edges(src, dst)

    x = F.randn((3, 7))
    y = F.randn((3, 5))
    a = F.randn((2, 6))
    b = F.randn((2, 4))

    g.ndata['x'] = x
    g.ndata['y'] = y
    g.edata['a'] = a
    g.edata['b'] = b

    # registered functions are pickled
    g.register_message_func(_global_message_func)
    reduce_func = fn.sum('x', 'x')
    g.register_reduce_func(reduce_func)

    # custom attributes should be pickled
    g.foo = 2

    new_g = _reconstruct_pickle(g)

    _assert_is_identical(g, new_g)
    assert new_g.foo == 2
    assert new_g._message_func == _global_message_func
    assert isinstance(new_g._reduce_func, type(reduce_func))
    assert new_g._reduce_func._name == 'sum'
    assert new_g._reduce_func.msg_field == 'x'
    assert new_g._reduce_func.out_field == 'x'

    # test batched graph with partial set case
    g2 = dgl.DGLGraph()
    g2.add_nodes(4)
    src2 = F.tensor([0, 1])
    dst2 = F.tensor([2, 3])
    g2.add_edges(src2, dst2)

    x2 = F.randn((4, 7))
    y2 = F.randn((3, 5))
    a2 = F.randn((2, 6))
    b2 = F.randn((2, 4))

    g2.ndata['x'] = x2
    g2.nodes[[0, 1, 3]].data['y'] = y2
    g2.edata['a'] = a2
    g2.edata['b'] = b2

    bg = dgl.batch([g, g2])

    bg2 = _reconstruct_pickle(bg)

    _assert_is_identical(bg, bg2)
    new_g, new_g2 = dgl.unbatch(bg2)
    _assert_is_identical(g, new_g)
    _assert_is_identical(g2, new_g2)

    # readonly graph
    g = dgl.DGLGraph([(0, 1), (1, 2)], readonly=True)
    new_g = _reconstruct_pickle(g)
    _assert_is_identical(g, new_g)

    # multigraph
    g = dgl.DGLGraph([(0, 1), (0, 1), (1, 2)], multigraph=True)
    new_g = _reconstruct_pickle(g)
    _assert_is_identical(g, new_g)

    # readonly multigraph
    g = dgl.DGLGraph([(0, 1), (0, 1), (1, 2)], multigraph=True, readonly=True)
    new_g = _reconstruct_pickle(g)
    _assert_is_identical(g, new_g)

def test_pickling_nodeflow():
    elist = [(0, 1), (1, 2), (2, 3), (3, 0)]
    g = dgl.DGLGraph(elist, readonly=True)
    g.ndata['x'] = F.randn((4, 5))
    g.edata['y'] = F.randn((4, 3))
    nf = contrib.sampling.sampler.create_full_nodeflow(g, 5)
    nf.copy_from_parent()  # add features
    new_nf = _reconstruct_pickle(nf)
    _assert_is_identical_nodeflow(nf, new_nf)

def test_pickling_batched_graph():
    glist = [nx.path_graph(i + 5) for i in range(5)]
    glist = [dgl.DGLGraph(g) for g in glist]
    bg = dgl.batch(glist)
    bg.ndata['x'] = F.randn((35, 5))
    bg.edata['y'] = F.randn((60, 3))
    new_bg = _reconstruct_pickle(bg)
    _assert_is_identical_batchedgraph(bg, new_bg)

if __name__ == '__main__':
    test_pickling_index()
    test_pickling_graph_index()
    test_pickling_frame()
    test_pickling_graph()
    test_pickling_nodeflow()
    test_pickling_batched_graph()
