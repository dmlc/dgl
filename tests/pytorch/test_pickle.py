import dgl
from dgl.frame import Frame, FrameRef, Column
from dgl.graph_index import create_graph_index
from dgl.utils import toindex
import dgl.backend as backend
import dgl.function as F
import utils as U
import torch
import pickle
import io

def _reconstruct_pickle(obj):
    f = io.BytesIO()
    pickle.dump(obj, f)
    f.seek(0)
    obj = pickle.load(f)
    f.close()

    return obj

def test_pickling_index():
    i = toindex([1, 2, 3])
    i.tousertensor()
    i.todgltensor() # construct a dgl tensor which is unpicklable

    i2 = _reconstruct_pickle(i)

    assert torch.equal(i2.tousertensor(), i.tousertensor())


def test_pickling_graph_index():
    gi = create_graph_index()
    gi.add_nodes(3)
    src_idx = toindex([0, 0])
    dst_idx = toindex([1, 2])
    gi.add_edges(src_idx, dst_idx)

    gi2 = _reconstruct_pickle(gi)

    assert gi2.number_of_nodes() == gi.number_of_nodes()
    src_idx2, dst_idx2, _ = gi2.edges()
    assert torch.equal(src_idx.tousertensor(), src_idx2.tousertensor())
    assert torch.equal(dst_idx.tousertensor(), dst_idx2.tousertensor())


def test_pickling_frame():
    x = torch.randn(3, 7)
    y = torch.randn(3, 5)

    c = Column(x)

    c2 = _reconstruct_pickle(c)
    assert U.allclose(c.data, c2.data)

    fr = Frame({'x': x, 'y': y})

    fr2 = _reconstruct_pickle(fr)
    assert U.allclose(fr2['x'].data, x)
    assert U.allclose(fr2['y'].data, y)

    fr = Frame()


def _assert_is_identical(g, g2):
    assert g.number_of_nodes() == g2.number_of_nodes()
    src, dst = g.all_edges()
    src2, dst2 = g2.all_edges()
    assert torch.equal(src, src2)
    assert torch.equal(dst, dst2)

    assert len(g.ndata) == len(g2.ndata)
    assert len(g.edata) == len(g2.edata)
    for k in g.ndata:
        assert U.allclose(g.ndata[k], g2.ndata[k])
    for k in g.edata:
        assert U.allclose(g.edata[k], g2.edata[k])


def _global_message_func(nodes):
    return {'x': nodes.data['x']}

def test_pickling_graph():
    # graph structures and frames are pickled
    g = dgl.DGLGraph()
    g.add_nodes(3)
    src = torch.LongTensor([0, 0])
    dst = torch.LongTensor([1, 2])
    g.add_edges(src, dst)

    x = torch.randn(3, 7)
    y = torch.randn(3, 5)
    a = torch.randn(2, 6)
    b = torch.randn(2, 4)

    g.ndata['x'] = x
    g.ndata['y'] = y
    g.edata['a'] = a
    g.edata['b'] = b

    # registered functions are pickled
    g.register_message_func(_global_message_func)
    reduce_func = F.sum('x', 'x')
    g.register_reduce_func(reduce_func)

    # custom attributes should be pickled
    g.foo = 2

    new_g = _reconstruct_pickle(g)

    _assert_is_identical(g, new_g)
    assert new_g.foo == 2
    assert new_g._message_func == _global_message_func
    assert isinstance(new_g._reduce_func, type(reduce_func))
    assert new_g._reduce_func._name == 'sum'
    assert new_g._reduce_func.op == backend.sum
    assert new_g._reduce_func.msg_field == 'x'
    assert new_g._reduce_func.out_field == 'x'

    # test batched graph with partial set case
    g2 = dgl.DGLGraph()
    g2.add_nodes(4)
    src2 = torch.LongTensor([0, 1])
    dst2 = torch.LongTensor([2, 3])
    g2.add_edges(src2, dst2)

    x2 = torch.randn(4, 7)
    y2 = torch.randn(3, 5)
    a2 = torch.randn(2, 6)
    b2 = torch.randn(2, 4)

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


if __name__ == '__main__':
    test_pickling_index()
    test_pickling_graph_index()
    test_pickling_frame()
    test_pickling_graph()
