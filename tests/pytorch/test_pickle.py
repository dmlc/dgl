import dgl
from dgl.graph_index import create_graph_index
from dgl.utils import toindex
import utils as U
import torch
import pickle
import io

def test_pickling_index():
    i = toindex([1, 2, 3])
    i.tousertensor()
    i.todgltensor() # construct a dgl tensor which is unpicklable

    f = io.BytesIO()
    pickle.dump(i, f)
    f.seek(0)
    i2 = pickle.load(f)
    f.close()

    assert torch.equal(i2.tousertensor(), i.tousertensor())


def test_pickling_graph_index():
    gi = create_graph_index()
    gi.add_nodes(3)
    src_idx = toindex([0, 0])
    dst_idx = toindex([1, 2])
    gi.add_edges(src_idx, dst_idx)

    f = io.BytesIO()
    pickle.dump(gi, f)
    f.seek(0)
    gi2 = pickle.load(f)
    f.close()

    assert gi2.number_of_nodes() == gi.number_of_nodes()
    src_idx2, dst_idx2, _ = gi2.edges()
    assert torch.equal(src_idx.tousertensor(), src_idx2.tousertensor())
    assert torch.equal(dst_idx.tousertensor(), dst_idx2.tousertensor())


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

    # registered functions are not pickled
    g.register_message_func(lambda nodes: {'x': nodes.data['x']})

    f = io.BytesIO()
    pickle.dump(g, f)
    f.seek(0)
    g2 = pickle.load(f)
    f.close()

    assert g.number_of_nodes() == g2.number_of_nodes()
    src2, dst2 = g2.all_edges()
    assert torch.equal(src, src2)
    assert torch.equal(dst, dst2)

    for k in g.ndata:
        assert U.allclose(g.ndata[k], g2.ndata[k])
    for k in g.edata:
        assert U.allclose(g.edata[k], g2.edata[k])


if __name__ == '__main__':
    test_pickling_index()
    test_pickling_graph_index()
    test_pickling_graph()
