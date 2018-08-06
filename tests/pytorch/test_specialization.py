import torch as th
import numpy as np
from dgl.graph import DGLGraph

D = 5

def check_eq(a, b):
    if not np.allclose(a.numpy(), b.numpy()):
        print(a, b)

def message_func(hu, edge):
    return hu

def reduce_func(hv, msgs):
    return th.sum(msgs, 1)

def update_func(hv, accum):
    assert hv.shape == accum.shape
    return hv + accum

def generate_graph():
    g = DGLGraph()
    for i in range(10):
        g.add_node(i) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    col = th.randn(10, D)
    g.set_n_repr(col)
    return g

def test_spmv_specialize():
    g = generate_graph()
    g.register_message_func('from_src', batchable=True)
    g.register_reduce_func('sum', batchable=True)
    g.register_update_func(update_func, batchable=True)
    v1 = g.get_n_repr()
    g.update_all()
    v2 = g.get_n_repr()
    g.set_n_repr(v1)
    g.register_message_func(message_func, batchable=True)
    g.register_reduce_func(reduce_func, batchable=True)
    g.update_all()
    v3 = g.get_n_repr()
    check_eq(v2, v3)

if __name__ == '__main__':
    test_spmv_specialize()
