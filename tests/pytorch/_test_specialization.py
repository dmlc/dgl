import torch as th
from dgl.graph import DGLGraph

D = 32

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
    # TODO: use internal interface to set data.
    col = th.randn(10, D)
    g._node_frame['h'] = col
    return g

def test_spmv_specialize():
    g = generate_graph()
    g.register_message_func('from_src', batchable=True)
    g.register_reduce_func('sum', batchable=True)
    g.register_update_func(update_func, batchable=True)
    g.update_all()

if __name__ == '__main__':
    test_spmv_specialize()
