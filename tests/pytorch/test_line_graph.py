import torch as th
import networkx as nx
import numpy as np
import dgl

D = 5

def check_eq(a, b):
    return a.shape == b.shape and np.allclose(a.numpy(), b.numpy())

def test_line_graph():
    N = 5
    G = dgl.DGLGraph(nx.star_graph(N))
    G.set_e_repr(th.randn((2*N, D)))
    L = dgl.line_graph(G)
    print(L.get_n_repr())
    # update node features on line graph should reflect to edge features on
    # original graph.
    u = [0, 0, 2, 3]
    v = [1, 2, 0, 0]
    eid = G.get_edge_id(u, v)
    L.set_n_repr(th.zeros((4, D)), eid)
    assert check_eq(G.get_e_repr(u, v), th.zeros((4, D)))

if __name__ == '__main__':
    test_line_graph()
