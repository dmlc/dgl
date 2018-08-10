from dgl.graph import DGLGraph
import torch as th

eps = 1e-5

g = DGLGraph()
g.add_edge(0, 1)
g.add_edge(0, 2)
n_repr = th.randn(3, 64)
g.set_n_repr(n_repr)
e_repr = th.randn(2, 64)
g.set_e_repr(e_repr)

sg = g.subgraph([0, 1])
assert th.max(th.abs(sg.get_n_repr() - n_repr[:2])) < eps
assert th.max(th.abs(sg.get_e_repr() - e_repr[:1])) < eps
