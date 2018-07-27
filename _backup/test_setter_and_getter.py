import numpy as np
import numpy.random as npr
import torch as th
import dgl.graph as G

M = 100
N = 10
D = 1024
epsilon = 1e-10

g = G.DGLGraph()

# node setter/getter
g.add_nodes_from(range(M))

g.init_node_repr(th.zeros(M, N)) # init_node_repr uses set_node_repr
assert th.max(th.abs(g.get_node_repr(list(g.nodes)))) < epsilon

# edge setter/getter
g.add_edges_from((i, i) for i in range(M))

g.init_edge_repr(th.zeros(M, N)) # init_edge_repr uses set_edge_repr
u, v = map(list, zip(*g.edges))
assert th.max(th.abs(g.get_edge_repr(u, v))) < epsilon
