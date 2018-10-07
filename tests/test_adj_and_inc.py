import dgl
import dgl.backend as F
import networkx as nx
import numpy as np
import scipy as sp

N = 1000
a = sp.sparse.random(N, N, 1 / N, data_rvs=lambda n: np.ones(n))
b = sp.sparse.triu(a) + sp.sparse.triu(a, 1).transpose()
g_nx = nx.from_scipy_sparse_matrix(b, create_using=nx.DiGraph())
g_dgl = dgl.DGLGraph()
g_dgl.from_scipy_sparse_matrix(b)

h_nx = g_dgl.to_networkx()
g_nodes = set(g_nx.nodes)
h_nodes = set(h_nx.nodes)
assert h_nodes.issubset(g_nodes)
assert all(g_nx.in_degree(x) == g_nx.out_degree(x) == 0
           for x in g_nodes.difference(h_nodes))
assert g_nx.edges == h_nx.edges

nx_adj = nx.adjacency_matrix(g_nx)
nx_inc = nx.incidence_matrix(g_nx, edgelist=sorted(g_nx.edges()))
nx_oriented = nx.incidence_matrix(g_nx, edgelist=sorted(g_nx.edges()), oriented=True)
ctx = F.get_context(F.ones((1,)))
dgl_adj = F.to_scipy_sparse(g_dgl.adjacency_matrix().get(ctx)).transpose()
dgl_inc = F.to_scipy_sparse(g_dgl.incidence_matrix().get(ctx))
dgl_oriented = F.to_scipy_sparse(g_dgl.incidence_matrix(oriented=True).get(ctx))

assert abs(nx_adj - dgl_adj).max() == 0
assert abs(nx_inc - dgl_inc).max() == 0
assert abs(nx_oriented - dgl_oriented).max() == 0
