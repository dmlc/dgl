import dgl
import dgl.backend as F
import networkx as nx
import numpy as np
import scipy as sp

N = 10000
a = sp.sparse.random(N, N, 1 / N, data_rvs=lambda n: np.ones(n))
b = sp.sparse.triu(a, 1) + sp.sparse.triu(a, 1).transpose()
g = dgl.DGLGraph()
g.from_scipy_sparse_matrix(b)

backtracking = True
lg_sparse = g.line_graph(backtracking)
lg_cpp = g._line_graph(backtracking)
assert lg_sparse.number_of_nodes() == lg_cpp.number_of_nodes()
assert lg_sparse.number_of_edges() == lg_cpp.number_of_edges()

src_sparse, dst_sparse, _ = lg_sparse.edges(sorted=True)
src_cpp, dst_cpp, _ = lg_cpp.edges(sorted=True)
assert (src_sparse == src_cpp).all()
assert (dst_sparse == dst_cpp).all()
