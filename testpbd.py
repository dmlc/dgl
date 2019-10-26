
import torch as th
import dgl
from build import dglpybind
import dgl.backend as F

# Compatible way of using pybind11 with current ffi
new_g = dglpybind.new_gindex()
print(new_g)
# def main():
g = dgl.DGLGraph()
g.add_nodes(10)
g.add_edge(1, 2)
th_src = th.tensor([1, 2])
th_dst = th.tensor([2, 3])
nd_src = F.zerocopy_to_dgl_ndarray(th_src)
nd_dst = F.zerocopy_to_dgl_ndarray(th_dst)
c = dglpybind.HasEdgesBetween(g._graph, nd_src, nd_dst)
print(c)

# Pure Pybind11 way
g = dglpybind.pure.MutableGraph()
g.add_nodes(10)
g.add_edges(nd_src, nd_dst)
earray = g.edges()
print(earray.src)