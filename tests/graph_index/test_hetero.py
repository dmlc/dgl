import dgl
import dgl.graph_index as dgl_gidx

"""
Test with a heterograph of three ntypes and three etypes
meta graph:
0 -> 1
1 -> 2
2 -> 1

Num nodes per ntype:
0 : 5
1 : 2
2 : 3

rel graph:
0->1 : [0 -> 0, 1 -> 0, 2 -> 0, 3 -> 0]
1->2 : [0 -> 0, 1 -> 1, 1 -> 2]
2->1 : [0 -> 1, 1 -> 1, 2 -> 1]
"""

def rel1_from_coo():
    row = dgl.toindex([0, 1, 2, 3])
    col = dgl.toindex([0, 0, 0, 0])
    return dgl_gidx.create_bipartite_from_coo(5, 2, row, col)

def rel2_from_coo():
    row = dgl.toindex([0, 1, 1])
    col = dgl.toindex([0, 1, 2])
    return dgl_gidx.create_bipartite_from_coo(2, 3, row, col)

def rel3_from_coo():
    row = dgl.toindex([0, 1, 2])
    col = dgl.toindex([1, 1, 1])
    return dgl_gidx.create_bipartite_from_coo(3, 2, row, col)

def rel1_from_csr():
    indptr = dgl.toindex([0, 1, 2, 3, 4, 4])
    indices = dgl.toindex([0, 0, 0, 0])
    edge_ids = dgl.toindex([0, 1, 2, 3])
    return dgl_gidx.create_bipartite_from_csr(5, 2, indptr, indices, edge_ids)

def rel2_from_csr():
    row = dgl.toindex([0, 1, 3])
    col = dgl.toindex([0, 1, 2])
    edge_ids = dgl.toindex([0, 1, 2])
    return dgl_gidx.create_bipartite_from_csr(2, 3, indptr, indices, edge_ids)

def rel3_from_csr():
    row = dgl.toindex([0, 1, 2, 3])
    col = dgl.toindex([1, 1, 1])
    edge_ids = dgl.toindex([0, 1, 2])
    return dgl_gidx.create_bipartite_from_csr(3, 2, indptr, indices, edge_ids)

def gen_from_coo():
    mg = dgl_gidx.from_edge_list([(0, 1), (1, 2), (2, 1)], readonly=True)
    return dgl_gidx.create_heterograph(mg, [rel1_from_coo(), rel2_from_coo(), rel3_from_coo()])

def gen_from_csr():
    mg = dgl_gidx.from_edge_list([(0, 1), (1, 2), (2, 1)], readonly=True)
    return dgl_gidx.create_heterograph(mg, [rel1_from_csr(), rel2_from_csr(), rel3_from_csr()])
