import dgl.graphbolt
import scipy.sparse as sp
import torch


def rand_csc_graph(N, density):
    adj = sp.random(N, N, density)
    adj = adj + adj.T
    adj = adj.tocsc()

    indptr = torch.LongTensor(adj.indptr)
    indices = torch.LongTensor(adj.indices)

    graph = dgl.graphbolt.from_csc(indptr, indices)

    return graph
