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


def rand_hetero_csc_graph(N, density, metadata):
    adj = sp.random(N, N, density)
    adj = adj + adj.T
    adj = adj.tocsc()

    indptr = torch.LongTensor(adj.indptr)
    indices = torch.LongTensor(adj.indices)

    num_ntypes = len(metadata.node_type_to_id)
    num_etypes = len(metadata.edge_type_to_id)
    num_edges = indices.size(0)
    num_nodes_per_type = N // num_ntypes
    node_type_offset = []
    for i in range(num_ntypes):
        node_type_offset.append(num_nodes_per_type * i)
    node_type_offset.append(N)
    node_type_offset = torch.LongTensor(node_type_offset)
    type_per_edge = torch.randint(0, num_etypes, (num_edges,))
    return dgl.graphbolt.from_csc(
        indptr, indices, node_type_offset, type_per_edge, metadata
    )
