import dgl.graphbolt as gb
import scipy.sparse as sp
import torch


def rand_csc_graph(N, density):
    adj = sp.random(N, N, density)
    adj = adj + adj.T
    adj = adj.tocsc()

    indptr = torch.LongTensor(adj.indptr)
    indices = torch.LongTensor(adj.indices)

    graph = gb.from_csc(indptr, indices)

    return graph


def random_homo_graph(num_nodes, num_edges):
    csc_indptr = torch.randint(0, num_edges, (num_nodes + 1,))
    csc_indptr = torch.sort(csc_indptr)[0]
    csc_indptr[0] = 0
    csc_indptr[-1] = num_edges
    indices = torch.randint(0, num_nodes, (num_edges,))
    return csc_indptr, indices


def get_metadata(num_ntypes, num_etypes):
    ntypes = {f"n{i}": i for i in range(num_ntypes)}
    etypes = {}
    count = 0
    for n1 in range(num_ntypes):
        for n2 in range(n1, num_ntypes):
            if count >= num_etypes:
                break
            etypes.update({(f"n{n1}", f"e{count}", f"n{n2}"): count})
            count += 1
    return gb.GraphMetadata(ntypes, etypes)


def random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    metadata = get_metadata(num_ntypes, num_etypes)
    # Randomly get node type split point.
    node_type_offset = torch.sort(
        torch.randint(0, num_nodes, (num_ntypes + 1,))
    )[0]
    node_type_offset[0] = 0
    node_type_offset[-1] = num_nodes

    type_per_edge = []
    for i in range(num_nodes):
        num = csc_indptr[i + 1] - csc_indptr[i]
        type_per_edge.append(
            torch.sort(torch.randint(0, num_etypes, (num,)))[0]
        )
    type_per_edge = torch.cat(type_per_edge, dim=0)
    return (csc_indptr, indices, node_type_offset, type_per_edge, metadata)
