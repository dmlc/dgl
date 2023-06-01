import torch
import dgl.graphbolt as gb

torch.manual_seed(3407)


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

def random_homo_graph(num_nodes, num_edges):
    csc_indptr = torch.randint(0, num_edges, (num_nodes + 1,))
    csc_indptr = torch.sort(csc_indptr)[0]
    csc_indptr[0] = 0
    csc_indptr[-1] = num_edges
    indices = torch.randint(0, num_nodes, (num_edges,))
    return csc_indptr, indices


def random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    metadata = get_metadata(num_ntypes, num_etypes)
    # random get node type split point
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


def csc_to_coo(csc_indptr, indices):
    node_num = csc_indptr[1:] - csc_indptr[:-1]
    col = torch.nonzero(node_num).squeeze()
    col = torch.repeat_interleave(col, node_num[col])
    return torch.stack([indices, col])


def random_graph_with_fixed_neighbors(num_nodes, neighbors_per_etype, num_ntypes, num_etypes):
    neighbors_per_node = num_etypes * neighbors_per_etype
    csc_indptr = torch.arange((num_nodes+1)) * neighbors_per_node
    num_edges = num_nodes * neighbors_per_node
    indices = torch.randint(0, num_nodes, (num_edges,))
    type_per_edge = torch.arange(num_etypes).repeat_interleave(neighbors_per_etype).repeat(num_nodes)
    node_type_offset = torch.sort(
        torch.randint(0, num_nodes, (num_ntypes + 1,))
    )[0]
    node_type_offset[0] = 0
    node_type_offset[-1] = num_nodes
    metadata = get_metadata(num_ntypes, num_etypes)
    return gb.from_csc(csc_indptr, indices, node_type_offset, type_per_edge, metadata)
    