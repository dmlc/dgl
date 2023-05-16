from dgl.dataloading2 import *
import pytest
import torch


def random_heterogeneous_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    row = torch.randint(1, num_nodes, (num_edges,))
    col = torch.randint(1, num_nodes, (num_edges,))
    ntypes = [f"n{i}" for i in range(num_ntypes)]
    etypes = [f"e{i}" for i in range(num_etypes)]
    type_per_edge = torch.randint(0, num_etypes, (num_edges,))
    # random get node type split point
    node_type_offset = torch.sort(torch.randperm(num_nodes)[:num_ntypes])[0]
    node_type_offset[0] = 0
    return (
        from_coo(
            row,
            col,
            hetero_info=(ntypes, etypes, node_type_offset, type_per_edge),
        ),
        row,
        col,
        type_per_edge,
    )


@pytest.mark.parametrize(
    "num_nodes,num_edges", [(10, 20), (50, 300), (1000, 500000)]
)
def test_from_coo(num_nodes, num_edges, num_ntypes=3, num_etypes=5):
    graph, orig_row, orig_col, orig_per_edge_type = random_heterogeneous_graph(
        num_nodes, num_edges, num_ntypes, num_etypes
    )

    csc_indptr = graph.csc_indptr
    indices = graph.indices
    per_edge_type = graph.per_edge_type

    row = csc_indptr[1:] - csc_indptr[:-1]
    row_indices = torch.nonzero(row).squeeze()
    row = row_indices.repeat_interleave(row[row_indices]) + 1
    col = indices

    assert row.size() == orig_row.size()
    assert col.size() == orig_col.size()
    assert per_edge_type.size() == orig_per_edge_type.size()
    assert torch.equal(torch.sort(orig_row).values, torch.sort(row).values)
    assert torch.equal(torch.sort(orig_col).values, torch.sort(col).values)
    assert torch.equal(
        torch.sort(orig_per_edge_type).values, torch.sort(per_edge_type).values
    )


if __name__ == "__main__":
    test_from_coo(1000, 100000)
