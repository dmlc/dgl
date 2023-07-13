import multiprocessing as mp
import pickle
import unittest

import backend as F
import dgl.graphbolt as gb
import scipy.sparse as sp
import torch


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_sampled_subgraph_serialization():
    ntypes = {"n1": 0, "n2": 1}
    etypes = {("n1", "e1", "n2"): 0, ("n2", "e2", "n1"): 1}
    metadata = gb.GraphMetadata(ntypes, etypes)
    num_nodes = 5
    num_edges = 9
    indptr = torch.LongTensor([0, 2, 4, 6, 7, 9])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
    reverse_column_node_ids = torch.LongTensor([2, 4, 6, 8, 10])
    node_type_offset = torch.LongTensor([0, 2, 5])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        metadata=metadata,
    )

    # Generate subgraph (ScriptObject) via sample neighbors.
    nodes = torch.LongTensor([torch.LongTensor([0]), torch.LongTensor([0])])
    fanouts = torch.tensor([-1, -1])
    subgraph = graph._sample_neighbors(nodes, fanouts)

    # Serialize and deserialize with pickle.
    serialized = pickle.dumps(subgraph)
    result = pickle.loads(serialized)

    # Verification.
    assert torch.equal(subgraph.indptr, result.indptr)
    assert torch.equal(subgraph.indices, result.indices)
    assert torch.equal(
        subgraph.reverse_column_node_ids,
        result.reverse_column_node_ids,
    )
    assert (
        subgraph.reverse_row_node_ids is None
        and subgraph.reverse_row_node_ids is None
    ) or torch.equal(subgraph.reverse_row_node_ids, result.reverse_row_node_ids)
    assert (
        subgraph.reverse_edge_ids is None and result.reverse_edge_ids is None
    ) or torch.equal(subgraph.reverse_edge_ids, result.reverse_edge_ids)
    assert (
        subgraph.type_per_edge is None and result.type_per_edge is None
    ) or torch.equal(subgraph.type_per_edge, result.type_per_edge)
