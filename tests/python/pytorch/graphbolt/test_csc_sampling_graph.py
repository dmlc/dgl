import os
import tempfile
import unittest

import backend as F

import dgl.graphbolt as gb

import pytest
import torch

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


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [0, 1, 10, 100, 1000])
def test_empty_graph(num_nodes):
    csc_indptr = torch.zeros((num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    graph = gb.from_csc(csc_indptr, indices)
    assert graph.num_edges == 0
    assert graph.num_nodes == num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [0, 1, 10, 100, 1000])
def test_hetero_empty_graph(num_nodes):
    csc_indptr = torch.zeros((num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    metadata = get_metadata(num_ntypes=3, num_etypes=5)
    # Some node types have no nodes.
    if num_nodes == 0:
        node_type_offset = torch.zeros((4,), dtype=int)
    else:
        node_type_offset = torch.sort(torch.randint(0, num_nodes, (4,)))[0]
        node_type_offset[0] = 0
        node_type_offset[-1] = num_nodes
    type_per_edge = torch.tensor([])
    graph = gb.from_csc(
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    )
    assert graph.num_edges == 0
    assert graph.num_nodes == num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)
    assert graph.metadata.node_type_to_id == metadata.node_type_to_id
    assert graph.metadata.edge_type_to_id == metadata.edge_type_to_id
    assert torch.equal(graph.node_type_offset, node_type_offset)
    assert torch.equal(graph.type_per_edge, type_per_edge)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "ntypes", [{"n1": 1, "n2": 1}, {5: 1, "n2": 2}, {"n1": 1.5, "n2": 2.0}]
)
def test_metadata_with_ntype_exception(ntypes):
    with pytest.raises(Exception):
        gb.GraphMetadata(ntypes, {("n1", "e1", "n2"): 1})


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "etypes",
    [
        {("n1", 5, "n12"): 1},
        {"e1": 1},
        {("n1", "e1"): 1},
        {("n1", "e1", 10): 1},
        {("n1", "e1", "n2"): 1, ("n1", "e2", "n3"): 1},
        {("n1", "e1", "n10"): 1},
        {("n1", "e1", "n2"): 1.5},
    ],
)
def test_metadata_with_etype_exception(etypes):
    with pytest.raises(Exception):
        gb.GraphMetadata({"n1": 0, "n2": 1, "n3": 2}, etypes)


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


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
def test_homo_graph(num_nodes, num_edges):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    graph = gb.from_csc(csc_indptr, indices)

    assert graph.num_nodes == num_nodes
    assert graph.num_edges == num_edges

    assert torch.equal(csc_indptr, graph.csc_indptr)
    assert torch.equal(indices, graph.indices)

    assert graph.metadata is None
    assert graph.node_type_offset is None
    assert graph.type_per_edge is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
@pytest.mark.parametrize("num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1)])
def test_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    ) = random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes)
    graph = gb.from_csc(
        csc_indptr, indices, node_type_offset, type_per_edge, metadata
    )

    assert graph.num_nodes == num_nodes
    assert graph.num_edges == num_edges

    assert torch.equal(csc_indptr, graph.csc_indptr)
    assert torch.equal(indices, graph.indices)
    assert torch.equal(node_type_offset, graph.node_type_offset)
    assert torch.equal(type_per_edge, graph.type_per_edge)
    assert metadata.node_type_to_id == graph.metadata.node_type_to_id
    assert metadata.edge_type_to_id == graph.metadata.edge_type_to_id


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "node_type_offset",
    [
        torch.tensor([0, 1]),
        torch.tensor([0, 1, 5, 6, 10]),
        torch.tensor([0, 1, 10]),
    ],
)
def test_node_type_offset_wrong_legnth(node_type_offset):
    num_ntypes = 3
    csc_indptr, indices, _, type_per_edge, metadata = random_hetero_graph(
        10, 50, num_ntypes, 5
    )
    with pytest.raises(Exception):
        gb.from_csc(
            csc_indptr, indices, node_type_offset, type_per_edge, metadata
        )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
def test_load_save_homo_graph(num_nodes, num_edges):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    graph = gb.from_csc(csc_indptr, indices)

    with tempfile.TemporaryDirectory() as test_dir:
        filename = os.path.join(test_dir, "csc_sampling_graph.tar")
        gb.save_csc_sampling_graph(graph, filename)
        graph2 = gb.load_csc_sampling_graph(filename)

    assert graph.num_nodes == graph2.num_nodes
    assert graph.num_edges == graph2.num_edges

    assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
    assert torch.equal(graph.indices, graph2.indices)

    assert graph.metadata is None and graph2.metadata is None
    assert graph.node_type_offset is None and graph2.node_type_offset is None
    assert graph.type_per_edge is None and graph2.type_per_edge is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
@pytest.mark.parametrize("num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1)])
def test_load_save_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    ) = random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes)
    graph = gb.from_csc(
        csc_indptr, indices, node_type_offset, type_per_edge, metadata
    )

    with tempfile.TemporaryDirectory() as test_dir:
        filename = os.path.join(test_dir, "csc_sampling_graph.tar")
        gb.save_csc_sampling_graph(graph, filename)
        graph2 = gb.load_csc_sampling_graph(filename)

    assert graph.num_nodes == graph2.num_nodes
    assert graph.num_edges == graph2.num_edges

    assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
    assert torch.equal(graph.indices, graph2.indices)
    assert torch.equal(graph.node_type_offset, graph2.node_type_offset)
    assert torch.equal(graph.type_per_edge, graph2.type_per_edge)
    assert graph.metadata.node_type_to_id == graph2.metadata.node_type_to_id
    assert graph.metadata.edge_type_to_id == graph2.metadata.edge_type_to_id


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_in_subgraph_homogeneous():
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    num_nodes = 5
    num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(indptr, indices)

    # Extract in subgraph.
    nodes = torch.LongTensor([1, 3, 4])
    in_subgraph = graph.in_subgraph(nodes)

    # Verify in subgraph.
    assert torch.equal(in_subgraph.indptr, torch.LongTensor([0, 2, 4, 7]))
    assert torch.equal(
        in_subgraph.indices, torch.LongTensor([2, 3, 1, 2, 0, 3, 4])
    )
    assert torch.equal(in_subgraph.reverse_column_node_ids, nodes)
    assert torch.equal(
        in_subgraph.reverse_row_node_ids, torch.arange(0, num_nodes)
    )
    assert torch.equal(
        in_subgraph.reverse_edge_ids, torch.LongTensor([3, 4, 7, 8, 9, 10, 11])
    )
    assert in_subgraph.type_per_edge is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_in_subgraph_heterogeneous():
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1

    node_type_0: [0, 1]
    node_type_1: [2, 3, 4]
    edge_type_0: node_type_0 -> node_type_0
    edge_type_1: node_type_0 -> node_type_1
    edge_type_2: node_type_1 -> node_type_0
    edge_type_3: node_type_1 -> node_type_1
    """
    # Initialize data.
    num_nodes = 5
    num_edges = 12
    ntypes = {
        "N0": 0,
        "N1": 1,
    }
    etypes = {
        ("N0", "R0", "N0"): 0,
        ("N0", "R1", "N1"): 1,
        ("N1", "R2", "N0"): 2,
        ("N1", "R3", "N1"): 3,
    }
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    node_type_offset = torch.LongTensor([0, 2, 5])
    type_per_edge = torch.LongTensor([0, 0, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)
    assert node_type_offset[-1] == num_nodes
    assert all(type_per_edge < len(etypes))

    # Construct CSCSamplingGraph.
    metadata = gb.GraphMetadata(ntypes, etypes)
    graph = gb.from_csc(
        indptr, indices, node_type_offset, type_per_edge, metadata
    )

    # Extract in subgraph.
    nodes = torch.LongTensor([1, 3, 4])
    in_subgraph = graph.in_subgraph(nodes)

    # Verify in subgraph.
    assert torch.equal(in_subgraph.indptr, torch.LongTensor([0, 2, 4, 7]))
    assert torch.equal(
        in_subgraph.indices, torch.LongTensor([2, 3, 1, 2, 0, 3, 4])
    )
    assert torch.equal(in_subgraph.reverse_column_node_ids, nodes)
    assert torch.equal(
        in_subgraph.reverse_row_node_ids, torch.arange(0, num_nodes)
    )
    assert torch.equal(
        in_subgraph.reverse_edge_ids, torch.LongTensor([3, 4, 7, 8, 9, 10, 11])
    )
    assert torch.equal(
        in_subgraph.type_per_edge, torch.LongTensor([2, 2, 1, 3, 1, 3, 3])
    )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_sample_neighbors():
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    num_nodes = 5
    num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    type_per_edge = torch.LongTensor([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(indptr, indices, type_per_edge=type_per_edge)

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4])
    fanouts = torch.tensor([2, 2])
    subgraph = graph.sample_neighbors(nodes, fanouts, return_eids=True)

    # Verify in subgraph.
    assert torch.equal(subgraph.indptr, torch.LongTensor([0, 2, 4, 7]))
    assert torch.equal(
        torch.sort(subgraph.indices)[0],
        torch.sort(torch.LongTensor([2, 3, 1, 2, 0, 3, 4]))[0],
    )
    assert torch.equal(subgraph.reverse_column_node_ids, nodes)
    assert torch.equal(
        subgraph.reverse_edge_ids, torch.LongTensor([3, 4, 7, 8, 9, 10, 11])
    )
    assert torch.equal(
        subgraph.type_per_edge, torch.LongTensor([0, 1, 0, 1, 0, 0, 1])
    )
    assert subgraph.reverse_row_node_ids is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "fanouts, expected_sampled_num",
    [
        ([0], 0),
        ([1], 3),
        ([2], 6),
        ([4], 7),
        ([-1], 7),
        ([0, 0], 0),
        ([1, 0], 3),
        ([1, 1], 6),
        ([2, 2], 7),
        ([-1, -1], 7),
    ],
)
def test_sample_neighbors_fanouts(fanouts, expected_sampled_num):
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    num_nodes = 5
    num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    type_per_edge = torch.LongTensor([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(indptr, indices, type_per_edge=type_per_edge)

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4])
    fanouts = torch.LongTensor(fanouts)
    subgraph = graph.sample_neighbors(nodes, fanouts)

    # Verify in subgraph.
    sampled_num = subgraph.indices.size(0)
    assert sampled_num == expected_sampled_num


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "replace, expected_sampled_num", [(False, 7), (True, 12)]
)
def test_sample_neighbors_replace(replace, expected_sampled_num):
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    num_nodes = 5
    num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(indptr, indices)

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4])
    subgraph = graph.sample_neighbors(
        nodes, fanouts=torch.LongTensor([4]), replace=replace
    )

    # Verify in subgraph.
    sampled_num = subgraph.indices.size(0)
    assert sampled_num == expected_sampled_num


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "probs_or_mask",
    [
        torch.tensor([2.5, 0, 8.4, 0, 0.4, 1.2, 2.5, 0, 8.4, 0.5, 0.4, 1.2]),
        torch.tensor(
            [
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
            ]
        ),
    ],
)
def test_sample_neighbors_probs(replace, probs_or_mask):
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    num_nodes = 5
    num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(indptr, indices)

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4])
    subgraph = graph.sample_neighbors(
        nodes,
        fanouts=torch.tensor([2]),
        replace=replace,
        probs_or_mask=probs_or_mask,
    )

    # Verify in subgraph.
    sampled_num = subgraph.indices.size(0)
    if replace:
        assert sampled_num == 6
    else:
        assert sampled_num == 4


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "probs_or_mask",
    [
        torch.zeros(12, dtype=torch.float32),
        torch.zeros(12, dtype=torch.bool),
    ],
)
def test_sample_neighbors_zero_probs(replace, probs_or_mask):
    # Initialize data.
    num_nodes = 5
    num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == num_edges
    assert indptr[-1] == len(indices)

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(indptr, indices)

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4])
    subgraph = graph.sample_neighbors(
        nodes,
        fanouts=torch.tensor([5]),
        replace=replace,
        probs_or_mask=probs_or_mask,
    )

    # Verify in subgraph.
    sampled_num = subgraph.indices.size(0)
    assert sampled_num == 0


def check_tensors_on_the_same_shared_memory(t1: torch.Tensor, t2: torch.Tensor):
    """Check if two tensors are on the same shared memory.

    This function copies a random tensor value to `t1` and checks whether `t2`
    holds the same random value and checks whether t2 is a distinct tensor
    object from `t1`. Their equality confirms that they are separate tensors
    that rely on the shared memory for their tensor value.
    """
    assert t1.data_ptr() != t2.data_ptr()
    old_t1 = t1.clone()
    v = torch.randint_like(t1, 100)
    t1[:] = v
    assert torch.equal(t1, t2)
    t1[:] = old_t1


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="CSCSamplingGraph is only supported on CPU.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
def test_homo_graph_on_shared_memory(num_nodes, num_edges):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    graph = gb.from_csc(csc_indptr, indices)

    shm_name = "test_homo_g"
    graph1 = graph.copy_to_shared_memory(shm_name)
    graph2 = gb.load_from_shared_memory(shm_name, graph.metadata)

    assert graph1.num_nodes == num_nodes
    assert graph1.num_nodes == num_nodes
    assert graph2.num_edges == num_edges
    assert graph2.num_edges == num_edges

    # Test the value of graph1 is correct
    assert torch.equal(graph1.csc_indptr, csc_indptr)
    assert torch.equal(graph1.indices, indices)

    # Test the value of graph2 is correct
    assert torch.equal(graph2.csc_indptr, csc_indptr)
    assert torch.equal(graph2.indices, indices)

    # Test the memory of graph1 and graph2 is on shared memory
    check_tensors_on_the_same_shared_memory(
        graph1.csc_indptr, graph2.csc_indptr
    )
    check_tensors_on_the_same_shared_memory(graph1.indices, graph2.indices)

    assert graph1.metadata is None and graph2.metadata is None
    assert graph1.node_type_offset is None and graph2.node_type_offset is None
    assert graph1.type_per_edge is None and graph2.type_per_edge is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="CSCSamplingGraph is only supported on CPU.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
@pytest.mark.parametrize("num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1)])
def test_hetero_graph_on_shared_memory(
    num_nodes, num_edges, num_ntypes, num_etypes
):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    ) = random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes)
    graph = gb.from_csc(
        csc_indptr, indices, node_type_offset, type_per_edge, metadata
    )

    shm_name = "test_hetero_g"
    graph1 = graph.copy_to_shared_memory(shm_name)
    graph2 = gb.load_from_shared_memory(shm_name, graph.metadata)

    assert graph1.num_nodes == num_nodes
    assert graph1.num_nodes == num_nodes
    assert graph2.num_edges == num_edges
    assert graph2.num_edges == num_edges

    # Test the value of graph1 is correct
    assert torch.equal(graph1.csc_indptr, csc_indptr)
    assert torch.equal(graph1.indices, indices)
    assert torch.equal(graph1.node_type_offset, node_type_offset)
    assert torch.equal(graph1.type_per_edge, type_per_edge)

    # Test the value of graph2 is correct
    assert torch.equal(graph2.csc_indptr, csc_indptr)
    assert torch.equal(graph2.indices, indices)
    assert torch.equal(graph2.node_type_offset, node_type_offset)
    assert torch.equal(graph2.type_per_edge, type_per_edge)

    # Test the memory of graph1 and graph2 is on shared memory
    check_tensors_on_the_same_shared_memory(
        graph1.csc_indptr, graph2.csc_indptr
    )
    check_tensors_on_the_same_shared_memory(graph1.indices, graph2.indices)
    check_tensors_on_the_same_shared_memory(
        graph1.node_type_offset, graph2.node_type_offset
    )
    check_tensors_on_the_same_shared_memory(
        graph1.type_per_edge, graph2.type_per_edge
    )

    assert metadata.node_type_to_id == graph1.metadata.node_type_to_id
    assert metadata.edge_type_to_id == graph1.metadata.edge_type_to_id
    assert metadata.node_type_to_id == graph2.metadata.node_type_to_id
    assert metadata.edge_type_to_id == graph2.metadata.edge_type_to_id


if __name__ == "__main__":
    test_sample_neighbors()
    test_sample_neighbors_replace(True, 12)
    test_sample_neighbors_probs(
        False,
        torch.tensor([2.5, 0, 8.4, 0, 0.4, 1.2, 2.5, 0, 8.4, 0, 0.4, 1.2]),
    )
    test_sample_neighbors_zero_probs(True, torch.zeros(12, dtype=torch.float32))
