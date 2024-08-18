import os

import pickle
import re
import tempfile
import unittest

import backend as F

import dgl
import dgl.graphbolt as gb
import pytest
import torch
import torch.multiprocessing as mp

from dgl.graphbolt.base import etype_str_to_tuple
from scipy import sparse as spsp

from .. import gb_test_utils as gbt

torch.manual_seed(3407)
mp.set_sharing_strategy("file_system")


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("total_num_nodes", [0, 1, 10, 100, 1000])
def test_empty_graph(total_num_nodes):
    csc_indptr = torch.zeros((total_num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    graph = gb.fused_csc_sampling_graph(csc_indptr, indices)
    assert graph.total_num_edges == 0
    assert graph.total_num_nodes == total_num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("total_num_nodes", [0, 1, 10, 100, 1000])
def test_hetero_empty_graph(total_num_nodes):
    csc_indptr = torch.zeros((total_num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    node_type_to_id, edge_type_to_id = gbt.get_type_to_id(
        num_ntypes=3, num_etypes=5
    )
    # Some node types have no nodes.
    if total_num_nodes == 0:
        node_type_offset = torch.zeros((4,), dtype=int)
    else:
        node_type_offset = torch.sort(torch.randint(0, total_num_nodes, (4,)))[
            0
        ]
        node_type_offset[0] = 0
        node_type_offset[-1] = total_num_nodes
    type_per_edge = torch.tensor([])
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        edge_attributes=None,
    )
    assert graph.total_num_edges == 0
    assert graph.total_num_nodes == total_num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)
    assert graph.node_type_to_id == node_type_to_id
    assert graph.edge_type_to_id == edge_type_to_id
    assert torch.equal(graph.node_type_offset, node_type_offset)
    assert torch.equal(graph.type_per_edge, type_per_edge)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "ntypes", [{"n1": 1, "n2": 1}, {5: 1, "n2": 2}, {"n1": 1.5, "n2": 2.0}]
)
def test_type_to_id_with_ntype_exception(ntypes):
    with pytest.raises(AssertionError):
        gb.fused_csc_sampling_graph(
            None, None, node_type_to_id=ntypes, edge_type_to_id={"e1": 1}
        )


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
        {"n1:e1:n2": 1, ("n1", "e2", "n3"): 1},
        {("n1", "e1", "n10"): 1},
        {"n1:e1:n2": 1.5},
    ],
)
def test_type_to_id_with_etype_exception(etypes):
    with pytest.raises(Exception):
        gb.fused_csc_sampling_graph(
            None,
            None,
            node_type_to_id={"n1": 0, "n2": 1, "n3": 2},
            edge_type_to_id=etypes,
        )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
def test_homo_graph(total_num_nodes, total_num_edges):
    csc_indptr, indices = gbt.random_homo_graph(
        total_num_nodes, total_num_edges
    )
    node_attributes = {
        "A1": torch.arange(total_num_nodes),
        "A2": torch.arange(total_num_nodes),
    }
    edge_attributes = {
        "A1": torch.randn(total_num_edges),
        "A2": torch.randn(total_num_edges),
    }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    assert graph.total_num_nodes == total_num_nodes
    assert graph.total_num_edges == total_num_edges

    assert torch.equal(csc_indptr, graph.csc_indptr)
    assert torch.equal(indices, graph.indices)

    assert graph.node_attributes == node_attributes
    assert graph.edge_attributes == edge_attributes
    assert graph.node_type_offset is None
    assert graph.type_per_edge is None
    assert graph.node_type_to_id is None
    assert graph.edge_type_to_id is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
@pytest.mark.parametrize("num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1)])
def test_hetero_graph(total_num_nodes, total_num_edges, num_ntypes, num_etypes):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(
        total_num_nodes, total_num_edges, num_ntypes, num_etypes
    )
    node_attributes = {
        "A1": torch.arange(total_num_nodes),
        "A2": torch.arange(total_num_nodes),
    }
    edge_attributes = {
        "A1": torch.randn(total_num_edges),
        "A2": torch.randn(total_num_edges),
    }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    assert graph.total_num_nodes == total_num_nodes
    assert graph.total_num_edges == total_num_edges

    assert torch.equal(csc_indptr, graph.csc_indptr)
    assert torch.equal(indices, graph.indices)
    assert torch.equal(node_type_offset, graph.node_type_offset)
    assert torch.equal(type_per_edge, graph.type_per_edge)
    assert graph.node_attributes == node_attributes
    assert graph.edge_attributes == edge_attributes
    assert node_type_to_id == graph.node_type_to_id
    assert edge_type_to_id == graph.edge_type_to_id


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
def test_num_nodes_edges_homo(total_num_nodes, total_num_edges):
    csc_indptr, indices = gbt.random_homo_graph(
        total_num_nodes, total_num_edges
    )
    edge_attributes = {
        "A1": torch.randn(total_num_edges),
        "A2": torch.randn(total_num_edges),
    }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr, indices, edge_attributes=edge_attributes
    )

    assert graph.num_nodes == total_num_nodes
    assert graph.num_edges == total_num_edges


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_num_nodes_hetero():
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
    total_num_nodes = 5
    total_num_edges = 12
    ntypes = {
        "N0": 0,
        "N1": 1,
    }
    etypes = {
        "N0:R0:N0": 0,
        "N0:R1:N1": 1,
        "N1:R2:N0": 2,
        "N1:R3:N1": 3,
        "N1:R4:N0": 4,
    }
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    node_type_offset = torch.LongTensor([0, 2, 5])
    type_per_edge = torch.LongTensor([0, 0, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)
    assert node_type_offset[-1] == total_num_nodes
    assert all(type_per_edge < len(etypes))

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    )

    # Verify nodes number per node types.
    assert graph.num_nodes == {
        "N0": 2,
        "N1": 3,
    }
    assert sum(graph.num_nodes.values()) == total_num_nodes
    # Verify edges number per edge types.
    assert graph.num_edges == {
        "N0:R0:N0": 2,
        "N0:R1:N1": 4,
        "N1:R2:N0": 3,
        "N1:R3:N1": 3,
        "N1:R4:N0": 0,
    }
    assert sum(graph.num_edges.values()) == total_num_edges


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
    (
        csc_indptr,
        indices,
        _,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(10, 50, num_ntypes, 5)
    with pytest.raises(Exception):
        gb.fused_csc_sampling_graph(
            csc_indptr,
            indices,
            node_type_offset=node_type_offset,
            type_per_edge=type_per_edge,
            node_type_to_id=node_type_to_id,
            edge_type_to_id=edge_type_to_id,
        )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
@pytest.mark.parametrize("has_node_attrs", [True, False])
@pytest.mark.parametrize("has_edge_attrs", [True, False])
def test_load_save_homo_graph(
    total_num_nodes, total_num_edges, has_node_attrs, has_edge_attrs
):
    csc_indptr, indices = gbt.random_homo_graph(
        total_num_nodes, total_num_edges
    )
    node_attributes = None
    if has_node_attrs:
        node_attributes = {
            "A": torch.arange(total_num_nodes),
            "B": torch.arange(total_num_nodes),
        }
    edge_attributes = None
    if has_edge_attrs:
        edge_attributes = {
            "A": torch.arange(total_num_edges),
            "B": torch.arange(total_num_edges),
        }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    with tempfile.TemporaryDirectory() as test_dir:
        filename = os.path.join(test_dir, "fused_csc_sampling_graph.pt")
        torch.save(graph, filename)
        graph2 = torch.load(filename)

    assert graph.total_num_nodes == graph2.total_num_nodes
    assert graph.total_num_edges == graph2.total_num_edges

    assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
    assert torch.equal(graph.indices, graph2.indices)

    assert graph.node_type_offset is None and graph2.node_type_offset is None
    assert graph.type_per_edge is None and graph2.type_per_edge is None
    assert graph.node_type_to_id is None and graph2.node_type_to_id is None
    assert graph.edge_type_to_id is None and graph2.edge_type_to_id is None
    if has_node_attrs:
        assert graph.node_attributes.keys() == graph2.node_attributes.keys()
        for key in graph.node_attributes.keys():
            assert torch.equal(
                graph.node_attributes[key], graph2.node_attributes[key]
            )
    else:
        assert graph.node_attributes is None and graph2.node_attributes is None
    if has_edge_attrs:
        assert graph.edge_attributes.keys() == graph2.edge_attributes.keys()
        for key in graph.edge_attributes.keys():
            assert torch.equal(
                graph.edge_attributes[key], graph2.edge_attributes[key]
            )
    else:
        assert graph.edge_attributes is None and graph2.edge_attributes is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
@pytest.mark.parametrize("num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1)])
@pytest.mark.parametrize("has_node_attrs", [True, False])
@pytest.mark.parametrize("has_edge_attrs", [True, False])
def test_load_save_hetero_graph(
    total_num_nodes,
    total_num_edges,
    num_ntypes,
    num_etypes,
    has_node_attrs,
    has_edge_attrs,
):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(
        total_num_nodes, total_num_edges, num_ntypes, num_etypes
    )
    node_attributes = None
    if has_node_attrs:
        node_attributes = {
            "A": torch.arange(total_num_nodes),
            "B": torch.arange(total_num_nodes),
        }
    edge_attributes = None
    if has_edge_attrs:
        edge_attributes = {
            "A": torch.arange(total_num_edges),
            "B": torch.arange(total_num_edges),
        }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    with tempfile.TemporaryDirectory() as test_dir:
        filename = os.path.join(test_dir, "fused_csc_sampling_graph.pt")
        torch.save(graph, filename)
        graph2 = torch.load(filename)

    assert graph.total_num_nodes == graph2.total_num_nodes
    assert graph.total_num_edges == graph2.total_num_edges

    assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
    assert torch.equal(graph.indices, graph2.indices)
    assert torch.equal(graph.node_type_offset, graph2.node_type_offset)
    assert torch.equal(graph.type_per_edge, graph2.type_per_edge)
    assert graph.node_type_to_id == graph2.node_type_to_id
    assert graph.edge_type_to_id == graph2.edge_type_to_id
    if has_node_attrs:
        assert graph.node_attributes.keys() == graph2.node_attributes.keys()
        for key in graph.node_attributes.keys():
            assert torch.equal(
                graph.node_attributes[key], graph2.node_attributes[key]
            )
    else:
        assert graph.node_attributes is None and graph2.node_attributes is None
    if has_edge_attrs:
        assert graph.edge_attributes.keys() == graph2.edge_attributes.keys()
        for key in graph.edge_attributes.keys():
            assert torch.equal(
                graph.edge_attributes[key], graph2.edge_attributes[key]
            )
    else:
        assert graph.edge_attributes is None and graph2.edge_attributes is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
@pytest.mark.parametrize("has_node_attrs", [True, False])
@pytest.mark.parametrize("has_edge_attrs", [True, False])
def test_pickle_homo_graph(
    total_num_nodes, total_num_edges, has_node_attrs, has_edge_attrs
):
    csc_indptr, indices = gbt.random_homo_graph(
        total_num_nodes, total_num_edges
    )
    node_attributes = None
    if has_node_attrs:
        node_attributes = {
            "A": torch.arange(total_num_nodes),
            "B": torch.arange(total_num_nodes),
        }
    edge_attributes = None
    if has_edge_attrs:
        edge_attributes = {
            "A": torch.arange(total_num_edges),
            "B": torch.arange(total_num_edges),
        }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    serialized = pickle.dumps(graph)
    graph2 = pickle.loads(serialized)

    assert graph.total_num_nodes == graph2.total_num_nodes
    assert graph.total_num_edges == graph2.total_num_edges

    assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
    assert torch.equal(graph.indices, graph2.indices)

    assert graph.node_type_offset is None and graph2.node_type_offset is None
    assert graph.type_per_edge is None and graph2.type_per_edge is None
    assert graph.node_type_to_id is None and graph2.node_type_to_id is None
    assert graph.edge_type_to_id is None and graph2.edge_type_to_id is None
    if has_node_attrs:
        assert graph.node_attributes.keys() == graph2.node_attributes.keys()
        for key in graph.node_attributes.keys():
            assert torch.equal(
                graph.node_attributes[key], graph2.node_attributes[key]
            )
    else:
        assert graph.node_attributes is None and graph2.node_attributes is None
    if has_edge_attrs:
        assert graph.edge_attributes.keys() == graph2.edge_attributes.keys()
        for key in graph.edge_attributes.keys():
            assert torch.equal(
                graph.edge_attributes[key], graph2.edge_attributes[key]
            )
    else:
        assert graph.edge_attributes is None and graph2.edge_attributes is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
@pytest.mark.parametrize("num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1)])
@pytest.mark.parametrize("has_node_attrs", [True, False])
@pytest.mark.parametrize("has_edge_attrs", [True, False])
def test_pickle_hetero_graph(
    total_num_nodes,
    total_num_edges,
    num_ntypes,
    num_etypes,
    has_node_attrs,
    has_edge_attrs,
):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(
        total_num_nodes, total_num_edges, num_ntypes, num_etypes
    )
    node_attributes = None
    if has_node_attrs:
        node_attributes = {
            "A": torch.arange(total_num_nodes),
            "B": torch.arange(total_num_nodes),
        }
    edge_attributes = None
    if has_edge_attrs:
        edge_attributes = {
            "A": torch.arange(total_num_edges),
            "B": torch.arange(total_num_edges),
        }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    serialized = pickle.dumps(graph)
    graph2 = pickle.loads(serialized)

    assert graph.total_num_nodes == graph2.total_num_nodes
    assert graph.total_num_edges == graph2.total_num_edges

    assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
    assert torch.equal(graph.indices, graph2.indices)
    assert torch.equal(graph.node_type_offset, graph2.node_type_offset)
    assert torch.equal(graph.type_per_edge, graph2.type_per_edge)
    assert graph.node_type_to_id.keys() == graph2.node_type_to_id.keys()
    for i in graph.node_type_to_id.keys():
        assert graph.node_type_to_id[i] == graph2.node_type_to_id[i]
    assert graph.edge_type_to_id.keys() == graph2.edge_type_to_id.keys()
    for i in graph.edge_type_to_id.keys():
        assert graph.edge_type_to_id[i] == graph2.edge_type_to_id[i]
    if has_node_attrs:
        assert graph.node_attributes.keys() == graph2.node_attributes.keys()
        for key in graph.node_attributes.keys():
            assert torch.equal(
                graph.node_attributes[key], graph2.node_attributes[key]
            )
    else:
        assert graph.node_attributes is None and graph2.node_attributes is None
    if has_edge_attrs:
        assert graph.edge_attributes.keys() == graph2.edge_attributes.keys()
        for key in graph.edge_attributes.keys():
            assert torch.equal(
                graph.edge_attributes[key], graph2.edge_attributes[key]
            )
    else:
        assert graph.edge_attributes is None and graph2.edge_attributes is None


def process_csc_sampling_graph_multiprocessing(graph):
    return graph.total_num_nodes


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_multiprocessing():
    total_num_nodes = 5
    total_num_edges = 10
    num_ntypes = 2
    num_etypes = 3
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(
        total_num_nodes, total_num_edges, num_ntypes, num_etypes
    )
    edge_attributes = {
        "a": torch.randn((total_num_edges,)),
    }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        edge_attributes=edge_attributes,
    )

    p = mp.Process(
        target=process_csc_sampling_graph_multiprocessing, args=(graph,)
    )
    p.start()
    p.join()


def test_in_subgraph_homo():
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    total_num_nodes = 5
    total_num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(indptr, indices).to(F.ctx())

    # Extract in subgraph.
    nodes = torch.tensor([4, 1, 3], device=F.ctx())
    in_subgraph = graph.in_subgraph(nodes)

    # Verify in subgraph.
    assert torch.equal(
        in_subgraph.sampled_csc.indices,
        torch.tensor([0, 3, 4, 2, 3, 1, 2], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc.indptr,
        torch.tensor([0, 3, 5, 7], device=F.ctx()),
    )
    assert in_subgraph.original_column_node_ids is None
    assert in_subgraph.original_row_node_ids is None
    assert torch.equal(
        in_subgraph.original_edge_ids,
        torch.tensor([9, 10, 11, 3, 4, 7, 8], device=F.ctx()),
    )


def test_in_subgraph_hetero():
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
    total_num_nodes = 5
    total_num_edges = 12
    ntypes = {
        "N0": 0,
        "N1": 1,
    }
    etypes = {
        "N0:R0:N0": 0,
        "N0:R1:N1": 1,
        "N1:R2:N0": 2,
        "N1:R3:N1": 3,
    }
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    node_type_offset = torch.LongTensor([0, 2, 5])
    type_per_edge = torch.LongTensor([0, 0, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)
    assert node_type_offset[-1] == total_num_nodes
    assert all(type_per_edge < len(etypes))

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    ).to(F.ctx())

    # Extract in subgraph.
    nodes = {
        "N0": torch.tensor([1], device=F.ctx()),
        "N1": torch.tensor([2, 1], device=F.ctx()),
    }
    in_subgraph = graph.in_subgraph(nodes)

    # Verify in subgraph.
    assert torch.equal(
        in_subgraph.sampled_csc["N0:R0:N0"].indices,
        torch.tensor([], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc["N0:R0:N0"].indptr,
        torch.tensor([0, 0], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc["N0:R1:N1"].indices,
        torch.tensor([0, 1], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc["N0:R1:N1"].indptr,
        torch.tensor([0, 1, 2], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc["N1:R2:N0"].indices,
        torch.tensor([0, 1], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc["N1:R2:N0"].indptr,
        torch.tensor([0, 2], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc["N1:R3:N1"].indices,
        torch.tensor([1, 2, 0], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.sampled_csc["N1:R3:N1"].indptr,
        torch.tensor([0, 2, 3], device=F.ctx()),
    )
    assert in_subgraph.original_column_node_ids is None
    assert in_subgraph.original_row_node_ids is None
    assert torch.equal(
        in_subgraph.original_edge_ids["N0:R0:N0"],
        torch.tensor([], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.original_edge_ids["N0:R1:N1"],
        torch.tensor([9, 7], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.original_edge_ids["N1:R2:N0"],
        torch.tensor([3, 4], device=F.ctx()),
    )
    assert torch.equal(
        in_subgraph.original_edge_ids["N1:R3:N1"],
        torch.tensor([10, 11, 8], device=F.ctx()),
    )


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize("use_node_timestamp", [False, True])
@pytest.mark.parametrize("use_edge_timestamp", [False, True])
def test_temporal_sample_neighbors_homo(
    indptr_dtype,
    indices_dtype,
    replace,
    labor,
    use_node_timestamp,
    use_edge_timestamp,
):
    if replace and F._default_context_str == "gpu":
        pytest.skip("Sampling with replacement not yet implemented on the GPU.")
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    total_num_nodes = 5
    total_num_edges = 12
    indptr = torch.tensor([0, 3, 5, 7, 9, 12], dtype=indptr_dtype)
    indices = torch.tensor(
        [0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4], dtype=indices_dtype
    )
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)
    assert len(indptr) == total_num_nodes + 1

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(indptr, indices).to(F.ctx())

    # Generate subgraph via sample neighbors.
    fanouts = torch.LongTensor([2])
    sampler = (
        graph.temporal_sample_layer_neighbors
        if labor
        else graph.temporal_sample_neighbors
    )

    seed_list = [1, 3, 4]
    seed_timestamp = torch.randint(
        0, 100, (len(seed_list),), dtype=torch.int64, device=F.ctx()
    )
    if use_node_timestamp:
        node_timestamp = torch.randint(
            0, 100, (total_num_nodes,), dtype=torch.int64, device=F.ctx()
        )
        graph.node_attributes = {"timestamp": node_timestamp}
    if use_edge_timestamp:
        edge_timestamp = torch.randint(
            0, 100, (total_num_edges,), dtype=torch.int64, device=F.ctx()
        )
        graph.edge_attributes = {"timestamp": edge_timestamp}

    # Sample with nodes in mismatched dtype with graph's indices.
    nodes = torch.tensor(
        seed_list,
        dtype=(torch.int64 if indices_dtype == torch.int32 else torch.int32),
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Data type of nodes must be consistent with indices.dtype"
        ),
    ):
        _ = sampler(
            nodes,
            seed_timestamp,
            fanouts,
            replace=replace,
            node_timestamp_attr_name=(
                "timestamp" if use_node_timestamp else None
            ),
            edge_timestamp_attr_name=(
                "timestamp" if use_edge_timestamp else None
            ),
        )

    def _get_available_neighbors():
        available_neighbors = []
        for i, seed in enumerate(seed_list):
            neighbors = []
            start = indptr[seed].item()
            end = indptr[seed + 1].item()
            for j in range(start, end):
                neighbor = indices[j].item()
                if (
                    use_node_timestamp
                    and (node_timestamp[neighbor] >= seed_timestamp[i]).item()
                ):
                    continue
                if (
                    use_edge_timestamp
                    and (edge_timestamp[j] >= seed_timestamp[i]).item()
                ):
                    continue
                neighbors.append(neighbor)
            available_neighbors.append(neighbors)
        return available_neighbors

    nodes = torch.tensor(seed_list, dtype=indices_dtype, device=F.ctx())
    subgraph = sampler(
        nodes,
        seed_timestamp,
        fanouts,
        replace=replace,
        node_timestamp_attr_name="timestamp" if use_node_timestamp else None,
        edge_timestamp_attr_name="timestamp" if use_edge_timestamp else None,
    )
    sampled_count = torch.diff(subgraph.sampled_csc.indptr).tolist()
    available_neighbors = _get_available_neighbors()
    assert len(available_neighbors) == len(sampled_count)
    for i, count in enumerate(sampled_count):
        if not replace:
            expect_count = min(fanouts[0], len(available_neighbors[i]))
        else:
            expect_count = fanouts[0] if len(available_neighbors[i]) > 0 else 0
        assert count == expect_count
    sampled_neighbors = torch.split(subgraph.sampled_csc.indices, sampled_count)
    for i, neighbors in enumerate(sampled_neighbors):
        assert set(neighbors.tolist()).issubset(set(available_neighbors[i]))


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize("use_node_timestamp", [False, True])
@pytest.mark.parametrize("use_edge_timestamp", [False, True])
def test_temporal_sample_neighbors_hetero(
    indptr_dtype,
    indices_dtype,
    replace,
    labor,
    use_node_timestamp,
    use_edge_timestamp,
):
    if replace and F._default_context_str == "gpu":
        pytest.skip("Sampling with replacement not yet implemented on the GPU.")
    """Original graph in COO:
    "n1:e1:n2":[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]
    "n2:e2:n1":[0, 0, 1, 2], [0, 1, 1 ,0]
    0   0   1   0   1
    0   0   1   1   1
    1   1   0   0   0
    0   1   0   0   0
    1   0   0   0   0
    """
    # Initialize data.
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    ntypes_to_offset = {"n1": 0, "n2": 2}
    total_num_nodes = 5
    total_num_edges = 9
    indptr = torch.tensor([0, 2, 4, 6, 7, 9], dtype=indptr_dtype)
    indices = torch.tensor([2, 4, 2, 3, 0, 1, 1, 0, 1], dtype=indices_dtype)
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    ).to(F.ctx())

    # Generate subgraph via sample neighbors.
    fanouts = torch.LongTensor([-1, -1])
    sampler = (
        graph.temporal_sample_layer_neighbors
        if labor
        else graph.temporal_sample_neighbors
    )

    seeds = {
        "n1": torch.tensor([0], dtype=indices_dtype, device=F.ctx()),
        "n2": torch.tensor([0], dtype=indices_dtype, device=F.ctx()),
    }
    per_etype_destination_nodes = {
        "n1:e1:n2": torch.tensor([1], dtype=indices_dtype),
        "n2:e2:n1": torch.tensor([0], dtype=indices_dtype),
    }

    seed_timestamp = {
        "n1": torch.randint(0, 100, (1,), dtype=torch.int64, device=F.ctx()),
        "n2": torch.randint(0, 100, (1,), dtype=torch.int64, device=F.ctx()),
    }
    if use_node_timestamp:
        node_timestamp = torch.randint(
            0, 100, (total_num_nodes,), dtype=torch.int64, device=F.ctx()
        )
        graph.node_attributes = {"timestamp": node_timestamp}
    if use_edge_timestamp:
        edge_timestamp = torch.randint(
            0, 100, (total_num_edges,), dtype=torch.int64, device=F.ctx()
        )
        graph.edge_attributes = {"timestamp": edge_timestamp}

    subgraph = sampler(
        seeds,
        seed_timestamp,
        fanouts,
        replace=replace,
        node_timestamp_attr_name="timestamp" if use_node_timestamp else None,
        edge_timestamp_attr_name="timestamp" if use_edge_timestamp else None,
    )

    def _to_homo():
        ret_seeds, ret_timestamps = [], []
        for ntype, nodes in seeds.items():
            ntype_id = ntypes[ntype]
            offset = node_type_offset[ntype_id]
            ret_seeds.append(nodes + offset)
            ret_timestamps.append(seed_timestamp[ntype])
        return torch.cat(ret_seeds), torch.cat(ret_timestamps)

    homo_seeds, homo_seed_timestamp = _to_homo()

    def _get_available_neighbors():
        available_neighbors = []
        for i, seed in enumerate(homo_seeds):
            neighbors = []
            start = indptr[seed].item()
            end = indptr[seed + 1].item()
            for j in range(start, end):
                neighbor = indices[j].item()
                if (
                    use_node_timestamp
                    and (
                        node_timestamp[neighbor] >= homo_seed_timestamp[i]
                    ).item()
                ):
                    continue
                if (
                    use_edge_timestamp
                    and (edge_timestamp[j] >= homo_seed_timestamp[i]).item()
                ):
                    continue
                neighbors.append(neighbor)
            available_neighbors.append(neighbors)
        return available_neighbors

    available_neighbors = _get_available_neighbors()
    sampled_count = [0] * homo_seeds.numel()
    sampled_neighbors = [[] for _ in range(homo_seeds.numel())]
    for etype, csc in subgraph.sampled_csc.items():
        stype, _, _ = etype_str_to_tuple(etype)
        ntype_offset = ntypes_to_offset[stype]
        dest_nodes = per_etype_destination_nodes[etype]
        for i in range(dest_nodes.numel()):
            l = csc.indptr[i]
            r = csc.indptr[i + 1]
            seed_offset = dest_nodes[i].item()
            sampled_neighbors[seed_offset].extend(
                (csc.indices[l:r] + ntype_offset).tolist()
            )
            sampled_count[seed_offset] += r - l

    for i, count in enumerate(sampled_count):
        assert count == len(available_neighbors[i])
        assert set(sampled_neighbors[i]).issubset(set(available_neighbors[i]))


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


def check_node_edge_attributes(graph1, graph2, attributes, attr_name):
    for name, attr in attributes.items():
        edge_attributes_1 = getattr(graph1, attr_name)
        edge_attributes_2 = getattr(graph2, attr_name)
        assert name in edge_attributes_1
        assert name in edge_attributes_2
        assert torch.equal(edge_attributes_1[name], attr)
        check_tensors_on_the_same_shared_memory(
            edge_attributes_1[name], edge_attributes_2[name]
        )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="FusedCSCSamplingGraph is only supported on CPU.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50000)],
)
@pytest.mark.parametrize("test_node_attrs", [True, False])
@pytest.mark.parametrize("test_edge_attrs", [True, False])
def test_homo_graph_on_shared_memory(
    total_num_nodes, total_num_edges, test_node_attrs, test_edge_attrs
):
    csc_indptr, indices = gbt.random_homo_graph(
        total_num_nodes, total_num_edges
    )
    node_attributes = None
    if test_node_attrs:
        node_attributes = {
            "A1": torch.arange(total_num_nodes),
            "A2": torch.arange(total_num_nodes),
        }
    edge_attributes = None
    if test_edge_attrs:
        edge_attributes = {
            "A1": torch.randn(total_num_edges),
            "A2": torch.randn(total_num_edges),
        }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    shm_name = "test_homo_g"
    graph1 = graph.copy_to_shared_memory(shm_name)
    graph2 = gb.load_from_shared_memory(shm_name)

    assert graph1.total_num_nodes == total_num_nodes
    assert graph1.total_num_nodes == total_num_nodes
    assert graph2.total_num_edges == total_num_edges
    assert graph2.total_num_edges == total_num_edges

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

    if test_node_attrs:
        check_node_edge_attributes(
            graph1, graph2, node_attributes, "node_attributes"
        )
    if test_edge_attrs:
        check_node_edge_attributes(
            graph1, graph2, edge_attributes, "edge_attributes"
        )

    assert graph1.node_type_offset is None and graph2.node_type_offset is None
    assert graph1.type_per_edge is None and graph2.type_per_edge is None
    assert graph1.node_type_to_id is None and graph2.node_type_to_id is None
    assert graph1.edge_type_to_id is None and graph2.edge_type_to_id is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="FusedCSCSamplingGraph is only supported on CPU.",
)
@pytest.mark.parametrize(
    "total_num_nodes, total_num_edges",
    [(1, 1), (100, 1), (10, 50), (1000, 50 * 1000), (10 * 1000, 100 * 1000)],
)
@pytest.mark.parametrize(
    "num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1), (1000, 1000)]
)
@pytest.mark.parametrize("test_node_attrs", [True, False])
@pytest.mark.parametrize("test_edge_attrs", [True, False])
def test_hetero_graph_on_shared_memory(
    total_num_nodes,
    total_num_edges,
    num_ntypes,
    num_etypes,
    test_node_attrs,
    test_edge_attrs,
):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(
        total_num_nodes, total_num_edges, num_ntypes, num_etypes
    )

    node_attributes = None
    if test_node_attrs:
        node_attributes = {
            "A1": torch.arange(total_num_nodes),
            "A2": torch.arange(total_num_nodes),
        }

    edge_attributes = None
    if test_edge_attrs:
        edge_attributes = {
            "A1": torch.randn(total_num_edges),
            "A2": torch.randn(total_num_edges),
        }

    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    shm_name = "test_hetero_g"
    graph1 = graph.copy_to_shared_memory(shm_name)
    graph2 = gb.load_from_shared_memory(shm_name)

    assert graph1.total_num_nodes == total_num_nodes
    assert graph1.total_num_nodes == total_num_nodes
    assert graph2.total_num_edges == total_num_edges
    assert graph2.total_num_edges == total_num_edges

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

    if test_node_attrs:
        check_node_edge_attributes(
            graph1, graph2, node_attributes, "node_attributes"
        )
    if test_edge_attrs:
        check_node_edge_attributes(
            graph1, graph2, edge_attributes, "edge_attributes"
        )

    assert node_type_to_id == graph1.node_type_to_id
    assert edge_type_to_id == graph1.edge_type_to_id
    assert node_type_to_id == graph2.node_type_to_id
    assert edge_type_to_id == graph2.edge_type_to_id


def process_csc_sampling_graph_on_shared_memory(graph, data_queue, flag_queue):
    # Backup the attributes.
    csc_indptr = graph.csc_indptr.clone()
    indices = graph.indices.clone()
    node_type_offset = graph.node_type_offset.clone()
    type_per_edge = graph.type_per_edge.clone()

    # Change the value to random integers. Send the new value to the main
    # process.
    v = torch.randint_like(graph.csc_indptr, 100)
    graph.csc_indptr[:] = v
    data_queue.put(v.clone())

    v = torch.randint_like(graph.indices, 100)
    graph.indices[:] = v
    data_queue.put(v.clone())

    v = torch.randint_like(graph.node_type_offset, 100)
    graph.node_type_offset[:] = v
    data_queue.put(v.clone())

    v = torch.randint_like(graph.type_per_edge, 100)
    graph.type_per_edge[:] = v
    data_queue.put(v.clone())

    # Wait for the main process to finish.
    flag_queue.get()

    graph.csc_indptr[:] = csc_indptr
    graph.indices[:] = indices
    graph.node_type_offset[:] = node_type_offset
    graph.type_per_edge[:] = type_per_edge


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_multiprocessing_with_shared_memory():
    """Test if two CSCSamplingGraphs are on the same shared memory after
    spawning.

    For now this code only works when the sharing strategy of
    torch.multiprocessing is set to `file_system` at the beginning.
    The cause is still yet to be found.
    """

    total_num_nodes = 5
    total_num_edges = 10
    num_ntypes = 2
    num_etypes = 3
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(
        total_num_nodes, total_num_edges, num_ntypes, num_etypes
    )

    csc_indptr.share_memory_()
    indices.share_memory_()
    node_type_offset.share_memory_()
    type_per_edge.share_memory_()

    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        edge_attributes=None,
    )

    ctx = mp.get_context("spawn")  # Use spawn method.

    data_queue = ctx.Queue()  # Used for sending graph.
    flag_queue = ctx.Queue()  # Used for sending finish signal.

    p = ctx.Process(
        target=process_csc_sampling_graph_on_shared_memory,
        args=(graph, data_queue, flag_queue),
    )
    p.start()
    try:
        # Get data from the other process. Then check if the tensors here have
        # the same data.
        csc_indptr2 = data_queue.get()
        assert torch.equal(graph.csc_indptr, csc_indptr2)
        indices2 = data_queue.get()
        assert torch.equal(graph.indices, indices2)
        node_type_offset2 = data_queue.get()
        assert torch.equal(graph.node_type_offset, node_type_offset2)
        type_per_edge2 = data_queue.get()
        assert torch.equal(graph.type_per_edge, type_per_edge2)
    except:
        raise
    finally:
        # Send a finish signal to end sub-process.
        flag_queue.put(None)
    p.join()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph on GPU is not supported yet.",
)
def test_from_dglgraph_homogeneous():
    dgl_g = dgl.rand_graph(1000, 10 * 1000)

    # Check if the original edge id exist in edge attributes when the
    # original_edge_id is set to False.
    gb_g = gb.from_dglgraph(
        dgl_g, is_homogeneous=False, include_original_edge_id=False
    )
    assert (
        gb_g.edge_attributes is None
        or gb.ORIGINAL_EDGE_ID not in gb_g.edge_attributes
    )

    gb_g = gb.from_dglgraph(
        dgl_g, is_homogeneous=True, include_original_edge_id=True
    )
    # Get the COO representation of the FusedCSCSamplingGraph.
    num_columns = gb_g.csc_indptr.diff()
    rows = gb_g.indices
    columns = torch.arange(gb_g.total_num_nodes).repeat_interleave(num_columns)

    original_edge_ids = gb_g.edge_attributes[gb.ORIGINAL_EDGE_ID]
    assert torch.all(dgl_g.edges()[0][original_edge_ids] == rows)
    assert torch.all(dgl_g.edges()[1][original_edge_ids] == columns)

    assert gb_g.total_num_nodes == dgl_g.num_nodes()
    assert gb_g.total_num_edges == dgl_g.num_edges()
    assert gb_g.node_type_offset is None
    assert gb_g.type_per_edge is None
    assert gb_g.node_type_to_id is None
    assert gb_g.edge_type_to_id is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph on GPU is not supported yet.",
)
def test_from_dglgraph_heterogeneous():
    dgl_g = dgl.heterograph(
        {
            ("author", "writes", "paper"): (
                [1, 2, 3, 4, 5, 2],
                [1, 2, 3, 4, 5, 4],
            ),
            ("author", "affiliated_with", "institution"): (
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ),
            ("paper", "has_topic", "field"): ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ("paper", "cites", "paper"): (
                [2, 3, 4, 5, 6, 1],
                [1, 2, 3, 4, 5, 4],
            ),
        }
    )
    # Check if the original edge id exist in edge attributes when the
    # original_edge_id is set to False.
    gb_g = gb.from_dglgraph(
        dgl_g, is_homogeneous=False, include_original_edge_id=False
    )
    assert (
        gb_g.edge_attributes is None
        or gb.ORIGINAL_EDGE_ID not in gb_g.edge_attributes
    )

    gb_g = gb.from_dglgraph(
        dgl_g, is_homogeneous=False, include_original_edge_id=True
    )

    # `reverse_node_id` is used to map the node id in FusedCSCSamplingGraph to the
    # node id in Hetero-DGLGraph.
    num_ntypes = gb_g.node_type_offset.diff()
    reverse_node_id = torch.cat([torch.arange(num) for num in num_ntypes])

    # Get the COO representation of the FusedCSCSamplingGraph.
    num_columns = gb_g.csc_indptr.diff()
    rows = reverse_node_id[gb_g.indices]
    columns = reverse_node_id[
        torch.arange(gb_g.total_num_nodes).repeat_interleave(num_columns)
    ]

    # Check the order of etypes in DGLGraph is the same as FusedCSCSamplingGraph.
    assert (
        # Since the etypes in FusedCSCSamplingGraph is "srctype:etype:dsttype",
        # we need to split the string and get the middle part.
        list(
            map(
                lambda ss: ss.split(":")[1],
                gb_g.edge_type_to_id.keys(),
            )
        )
        == dgl_g.etypes
    )

    # Use ORIGINAL_EDGE_ID to check if the edge mapping is correct.
    for edge_idx in range(gb_g.total_num_edges):
        hetero_graph_idx = gb_g.type_per_edge[edge_idx]
        original_edge_id = gb_g.edge_attributes[gb.ORIGINAL_EDGE_ID][edge_idx]
        edge_type = dgl_g.etypes[hetero_graph_idx]
        dgl_edge_pairs = dgl_g.edges(etype=edge_type)
        assert dgl_edge_pairs[0][original_edge_id] == rows[edge_idx]
        assert dgl_edge_pairs[1][original_edge_id] == columns[edge_idx]

    assert gb_g.total_num_nodes == dgl_g.num_nodes()
    assert gb_g.total_num_edges == dgl_g.num_edges()
    assert torch.equal(gb_g.node_type_offset, torch.tensor([0, 6, 12, 18, 25]))
    assert torch.equal(
        gb_g.type_per_edge,
        torch.tensor(
            [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2]
        ),
    )
    assert gb_g.node_type_to_id == {
        "author": 0,
        "field": 1,
        "institution": 2,
        "paper": 3,
    }
    assert gb_g.edge_type_to_id == {
        "author:affiliated_with:institution": 0,
        "author:writes:paper": 1,
        "paper:cites:paper": 2,
        "paper:has_topic:field": 3,
    }


def create_fused_csc_sampling_graph():
    # Initialize data.
    total_num_nodes = 10
    total_num_edges = 9
    ntypes = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
    etypes = {
        "N0:R0:N1": 0,
        "N0:R1:N2": 1,
        "N0:R2:N3": 2,
    }
    indptr = torch.LongTensor([0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
    indices = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    node_type_offset = torch.LongTensor([0, 1, 4, 7, 10])
    type_per_edge = torch.LongTensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)
    assert node_type_offset[-1] == total_num_nodes
    assert all(type_per_edge < len(etypes))

    edge_attributes = {
        "mask": torch.BoolTensor([1, 1, 0, 1, 1, 1, 0, 0, 0]),
        "all": torch.BoolTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "zero": torch.BoolTensor([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    }

    # Construct FusedCSCSamplingGraph.
    return gb.fused_csc_sampling_graph(
        indptr,
        indices,
        edge_attributes=edge_attributes,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    )


def is_graph_on_device_type(graph, device_type):
    assert graph.csc_indptr.device.type == device_type
    assert graph.indices.device.type == device_type
    assert graph.node_type_offset.device.type == device_type
    assert graph.type_per_edge.device.type == device_type
    assert graph.csc_indptr.device.type == device_type
    for key in graph.edge_attributes:
        assert graph.edge_attributes[key].device.type == device_type


def is_graph_pinned(graph):
    assert graph.csc_indptr.is_pinned()
    assert graph.indices.is_pinned()
    assert graph.node_type_offset.is_pinned()
    assert graph.type_per_edge.is_pinned()
    assert graph.csc_indptr.is_pinned()
    for key in graph.edge_attributes:
        assert graph.edge_attributes[key].is_pinned()


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="`to` function needs GPU to test.",
)
@pytest.mark.parametrize("device", ["pinned", "cuda"])
def test_csc_sampling_graph_to_device(device):
    # Construct FusedCSCSamplingGraph.
    graph = create_fused_csc_sampling_graph()

    # Copy to device.
    graph2 = graph.to(device)

    if device == "cuda":
        is_graph_on_device_type(graph2, "cuda")
    elif device == "pinned":
        is_graph_on_device_type(graph2, "cpu")
        is_graph_pinned(graph2)

    # The original variable should be untouched.
    is_graph_on_device_type(graph, "cpu")


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Tests for pinned memory are only meaningful on GPU.",
)
@unittest.skipIf(
    gb.is_wsl(), reason="In place pinning is not supported on WSL."
)
def test_csc_sampling_graph_to_pinned_memory():
    # Construct FusedCSCSamplingGraph.
    graph = create_fused_csc_sampling_graph()
    ptr = graph.csc_indptr.data_ptr()

    # Copy to pinned_memory in-place.
    graph.pin_memory_()

    # Check if pinning is truly in-place.
    assert graph.csc_indptr.data_ptr() == ptr

    is_graph_on_device_type(graph, "cpu")
    is_graph_pinned(graph)


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize("is_pinned", [False, True])
@pytest.mark.parametrize("nodes", [None, True])
def test_sample_neighbors_homo(
    indptr_dtype, indices_dtype, labor, is_pinned, nodes
):
    if is_pinned and nodes is None:
        pytest.skip("Optional nodes and is_pinned is not supported together.")
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    if F._default_context_str == "cpu" and is_pinned:
        pytest.skip("Pinning is not meaningful without a GPU.")
    # Initialize data.
    total_num_edges = 12
    indptr = torch.tensor([0, 3, 5, 7, 9, 12], dtype=indptr_dtype)
    indices = torch.tensor(
        [0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4], dtype=indices_dtype
    )
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(indptr, indices).to(
        "pinned" if is_pinned else F.ctx()
    )

    # Generate subgraph via sample neighbors.
    if nodes:
        nodes = torch.tensor([1, 3, 4], dtype=indices_dtype).to(F.ctx())
    elif F._default_context_str != "gpu":
        pytest.skip("Optional nodes is supported only for the GPU.")
    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(nodes, fanouts=torch.LongTensor([2]))

    # Verify in subgraph.
    sampled_indptr_num = subgraph.sampled_csc.indptr.size(0)
    sampled_num = subgraph.sampled_csc.indices.size(0)
    assert sampled_num == len(subgraph.original_edge_ids)
    if nodes is None:
        assert sampled_indptr_num == indptr.shape[0]
        assert sampled_num == 10
    else:
        assert sampled_indptr_num == 4
        assert sampled_num == 6
    assert subgraph.original_column_node_ids is None
    assert subgraph.original_row_node_ids is None


@pytest.mark.parametrize("labor", [False, True])
def test_sample_neighbors_hetero_single_fanout(labor):
    u, i = torch.randint(20, size=(1000,)), torch.randint(10, size=(1000,))
    graph = dgl.heterograph({("u", "w", "i"): (u, i), ("i", "b", "u"): (i, u)})

    graph = gb.from_dglgraph(graph).to(F.ctx())

    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors

    for i in range(11):
        nodes = {"u": torch.randint(10, (100,), device=F.ctx())}
        sampler(nodes, fanouts=torch.tensor([-1]))
    # Should reach here without crashing.


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("labor", [False, True])
def test_sample_neighbors_hetero(indptr_dtype, indices_dtype, labor):
    """Original graph in COO:
    "n1:e1:n2":[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]
    "n2:e2:n1":[0, 0, 1, 2], [0, 1, 1 ,0]
    0   0   1   0   1
    0   0   1   1   1
    1   1   0   0   0
    0   1   0   0   0
    1   0   0   0   0
    """
    # Initialize data.
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    total_num_edges = 9
    indptr = torch.tensor([0, 2, 4, 6, 7, 9], dtype=indptr_dtype)
    indices = torch.tensor([2, 4, 2, 3, 0, 1, 1, 0, 1], dtype=indices_dtype)
    type_per_edge = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=indices_dtype
    )
    node_type_offset = torch.tensor([0, 2, 5], dtype=indices_dtype)
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    ).to(F.ctx())

    # Sample on both node types.
    nodes = {
        "n1": torch.tensor([0], dtype=indices_dtype, device=F.ctx()),
        "n2": torch.tensor([0], dtype=indices_dtype, device=F.ctx()),
    }
    fanouts = torch.tensor([-1, -1])
    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(nodes, fanouts)

    # Verify in subgraph.
    expected_sampled_csc = {
        "n1:e1:n2": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2], device=F.ctx()),
            indices=torch.tensor([0, 1], device=F.ctx()),
        ),
        "n2:e2:n1": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2], device=F.ctx()),
            indices=torch.tensor([0, 2], device=F.ctx()),
        ),
    }
    assert len(subgraph.sampled_csc) == 2
    for etype, pairs in expected_sampled_csc.items():
        assert torch.equal(subgraph.sampled_csc[etype].indptr, pairs.indptr)
        assert torch.equal(
            subgraph.sampled_csc[etype].indices.sort()[0], pairs.indices
        )
        assert len(pairs.indices) == len(subgraph.original_edge_ids[etype])
    assert subgraph.original_column_node_ids is None
    assert subgraph.original_row_node_ids is None

    # Sample on single node type.
    nodes = {"n1": torch.tensor([0], dtype=indices_dtype, device=F.ctx())}
    fanouts = torch.tensor([-1, -1])
    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(nodes, fanouts)

    # Verify in subgraph.
    expected_sampled_csc = {
        "n1:e1:n2": gb.CSCFormatBase(
            indptr=torch.tensor([0], device=F.ctx()),
            indices=torch.tensor([], device=F.ctx()),
        ),
        "n2:e2:n1": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2], device=F.ctx()),
            indices=torch.tensor([0, 2], device=F.ctx()),
        ),
    }
    assert len(subgraph.sampled_csc) == 2
    for etype, pairs in expected_sampled_csc.items():
        assert torch.equal(subgraph.sampled_csc[etype].indptr, pairs.indptr)
        assert torch.equal(
            subgraph.sampled_csc[etype].indices.sort()[0], pairs.indices
        )
        assert len(pairs.indices) == len(subgraph.original_edge_ids[etype])
    assert subgraph.original_column_node_ids is None
    assert subgraph.original_row_node_ids is None


@pytest.mark.parametrize(
    "fanouts, expected_sampled_num1, expected_sampled_num2",
    [
        ([0], 0, 0),
        ([1], 1, 1),
        ([2], 2, 2),
        ([4], 2, 2),
        ([-1], 2, 2),
        ([0, 0], 0, 0),
        ([1, 0], 1, 0),
        ([0, 1], 0, 1),
        ([1, 1], 1, 1),
        ([2, 1], 2, 1),
        ([-1, -1], 2, 2),
    ],
)
@pytest.mark.parametrize("labor", [False, True])
def test_sample_neighbors_fanouts(
    fanouts, expected_sampled_num1, expected_sampled_num2, labor
):
    """Original graph in COO:
    "n1:e1:n2":[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]
    "n2:e2:n1":[0, 0, 1, 2], [0, 1, 1 ,0]
    0   0   1   0   1
    0   0   1   1   1
    1   1   0   0   0
    0   1   0   0   0
    1   0   0   0   0
    """
    # Initialize data.
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    total_num_edges = 9
    indptr = torch.LongTensor([0, 2, 4, 6, 7, 9])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    ).to(F.ctx())

    nodes = {
        "n1": torch.tensor([0], device=F.ctx()),
        "n2": torch.tensor([0], device=F.ctx()),
    }
    fanouts = torch.LongTensor(fanouts)
    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(nodes, fanouts)

    # Verify in subgraph.
    assert (
        expected_sampled_num1 == 0
        or subgraph.sampled_csc["n1:e1:n2"].indices.numel()
        == expected_sampled_num1
    )
    assert subgraph.sampled_csc["n1:e1:n2"].indptr.size(0) == 2
    assert (
        expected_sampled_num2 == 0
        or subgraph.sampled_csc["n2:e2:n1"].indices.numel()
        == expected_sampled_num2
    )
    assert subgraph.sampled_csc["n2:e2:n1"].indptr.size(0) == 2


@pytest.mark.parametrize(
    "replace, expected_sampled_num1, expected_sampled_num2",
    [(False, 2, 2), (True, 4, 4)],
)
def test_sample_neighbors_replace(
    replace, expected_sampled_num1, expected_sampled_num2
):
    if F._default_context_str == "gpu" and replace == True:
        pytest.skip("Sampling with replacement not yet supported on GPU.")
    """Original graph in COO:
    "n1:e1:n2":[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]
    "n2:e2:n1":[0, 0, 1, 2], [0, 1, 1 ,0]
    0   0   1   0   1
    0   0   1   1   1
    1   1   0   0   0
    0   1   0   0   0
    1   0   0   0   0
    """
    # Initialize data.
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    total_num_edges = 9
    indptr = torch.LongTensor([0, 2, 4, 6, 7, 9])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    )

    nodes = {"n1": torch.LongTensor([0]), "n2": torch.LongTensor([0])}
    subgraph = graph.sample_neighbors(
        nodes, torch.LongTensor([4]), replace=replace
    )

    # Verify in subgraph.
    assert (
        subgraph.sampled_csc["n1:e1:n2"].indices.numel()
        == expected_sampled_num1
    )
    assert subgraph.sampled_csc["n1:e1:n2"].indptr.size(0) == 2
    assert (
        subgraph.sampled_csc["n2:e2:n1"].indices.numel()
        == expected_sampled_num2
    )
    assert subgraph.sampled_csc["n2:e2:n1"].indptr.size(0) == 2


@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize("is_pinned", [False, True])
def test_sample_neighbors_return_eids_homo(labor, is_pinned):
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    if F._default_context_str == "cpu" and is_pinned:
        pytest.skip("Pinning is not meaningful without a GPU.")
    # Initialize data.
    total_num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Add edge id mapping from CSC graph -> original graph.
    edge_attributes = {gb.ORIGINAL_EDGE_ID: torch.randperm(total_num_edges)}

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr, indices, edge_attributes=edge_attributes
    ).to("pinned" if is_pinned else F.ctx())

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4]).to(F.ctx())
    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(nodes, fanouts=torch.LongTensor([-1]))

    # Verify in subgraph.
    expected_reverse_edge_ids = edge_attributes[gb.ORIGINAL_EDGE_ID][
        torch.tensor([3, 4, 7, 8, 9, 10, 11])
    ].to(F.ctx())
    assert torch.equal(
        torch.sort(expected_reverse_edge_ids)[0],
        torch.sort(subgraph.original_edge_ids)[0],
    )
    assert subgraph.original_column_node_ids is None
    assert subgraph.original_row_node_ids is None


@pytest.mark.parametrize("labor", [False, True])
def test_sample_neighbors_return_eids_hetero(labor):
    """
    Original graph in COO:
    "n1:e1:n2":[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]
    "n2:e2:n1":[0, 0, 1, 2], [0, 1, 1 ,0]
    0   0   1   0   1
    0   0   1   1   1
    1   1   0   0   0
    0   1   0   0   0
    1   0   0   0   0
    """
    # Initialize data.
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    total_num_edges = 9
    indptr = torch.LongTensor([0, 2, 4, 6, 7, 9])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    edge_attributes = {
        gb.ORIGINAL_EDGE_ID: torch.cat([torch.randperm(4), torch.randperm(5)])
    }
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        edge_attributes=edge_attributes,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    ).to(F.ctx())

    # Sample on both node types.
    nodes = {
        "n1": torch.LongTensor([0]).to(F.ctx()),
        "n2": torch.LongTensor([0]).to(F.ctx()),
    }
    fanouts = torch.tensor([-1, -1])
    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(nodes, fanouts)

    expected_reverse_edge_ids = {
        "n2:e2:n1": graph.edge_attributes[gb.ORIGINAL_EDGE_ID][
            torch.tensor([0, 1], device=F.ctx())
        ],
        "n1:e1:n2": graph.edge_attributes[gb.ORIGINAL_EDGE_ID][
            torch.tensor([4, 5], device=F.ctx())
        ],
    }
    assert subgraph.original_column_node_ids is None
    assert subgraph.original_row_node_ids is None
    for etype in etypes.keys():
        assert torch.equal(
            subgraph.original_edge_ids[etype].sort()[0],
            expected_reverse_edge_ids[etype].sort()[0],
        )


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize("probs_name", ["weight", "mask"])
def test_sample_neighbors_probs(replace, labor, probs_name):
    if F._default_context_str == "gpu" and replace == True:
        pytest.skip("Sampling with replacement not yet supported on GPU.")
    """Original graph in COO:
    1   0   1   0   1
    1   0   1   1   0
    0   1   0   1   0
    0   1   0   0   1
    1   0   0   0   1
    """
    # Initialize data.
    total_num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    edge_attributes = {
        "weight": torch.FloatTensor(
            [2.5, 0, 8.4, 0, 0.4, 1.2, 2.5, 0, 8.4, 0.5, 0.4, 1.2]
        ),
        "mask": torch.BoolTensor([1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]),
    }

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr, indices, edge_attributes=edge_attributes
    )

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4])

    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(
        nodes,
        fanouts=torch.tensor([2]),
        replace=replace,
        probs_name=probs_name,
    )

    # Verify in subgraph.
    sampled_num = subgraph.sampled_csc.indices.size(0)
    assert subgraph.sampled_csc.indptr.size(0) == 4
    if replace:
        assert sampled_num == 6
    else:
        assert sampled_num == 4


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize(
    "probs_or_mask",
    [
        torch.zeros(12, dtype=torch.float32),
        torch.zeros(12, dtype=torch.bool),
    ],
)
def test_sample_neighbors_zero_probs(replace, labor, probs_or_mask):
    if F._default_context_str == "gpu" and replace == True:
        pytest.skip("Sampling with replacement not yet supported on GPU.")
    # Initialize data.
    total_num_nodes = 5
    total_num_edges = 12
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    edge_attributes = {"probs_or_mask": probs_or_mask}

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr, indices, edge_attributes=edge_attributes
    )

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([1, 3, 4])
    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors
    subgraph = sampler(
        nodes,
        fanouts=torch.tensor([5]),
        replace=replace,
        probs_name="probs_or_mask",
    )

    # Verify in subgraph.
    sampled_num = subgraph.sampled_csc.indices.size(0)
    assert subgraph.sampled_csc.indptr.size(0) == 4
    assert sampled_num == 0


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize(
    "fanouts, probs_name",
    [
        ([2], "mask"),
        ([3], "mask"),
        ([4], "mask"),
        ([-1], "mask"),
        ([7], "mask"),
        ([3], "all"),
        ([-1], "all"),
        ([7], "all"),
        ([3], "zero"),
        ([-1], "zero"),
        ([3], "none"),
        ([-1], "none"),
    ],
)
def test_sample_neighbors_homo_pick_number(fanouts, replace, labor, probs_name):
    if F._default_context_str == "gpu" and replace == True:
        pytest.skip("Sampling with replacement not yet supported on GPU.")
    """Original graph in COO:
    1   1   1   1   1   1
    0   0   0   0   0   0
    0   0   0   0   0   0
    0   0   0   0   0   0
    0   0   0   0   0   0
    0   0   0   0   0   0
    """
    # Initialize data.
    total_num_edges = 6
    indptr = torch.LongTensor([0, 6, 6, 6, 6, 6, 6])
    indices = torch.LongTensor([0, 1, 2, 3, 4, 5])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)

    edge_attributes = {
        "mask": torch.BoolTensor([1, 0, 0, 1, 0, 1]),
        "all": torch.BoolTensor([1, 1, 1, 1, 1, 1]),
        "zero": torch.BoolTensor([0, 0, 0, 0, 0, 0]),
    }

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr, indices, edge_attributes=edge_attributes
    )

    # Generate subgraph via sample neighbors.
    nodes = torch.LongTensor([0, 1])

    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors

    # Make sure no exception will be thrown.
    subgraph = sampler(
        nodes,
        fanouts=torch.LongTensor(fanouts),
        replace=replace,
        probs_name=probs_name if probs_name != "none" else None,
    )
    sampled_num = subgraph.sampled_csc.indices.size(0)
    assert subgraph.sampled_csc.indptr.size(0) == 3
    # Verify in subgraph.
    if probs_name == "mask":
        if fanouts[0] == -1:
            assert sampled_num == 3
        else:
            if replace:
                assert sampled_num == fanouts[0]
            else:
                assert sampled_num == min(fanouts[0], 3)
    elif probs_name == "zero":
        assert sampled_num == 0
    else:
        if fanouts[0] == -1:
            assert sampled_num == 6
        else:
            if replace:
                assert sampled_num == fanouts[0]
            else:
                assert sampled_num == min(fanouts[0], 6)


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("labor", [False, True])
@pytest.mark.parametrize(
    "fanouts, probs_name",
    [
        ([-1, -1, -1], "mask"),
        ([1, 1, 1], "mask"),
        ([2, 2, 2], "mask"),
        ([3, 3, 3], "mask"),
        ([4, 4, 4], "mask"),
        ([-1, 1, 3], "none"),
        ([2, -1, 4], "none"),
    ],
)
def test_sample_neighbors_hetero_pick_number(
    fanouts, replace, labor, probs_name
):
    if F._default_context_str == "gpu" and replace == True:
        pytest.skip("Sampling with replacement not yet supported on GPU.")
    # Initialize data.
    total_num_nodes = 10
    total_num_edges = 9
    ntypes = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
    etypes = {
        "N1:R0:N0": 0,
        "N2:R1:N0": 1,
        "N3:R2:N0": 2,
    }
    indptr = torch.LongTensor([0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
    indices = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    node_type_offset = torch.LongTensor([0, 1, 4, 7, 10])
    type_per_edge = torch.LongTensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert indptr[-1] == total_num_edges
    assert indptr[-1] == len(indices)
    assert node_type_offset[-1] == total_num_nodes
    assert all(type_per_edge < len(etypes))

    edge_attributes = {
        "mask": torch.BoolTensor([1, 1, 0, 1, 1, 1, 0, 0, 0]),
        "all": torch.BoolTensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "zero": torch.BoolTensor([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    }

    # Construct FusedCSCSamplingGraph.
    graph = gb.fused_csc_sampling_graph(
        indptr,
        indices,
        edge_attributes=edge_attributes,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    ).to(F.ctx())

    # Generate subgraph via sample neighbors.
    nodes = {
        "N0": torch.LongTensor([0]).to(F.ctx()),
        "N1": torch.LongTensor([1]).to(F.ctx()),
    }

    sampler = graph.sample_layer_neighbors if labor else graph.sample_neighbors

    # Make sure no exception will be thrown.
    subgraph = sampler(
        nodes,
        fanouts=torch.LongTensor(fanouts),
        replace=replace,
        probs_name=probs_name if probs_name != "none" else None,
    )
    print(subgraph)
    if probs_name == "none":
        for etype, pairs in subgraph.sampled_csc.items():
            assert pairs.indptr.size(0) == 2
            sampled_num = pairs.indices.size(0)
            fanout = fanouts[etypes[etype]]
            if fanout == -1:
                assert sampled_num == 3
            else:
                if replace:
                    assert sampled_num == fanout
                else:
                    assert sampled_num == min(fanout, 3)
    else:
        fanout = fanouts[0]  # Here fanout is the same for all etypes.
        for etype, pairs in subgraph.sampled_csc.items():
            assert pairs.indptr.size(0) == 2
            sampled_num = pairs.indices.size(0)
            if etypes[etype] == 0:
                # Etype 0: 2 valid neighbors.
                if fanout == -1:
                    assert sampled_num == 2
                else:
                    if replace:
                        assert sampled_num == fanout
                    else:
                        assert sampled_num == min(fanout, 2)
            elif etypes[etype] == 1:
                # Etype 1: 3 valid neighbors.
                if fanout == -1:
                    assert sampled_num == 3
                else:
                    if replace:
                        assert sampled_num == fanout
                    else:
                        assert sampled_num == min(fanout, 3)
            else:
                # Etype 2: 0 valid neighbors.
                assert sampled_num == 0


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
def test_graph_attributes():
    num_nodes = 1000
    num_edges = 10 * 1000
    csc_indptr, indices = gbt.random_homo_graph(num_nodes, num_edges)
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_attributes=None,
        edge_attributes=None,
    )

    # Case 1: default is None.
    assert graph.node_attributes is None
    assert graph.edge_attributes is None

    # Case 2: Assign the whole node/edge attributes.
    node_attributes = {
        "A": torch.rand(num_nodes, 2),
        "B": torch.rand(num_nodes, 2),
    }
    edge_attributes = {
        "A": torch.rand(num_nodes, 2),
        "B": torch.rand(num_nodes, 2),
    }
    graph.node_attributes = node_attributes
    graph.edge_attributes = edge_attributes
    for k, v in node_attributes.items():
        assert torch.equal(v, graph.node_attributes[k])
        assert torch.equal(v, graph.node_attribute(k))
    for k, v in edge_attributes.items():
        assert torch.equal(v, graph.edge_attributes[k])
        assert torch.equal(v, graph.edge_attribute(k))
    assert "C" not in graph.node_attributes
    assert "C" not in graph.edge_attributes
    with pytest.raises(RuntimeError, match="Node attribute C does not exist."):
        graph.node_attribute("C")
    with pytest.raises(RuntimeError, match="Edge attribute C does not exist."):
        graph.edge_attribute("C")

    # Case 3: Assign/overwrite more node/edge attributes into existing ones.
    for key in ["B", "C"]:
        node_attributes[key] = torch.rand(num_nodes, 2)
        edge_attributes[key] = torch.rand(num_edges, 2)
        graph.add_node_attribute(key, node_attributes[key])
        graph.add_edge_attribute(key, edge_attributes[key])
    for k, v in node_attributes.items():
        assert torch.equal(v, graph.node_attributes[k])
        assert torch.equal(v, graph.node_attribute(k))
    for k, v in edge_attributes.items():
        assert torch.equal(v, graph.edge_attributes[k])
        assert torch.equal(v, graph.edge_attribute(k))

    # Case 4: Assign more node/edge attributes which were None previously.
    graph.node_attributes = None
    graph.edge_attributes = None
    graph.add_node_attribute("C", node_attributes["C"])
    graph.add_edge_attribute("C", edge_attributes["C"])
    assert torch.equal(node_attributes["C"], graph.node_attribute("C"))
    assert torch.equal(node_attributes["C"], graph.node_attributes["C"])
    assert torch.equal(edge_attributes["C"], graph.edge_attribute("C"))
    assert torch.equal(edge_attributes["C"], graph.edge_attributes["C"])
