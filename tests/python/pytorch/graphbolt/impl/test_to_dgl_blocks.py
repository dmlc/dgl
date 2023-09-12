import dgl
import dgl.graphbolt as gb
import torch


def test_to_dgl_blocks_hetero():
    relation = "A:relation:B"
    node_pairs = {relation: (torch.tensor([0, 1, 2]), torch.tensor([0, 4, 5]))}
    reverse_column_node_ids = {"B": torch.tensor([10, 11, 12, 13, 14, 16])}
    reverse_row_node_ids = {
        "A": torch.tensor([5, 9, 7]),
        "B": torch.tensor([10, 11, 12, 13, 14, 16]),
    }
    reverse_edge_ids = {relation: torch.tensor([19, 20, 21])}
    node_features = {
        ("A", "x"): torch.randint(0, 10, (3,)),
    }
    edge_features = {(relation, "x"): torch.randint(0, 10, (3,))}
    subgraph = gb.SampledSubgraphImpl(
        node_pairs=node_pairs,
        reverse_column_node_ids=reverse_column_node_ids,
        reverse_row_node_ids=reverse_row_node_ids,
        reverse_edge_ids=reverse_edge_ids,
    )
    b = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features=node_features,
        edge_features=[edge_features],
    )
    b = b.to_dgl_blocks()[0]

    assert torch.equal(b.edges()[0], node_pairs[relation][0])
    assert torch.equal(b.edges()[1], node_pairs[relation][1])
    assert torch.equal(b.srcdata[dgl.NID]["A"], reverse_row_node_ids["A"])
    assert torch.equal(b.edata[dgl.EID], reverse_edge_ids[relation])
    assert torch.equal(b.srcnodes["A"].data["x"], node_features[("A", "x")])
    assert torch.equal(
        b.edges[gb.etype_str_to_tuple(relation)].data["x"],
        edge_features[(relation, "x")],
    )


def test_to_dgl_blocks_homo():
    node_pairs = (
        torch.tensor([0, 1, 2, 2, 5, 5]),
        torch.tensor([0, 1, 1, 2, 2, 2]),
    )
    reverse_column_node_ids = torch.tensor([10, 11, 12])
    reverse_row_node_ids = torch.tensor([10, 11, 12, 13, 14, 16])
    reverse_edge_ids = torch.tensor([19, 20, 21, 22, 25, 30])
    node_features = {"x": torch.randint(0, 10, (6,))}
    edge_features = {"x": torch.randint(0, 10, (6,))}
    subgraph = gb.SampledSubgraphImpl(
        node_pairs=node_pairs,
        reverse_column_node_ids=reverse_column_node_ids,
        reverse_row_node_ids=reverse_row_node_ids,
        reverse_edge_ids=reverse_edge_ids,
    )
    b = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features=node_features,
        edge_features=[edge_features],
    ).to_dgl_blocks()[0]

    assert torch.equal(b.edges()[0], node_pairs[0])
    assert torch.equal(b.edges()[1], node_pairs[1])
    assert torch.equal(b.srcdata[dgl.NID], reverse_row_node_ids)
    assert torch.equal(b.edata[dgl.EID], reverse_edge_ids)
    assert torch.equal(b.srcdata["x"], node_features["x"])
    assert torch.equal(b.edata["x"], edge_features["x"])
