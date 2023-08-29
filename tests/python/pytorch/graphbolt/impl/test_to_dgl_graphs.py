import dgl
import dgl.graphbolt as gb
import torch


def test_to_dgl_graphs_hetero():
    relation = ("A", "relation", "B")
    node_pairs = {relation: (torch.tensor([0, 1, 2]), torch.tensor([0, 4, 5]))}
    reverse_column_node_ids = {"B": torch.tensor([10, 11, 12, 13, 14, 16])}
    reverse_row_node_ids = {
        "A": torch.tensor([5, 9, 7]),
        "B": torch.tensor([10, 11, 12, 13, 14, 16]),
    }
    reverse_edge_ids = {relation: torch.tensor([19, 20, 21])}
    node_feature = {
        ("A", "x"): torch.randint(0, 10, (3,)),
        ("B", "y"): torch.randint(0, 10, (6,)),
    }
    edge_feature = {(":".join(relation), "x"): torch.randint(0, 10, (3,))}
    subgraph = gb.SampledSubgraphImpl(
        node_pairs=node_pairs,
        reverse_column_node_ids=reverse_column_node_ids,
        reverse_row_node_ids=reverse_row_node_ids,
        reverse_edge_ids=reverse_edge_ids,
    )
    g = gb.DataBlock(
        sampled_subgraphs=[subgraph],
        node_feature=node_feature,
        edge_feature=[edge_feature],
    ).to_dgl_graphs()[0]

    assert torch.equal(g.edges()[0], node_pairs[relation][0])
    assert torch.equal(g.edges()[1], node_pairs[relation][1])
    assert torch.equal(g.ndata[dgl.NID]["A"], reverse_row_node_ids["A"])
    assert torch.equal(g.ndata[dgl.NID]["B"], reverse_row_node_ids["B"])
    assert torch.equal(g.edata[dgl.EID], reverse_edge_ids[relation])
    assert torch.equal(g.nodes["A"].data["x"], node_feature[("A", "x")])
    assert torch.equal(g.nodes["B"].data["y"], node_feature[("B", "y")])
    assert torch.equal(
        g.edges[relation].data["x"], edge_feature[("A:relation:B", "x")]
    )


def test_to_dgl_graphs_homo():
    node_pairs = (torch.tensor([0, 1, 2]), torch.tensor([0, 4, 5]))
    reverse_column_node_ids = torch.tensor([10, 11, 12])
    reverse_row_node_ids = torch.tensor([10, 11, 12, 13, 14, 16])
    reverse_edge_ids = torch.tensor([19, 20, 21])
    node_feature = {("None", "x"): torch.randint(0, 10, (6,))}
    edge_feature = {("None", "x"): torch.randint(0, 10, (3,))}
    subgraph = gb.SampledSubgraphImpl(
        node_pairs=node_pairs,
        reverse_column_node_ids=reverse_column_node_ids,
        reverse_row_node_ids=reverse_row_node_ids,
        reverse_edge_ids=reverse_edge_ids,
    )
    g = gb.DataBlock(
        sampled_subgraphs=[subgraph],
        node_feature=node_feature,
        edge_feature=[edge_feature],
    ).to_dgl_graphs()[0]

    assert torch.equal(g.edges()[0], node_pairs[0])
    assert torch.equal(g.edges()[1], node_pairs[1])
    assert torch.equal(g.ndata[dgl.NID], reverse_row_node_ids)
    assert torch.equal(g.edata[dgl.EID], reverse_edge_ids)
    assert torch.equal(g.ndata["x"], node_feature[("None", "x")])
    assert torch.equal(g.edata["x"], edge_feature[("None", "x")])
