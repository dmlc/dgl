import dgl
import dgl.graphbolt as gb
import torch


def test_to_dgl_blocks_hetero():
    relation = "A:relation:B"
    node_pairs = [
        {relation: (torch.tensor([0, 1, 2]), torch.tensor([0, 4, 5]))},
        {relation: (torch.tensor([0, 1]), torch.tensor([3, 1]))},
    ]
    reverse_column_node_ids = [
        {"B": torch.tensor([10, 11, 12, 13, 14, 16])},
        {"B": torch.tensor([10, 11, 12, 13])},
    ]
    reverse_row_node_ids = [
        {
            "A": torch.tensor([5, 9, 7]),
            "B": torch.tensor([10, 11, 12, 13, 14, 16]),
        },
        {
            "A": torch.tensor([5, 9]),
            "B": torch.tensor([10, 11, 12, 13]),
        },
    ]
    reverse_edge_ids = [
        {relation: torch.tensor([19, 20, 21])},
        {relation: torch.tensor([10, 12])},
    ]
    node_features = {
        ("A", "x"): torch.randint(0, 10, (3,)),
    }
    edge_features = [
        {(relation, "x"): torch.randint(0, 10, (3,))},
        {(relation, "x"): torch.randint(0, 10, (2,))},
    ]
    subgraphs = []
    for i in range(2):
        subgraphs.append(
            gb.SampledSubgraphImpl(
                node_pairs=node_pairs[i],
                reverse_column_node_ids=reverse_column_node_ids[i],
                reverse_row_node_ids=reverse_row_node_ids[i],
                reverse_edge_ids=reverse_edge_ids[i],
            )
        )
    blocks = gb.MiniBatch(
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
    ).to_dgl_blocks()

    for i, block in enumerate(blocks):
        assert torch.equal(block.edges()[0], node_pairs[i][relation][0])
        assert torch.equal(block.edges()[1], node_pairs[i][relation][1])
        assert torch.equal(block.edata[dgl.EID], reverse_edge_ids[i][relation])
        assert torch.equal(
            block.edges[gb.etype_str_to_tuple(relation)].data["x"],
            edge_features[i][(relation, "x")],
        )
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["A"], reverse_row_node_ids[0]["A"]
    )
    assert torch.equal(
        blocks[0].srcnodes["A"].data["x"], node_features[("A", "x")]
    )


def test_to_dgl_blocks_homo():
    node_pairs = [
        (
            torch.tensor([0, 1, 2, 2, 2, 1]),
            torch.tensor([0, 1, 1, 2, 5, 2]),
        ),
        (
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 0, 0]),
        ),
    ]
    reverse_column_node_ids = [
        torch.tensor([10, 11, 12, 13, 14, 16]),
        torch.tensor([10, 11]),
    ]
    reverse_row_node_ids = [
        torch.tensor([10, 11, 12, 13, 14, 16]),
        torch.tensor([10, 11, 12]),
    ]
    reverse_edge_ids = [
        torch.tensor([19, 20, 21, 22, 25, 30]),
        torch.tensor([10, 15, 17]),
    ]
    node_features = {"x": torch.randint(0, 10, (6,))}
    edge_features = [
        {"x": torch.randint(0, 10, (6,))},
        {"x": torch.randint(0, 10, (3,))},
    ]
    subgraphs = []
    for i in range(2):
        subgraphs.append(
            gb.SampledSubgraphImpl(
                node_pairs=node_pairs[i],
                reverse_column_node_ids=reverse_column_node_ids[i],
                reverse_row_node_ids=reverse_row_node_ids[i],
                reverse_edge_ids=reverse_edge_ids[i],
            )
        )
    blocks = gb.MiniBatch(
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
    ).to_dgl_blocks()

    for i, block in enumerate(blocks):
        assert torch.equal(block.edges()[0], node_pairs[i][0])
        assert torch.equal(block.edges()[1], node_pairs[i][1])
        assert torch.equal(block.edata[dgl.EID], reverse_edge_ids[i])
        assert torch.equal(block.edata["x"], edge_features[i]["x"])
    assert torch.equal(blocks[0].srcdata[dgl.NID], reverse_row_node_ids[0])
    assert torch.equal(blocks[0].srcdata["x"], node_features["x"])
