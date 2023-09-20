import dgl
import dgl.graphbolt as gb
import torch


def test_to_dgl_blocks_hetero():
    relation = "A:r:B"
    reverse_relation = "B:rr:A"
    node_pairs = [
        {
            relation: (torch.tensor([0, 1, 1]), torch.tensor([0, 1, 2])),
            reverse_relation: (torch.tensor([1, 0]), torch.tensor([2, 3])),
        },
        {relation: (torch.tensor([0, 1]), torch.tensor([1, 0]))},
    ]
    reverse_column_node_ids = [
        {"B": torch.tensor([10, 11, 12]), "A": torch.tensor([5, 7, 9, 11])},
        {"B": torch.tensor([10, 11])},
    ]
    reverse_row_node_ids = [
        {
            "A": torch.tensor([5, 7, 9, 11]),
            "B": torch.tensor([10, 11, 12]),
        },
        {
            "A": torch.tensor([5, 7]),
            "B": torch.tensor([10, 11]),
        },
    ]
    reverse_edge_ids = [
        {
            relation: torch.tensor([19, 20, 21]),
            reverse_relation: torch.tensor([23, 26]),
        },
        {relation: torch.tensor([10, 12])},
    ]
    node_features = {
        ("A", "x"): torch.randint(0, 10, (4,)),
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

    etype = gb.etype_str_to_tuple(relation)
    for i, block in enumerate(blocks):
        edges = block.edges(etype=etype)
        assert torch.equal(edges[0], node_pairs[i][relation][0])
        assert torch.equal(edges[1], node_pairs[i][relation][1])
        assert torch.equal(
            block.edges[etype].data[dgl.EID], reverse_edge_ids[i][relation]
        )
        assert torch.equal(
            block.edges[etype].data["x"],
            edge_features[i][(relation, "x")],
        )
    edges = blocks[0].edges(etype=gb.etype_str_to_tuple(reverse_relation))
    assert torch.equal(edges[0], node_pairs[0][reverse_relation][0])
    assert torch.equal(edges[1], node_pairs[0][reverse_relation][1])
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["A"], reverse_row_node_ids[0]["A"]
    )
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["B"], reverse_row_node_ids[0]["B"]
    )
    assert torch.equal(
        blocks[0].srcnodes["A"].data["x"], node_features[("A", "x")]
    )


test_to_dgl_blocks_hetero()


def test_to_dgl_blocks_homo():
    node_pairs = [
        (
            torch.tensor([0, 1, 2, 2, 2, 1]),
            torch.tensor([0, 1, 1, 2, 3, 2]),
        ),
        (
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 0, 0]),
        ),
    ]
    reverse_column_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11]),
    ]
    reverse_row_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11, 12]),
    ]
    reverse_edge_ids = [
        torch.tensor([19, 20, 21, 22, 25, 30]),
        torch.tensor([10, 15, 17]),
    ]
    node_features = {"x": torch.randint(0, 10, (4,))}
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


def test_minibatch_print():
    node_pairs = [
        (
            torch.tensor([0, 1, 2, 2, 2, 1]),
            torch.tensor([0, 1, 1, 2, 3, 2]),
        ),
        (
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 0, 0]),
        ),
    ]
    reverse_column_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11]),
    ]
    reverse_row_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11, 12]),
    ]
    reverse_edge_ids = [
        torch.tensor([19, 20, 21, 22, 25, 30]),
        torch.tensor([10, 15, 17]),
    ]
    node_features = {"x": torch.randint(0, 10, (4,))}
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
    negative_srcs = torch.tensor([[8], [1], [6]])
    negative_dsts = torch.tensor([[2], [8], [8]])
    input_nodes = torch.tensor([8, 1, 6, 5, 9, 0, 2, 4])
    compacted_node_pairs = (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    compacted_negative_srcs = torch.tensor([0, 1, 2])
    compacted_negative_dsts = torch.tensor([6, 0, 0])
    labels = torch.tensor([0.0, 1.0, 2.0])
    # minibatch without data
    mb_wo = gb.MiniBatch()
    print(mb_wo)
    # minibatch with all attributes
    mb_w = gb.MiniBatch(
        node_pairs=node_pairs,
        sampled_subgraphs=subgraphs,
        labels=labels,
        node_features=node_features,
        edge_features=edge_features,
        negative_srcs=negative_srcs,
        negative_dsts=negative_dsts,
        compacted_node_pairs=compacted_node_pairs,
        input_nodes=input_nodes,
        compacted_negative_srcs=compacted_negative_srcs,
        compacted_negative_dsts=compacted_negative_dsts,
    )
    print(mb_w)
