import dgl
import dgl.graphbolt as gb
import pytest
import torch


def create_homo_minibatch():
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
    return gb.MiniBatch(
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
    )


def create_hetero_minibatch():
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
    return gb.MiniBatch(
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
    )


def test_to_dgl_blocks_hetero():
    relation = "A:r:B"
    reverse_relation = "B:rr:A"
    minibatch = create_hetero_minibatch()
    blocks = minibatch.to_dgl_blocks()

    etype = gb.etype_str_to_tuple(relation)
    node_pairs = [
        subgraph.node_pairs for subgraph in minibatch.sampled_subgraphs
    ]
    reverse_edge_ids = [
        subgraph.reverse_edge_ids for subgraph in minibatch.sampled_subgraphs
    ]
    reverse_row_node_ids = [
        subgraph.reverse_row_node_ids
        for subgraph in minibatch.sampled_subgraphs
    ]

    for i, block in enumerate(blocks):
        edges = block.edges(etype=etype)
        assert torch.equal(edges[0], node_pairs[i][relation][0])
        assert torch.equal(edges[1], node_pairs[i][relation][1])
        assert torch.equal(
            block.edges[etype].data[dgl.EID], reverse_edge_ids[i][relation]
        )
        assert torch.equal(
            block.edges[etype].data["x"],
            minibatch.edge_features[i][(relation, "x")],
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
        blocks[0].srcnodes["A"].data["x"], minibatch.node_features[("A", "x")]
    )


def test_to_dgl_blocks_homo():
    minibatch = create_homo_minibatch()
    blocks = minibatch.to_dgl_blocks()
    node_pairs = [
        subgraph.node_pairs for subgraph in minibatch.sampled_subgraphs
    ]
    reverse_edge_ids = [
        subgraph.reverse_edge_ids for subgraph in minibatch.sampled_subgraphs
    ]
    reverse_row_node_ids = [
        subgraph.reverse_row_node_ids
        for subgraph in minibatch.sampled_subgraphs
    ]
    for i, block in enumerate(blocks):
        assert torch.equal(block.edges()[0], node_pairs[i][0])
        assert torch.equal(block.edges()[1], node_pairs[i][1])
        assert torch.equal(block.edata[dgl.EID], reverse_edge_ids[i])
        assert torch.equal(block.edata["x"], minibatch.edge_features[i]["x"])
    assert torch.equal(blocks[0].srcdata[dgl.NID], reverse_row_node_ids[0])
    assert torch.equal(blocks[0].srcdata["x"], minibatch.node_features["x"])


def test_to_dgl_computing_pack_node_homo():
    minibatch = create_homo_minibatch()
    minibatch.labels = torch.tensor([1, 2])
    pack = minibatch.to_dgl_computing_pack()
    # Assert
    assert len(pack["blocks"]) == 2
    assert torch.equal(pack["labels"], minibatch.labels)


@pytest.mark.parametrize("mode", ["neg_graph", "neg_src", "neg_dst"])
def test_to_dgl_computing_pack_link_homo(mode):
    # Arrange
    minibatch = create_homo_minibatch()
    minibatch.compacted_node_pairs = (
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )
    if mode == "neg_graph" or mode == "neg_src":
        minibatch.compacted_negative_srcs = torch.tensor([[0, 0], [1, 1]])
    if mode == "neg_graph" or mode == "neg_dst":
        minibatch.compacted_negative_dsts = torch.tensor([[1, 0], [0, 1]])
    # Act
    pack = minibatch.to_dgl_computing_pack()

    # Assert
    assert len(pack["blocks"]) == 2
    assert torch.equal(
        pack["positive_graph"][0], minibatch.compacted_node_pairs[0]
    )
    assert torch.equal(
        pack["positive_graph"][1], minibatch.compacted_node_pairs[1]
    )
    if mode == "neg_graph" or mode == "neg_src":
        assert torch.equal(
            pack["negative_graph"][0],
            minibatch.compacted_negative_srcs.view(-1),
        )
    if mode == "neg_graph" or mode == "neg_dst":
        assert torch.equal(
            pack["negative_graph"][1],
            minibatch.compacted_negative_dsts.view(-1),
        )


def test_to_dgl_computing_pack_node_hetero():
    minibatch = create_homo_minibatch()
    minibatch.labels = torch.tensor([1, 2])
    pack = minibatch.to_dgl_computing_pack()
    # Assert
    assert len(pack["blocks"]) == 2
    assert torch.equal(pack["labels"], minibatch.labels)


# def test_to_dgl_computing_pack_with_negative_samples_from_sources(self):
#     # Arrange
#     subgraph = Subgraph(
#         blocks=[None],
#         compacted_node_pairs=torch.tensor([[1, 2], [2, 3]]),
#         compacted_negative_srcs=torch.tensor([[4, 5], [5, 6]]),
#         compacted_negative_dsts=None,
#     )

#     # Act
#     pack = subgraph.to_dgl_computing_pack()

#     # Assert
#     self.assertEqual(pack["blocks"], None)
#     self.assertEqual(pack["positive_graph"], torch.tensor([[1, 2], [2, 3]]))
#     self.assertEqual(
#         pack["negative_graph"],
#         (
#             torch.tensor([[4, 5], [5, 6]]),
#             torch.tensor([[2, 2], [3, 3]]),
#         ),
#     )

# def test_to_dgl_computing_pack_with_negative_samples_from_destinations(self):
#     # Arrange
#     subgraph = Subgraph(
#         blocks=[None],
#         compacted_node_pairs=torch.tensor([[1, 2], [2, 3]]),
#         compacted_negative_srcs=None,
#         compacted_negative_dsts=torch.tensor([[7, 8], [8, 9]]),
#     )

#     # Act
#     pack = subgraph.to_dgl_computing_pack()

#     # Assert
#     self.assertEqual(pack["blocks"], None)
#     self.assertEqual(pack["positive_graph"], torch.tensor([[1, 2], [2, 3]]))
#     self.assertEqual(
#         pack["negative_graph"],
#         (
#             torch.tensor([[1, 1], [2, 2]]),
#             torch.tensor([[7, 8], [8, 9]]),
#         ),
#     )
