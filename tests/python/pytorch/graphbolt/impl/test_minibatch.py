import dgl
import dgl.graphbolt as gb
import pytest
import torch


relation = "A:r:B"
reverse_relation = "B:rr:A"


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
    original_column_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11]),
    ]
    original_row_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11, 12]),
    ]
    original_edge_ids = [
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
                original_column_node_ids=original_column_node_ids[i],
                original_row_node_ids=original_row_node_ids[i],
                original_edge_ids=original_edge_ids[i],
            )
        )
    return gb.MiniBatch(
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
        input_nodes=torch.tensor([10, 11, 12, 13]),
    )


def create_hetero_minibatch():
    node_pairs = [
        {
            relation: (torch.tensor([0, 1, 1]), torch.tensor([0, 1, 2])),
            reverse_relation: (torch.tensor([1, 0]), torch.tensor([2, 3])),
        },
        {relation: (torch.tensor([0, 1]), torch.tensor([1, 0]))},
    ]
    original_column_node_ids = [
        {"B": torch.tensor([10, 11, 12]), "A": torch.tensor([5, 7, 9, 11])},
        {"B": torch.tensor([10, 11])},
    ]
    original_row_node_ids = [
        {
            "A": torch.tensor([5, 7, 9, 11]),
            "B": torch.tensor([10, 11, 12]),
        },
        {
            "A": torch.tensor([5, 7]),
            "B": torch.tensor([10, 11]),
        },
    ]
    original_edge_ids = [
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
                original_column_node_ids=original_column_node_ids[i],
                original_row_node_ids=original_row_node_ids[i],
                original_edge_ids=original_edge_ids[i],
            )
        )
    return gb.MiniBatch(
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
        input_nodes={
            "A": torch.tensor([5, 7, 9, 11]),
            "B": torch.tensor([10, 11, 12]),
        },
    )


def test_minibatch_representation():
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
    original_column_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11]),
    ]
    original_row_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11, 12]),
    ]
    original_edge_ids = [
        torch.tensor([19, 20, 21, 22, 25, 30]),
        torch.tensor([10, 15, 17]),
    ]
    node_features = {"x": torch.tensor([7, 6, 2, 2])}
    edge_features = [
        {"x": torch.tensor([[8], [1], [6]])},
        {"x": torch.tensor([[2], [8], [8]])},
    ]
    subgraphs = []
    for i in range(2):
        subgraphs.append(
            gb.SampledSubgraphImpl(
                node_pairs=node_pairs[i],
                original_column_node_ids=original_column_node_ids[i],
                original_row_node_ids=original_row_node_ids[i],
                original_edge_ids=original_edge_ids[i],
            )
        )
    negative_srcs = torch.tensor([[8], [1], [6]])
    negative_dsts = torch.tensor([[2], [8], [8]])
    input_nodes = torch.tensor([8, 1, 6, 5, 9, 0, 2, 4])
    compacted_node_pairs = (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    compacted_negative_srcs = torch.tensor([0, 1, 2])
    compacted_negative_dsts = torch.tensor([6, 0, 0])
    labels = torch.tensor([0.0, 1.0, 2.0])
    # Test minibatch without data.
    minibatch = gb.MiniBatch()
    expect_result = str(
        """MiniBatch(seed_nodes=None,
          sampled_subgraphs=None,
          node_pairs=None,
          node_features=None,
          negative_srcs=None,
          negative_dsts=None,
          labels=None,
          input_nodes=None,
          edge_features=None,
          compacted_node_pairs=None,
          compacted_negative_srcs=None,
          compacted_negative_dsts=None,
       )"""
    )
    result = str(minibatch)
    assert result == expect_result, print(len(expect_result), len(result))
    # Test minibatch with all attributes.
    minibatch = gb.MiniBatch(
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
    expect_result = str(
        """MiniBatch(seed_nodes=None,
          sampled_subgraphs=[SampledSubgraphImpl(node_pairs=(tensor([0, 1, 2, 2, 2, 1]), tensor([0, 1, 1, 2, 3, 2])),
                                                original_column_node_ids=tensor([10, 11, 12, 13]),
                                                original_edge_ids=tensor([19, 20, 21, 22, 25, 30]),
                                                original_row_node_ids=tensor([10, 11, 12, 13]),),
                            SampledSubgraphImpl(node_pairs=(tensor([0, 1, 2]), tensor([1, 0, 0])),
                                                original_column_node_ids=tensor([10, 11]),
                                                original_edge_ids=tensor([10, 15, 17]),
                                                original_row_node_ids=tensor([10, 11, 12]),)],
          node_pairs=[(tensor([0, 1, 2, 2, 2, 1]), tensor([0, 1, 1, 2, 3, 2])),
                     (tensor([0, 1, 2]), tensor([1, 0, 0]))],
          node_features={'x': tensor([7, 6, 2, 2])},
          negative_srcs=tensor([[8],
                                [1],
                                [6]]),
          negative_dsts=tensor([[2],
                                [8],
                                [8]]),
          labels=tensor([0., 1., 2.]),
          input_nodes=tensor([8, 1, 6, 5, 9, 0, 2, 4]),
          edge_features=[{'x': tensor([[8],
                                       [1],
                                       [6]])},
                        {'x': tensor([[2],
                                       [8],
                                       [8]])}],
          compacted_node_pairs=(tensor([0, 1, 2]), tensor([3, 4, 5])),
          compacted_negative_srcs=tensor([0, 1, 2]),
          compacted_negative_dsts=tensor([6, 0, 0]),
       )"""
    )
    result = str(minibatch)
    assert result == expect_result, print(expect_result, result)


def test_dgl_minibatch_representation():
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
    original_column_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11]),
    ]
    original_row_node_ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([10, 11, 12]),
    ]
    original_edge_ids = [
        torch.tensor([19, 20, 21, 22, 25, 30]),
        torch.tensor([10, 15, 17]),
    ]
    node_features = {"x": torch.tensor([7, 6, 2, 2])}
    edge_features = [
        {"x": torch.tensor([[8], [1], [6]])},
        {"x": torch.tensor([[2], [8], [8]])},
    ]
    subgraphs = []
    for i in range(2):
        subgraphs.append(
            gb.SampledSubgraphImpl(
                node_pairs=node_pairs[i],
                original_column_node_ids=original_column_node_ids[i],
                original_row_node_ids=original_row_node_ids[i],
                original_edge_ids=original_edge_ids[i],
            )
        )
    negative_srcs = torch.tensor([[8], [1], [6]])
    negative_dsts = torch.tensor([[2], [8], [8]])
    input_nodes = torch.tensor([8, 1, 6, 5, 9, 0, 2, 4])
    compacted_node_pairs = (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    compacted_negative_srcs = torch.tensor([0, 1, 2])
    compacted_negative_dsts = torch.tensor([6, 0, 0])
    labels = torch.tensor([0.0, 1.0, 2.0])
    # Test dglminibatch with all attributes.
    minibatch = gb.MiniBatch(
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
    dgl_minibatch = minibatch.to_dgl()
    expect_result = str(
        """DGLMiniBatch(positive_node_pairs=(tensor([0, 1, 2]), tensor([3, 4, 5])),
             output_nodes=None,
             node_features={'x': tensor([7, 6, 2, 2])},
             negative_node_pairs=(tensor([0, 1, 2]), tensor([6, 0, 0])),
             labels=tensor([0., 1., 2.]),
             input_nodes=None,
             edge_features=[{'x': tensor([[8],
                                          [1],
                                          [6]])},
                            {'x': tensor([[2],
                                          [8],
                                          [8]])}],
             blocks=[Block(num_src_nodes=4,
                           num_dst_nodes=4,
                           num_edges=6),
                     Block(num_src_nodes=3,
                           num_dst_nodes=2,
                           num_edges=3)],
          )"""
    )
    result = str(dgl_minibatch)
    assert result == expect_result, print(result)


def check_dgl_blocks_hetero(minibatch, blocks):
    etype = gb.etype_str_to_tuple(relation)
    node_pairs = [
        subgraph.node_pairs for subgraph in minibatch.sampled_subgraphs
    ]
    original_edge_ids = [
        subgraph.original_edge_ids for subgraph in minibatch.sampled_subgraphs
    ]
    original_row_node_ids = [
        subgraph.original_row_node_ids
        for subgraph in minibatch.sampled_subgraphs
    ]

    for i, block in enumerate(blocks):
        edges = block.edges(etype=etype)
        assert torch.equal(edges[0], node_pairs[i][relation][0])
        assert torch.equal(edges[1], node_pairs[i][relation][1])
        assert torch.equal(
            block.edges[etype].data[dgl.EID], original_edge_ids[i][relation]
        )
    edges = blocks[0].edges(etype=gb.etype_str_to_tuple(reverse_relation))
    assert torch.equal(edges[0], node_pairs[0][reverse_relation][0])
    assert torch.equal(edges[1], node_pairs[0][reverse_relation][1])
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["A"], original_row_node_ids[0]["A"]
    )
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["B"], original_row_node_ids[0]["B"]
    )


def check_dgl_blocks_homo(minibatch, blocks):
    node_pairs = [
        subgraph.node_pairs for subgraph in minibatch.sampled_subgraphs
    ]
    original_edge_ids = [
        subgraph.original_edge_ids for subgraph in minibatch.sampled_subgraphs
    ]
    original_row_node_ids = [
        subgraph.original_row_node_ids
        for subgraph in minibatch.sampled_subgraphs
    ]
    for i, block in enumerate(blocks):
        assert torch.equal(block.edges()[0], node_pairs[i][0])
        assert torch.equal(block.edges()[1], node_pairs[i][1])
        assert torch.equal(block.edata[dgl.EID], original_edge_ids[i])
    assert torch.equal(blocks[0].srcdata[dgl.NID], original_row_node_ids[0])


def test_to_dgl_node_classification_without_feature():
    # Arrange
    minibatch = create_homo_minibatch()
    minibatch.node_features = None
    minibatch.labels = None
    minibatch.seed_nodes = torch.tensor([10, 15])
    # Act
    dgl_minibatch = minibatch.to_dgl()

    # Assert
    assert len(dgl_minibatch.blocks) == 2
    assert dgl_minibatch.node_features is None
    assert minibatch.edge_features is dgl_minibatch.edge_features
    assert dgl_minibatch.labels is None
    assert minibatch.input_nodes is dgl_minibatch.input_nodes
    assert minibatch.seed_nodes is dgl_minibatch.output_nodes
    check_dgl_blocks_homo(minibatch, dgl_minibatch.blocks)


def test_to_dgl_node_classification_homo():
    # Arrange
    minibatch = create_homo_minibatch()
    minibatch.seed_nodes = torch.tensor([10, 15])
    minibatch.labels = torch.tensor([2, 5])
    # Act
    dgl_minibatch = minibatch.to_dgl()

    # Assert
    assert len(dgl_minibatch.blocks) == 2
    assert minibatch.node_features is dgl_minibatch.node_features
    assert minibatch.edge_features is dgl_minibatch.edge_features
    assert minibatch.labels is dgl_minibatch.labels
    assert dgl_minibatch.input_nodes is None
    assert dgl_minibatch.output_nodes is None
    check_dgl_blocks_homo(minibatch, dgl_minibatch.blocks)


def test_to_dgl_node_classification_hetero():
    minibatch = create_hetero_minibatch()
    minibatch.labels = {"B": torch.tensor([2, 5])}
    minibatch.seed_nodes = {"B": torch.tensor([10, 15])}
    dgl_minibatch = minibatch.to_dgl()

    # Assert
    assert len(dgl_minibatch.blocks) == 2
    assert minibatch.node_features is dgl_minibatch.node_features
    assert minibatch.edge_features is dgl_minibatch.edge_features
    assert minibatch.labels is dgl_minibatch.labels
    assert dgl_minibatch.input_nodes is None
    assert dgl_minibatch.output_nodes is None
    check_dgl_blocks_hetero(minibatch, dgl_minibatch.blocks)


@pytest.mark.parametrize("mode", ["neg_graph", "neg_src", "neg_dst"])
def test_to_dgl_link_predication_homo(mode):
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
    dgl_minibatch = minibatch.to_dgl()

    # Assert
    assert len(dgl_minibatch.blocks) == 2
    assert minibatch.node_features is dgl_minibatch.node_features
    assert minibatch.edge_features is dgl_minibatch.edge_features
    assert minibatch.compacted_node_pairs is dgl_minibatch.positive_node_pairs
    check_dgl_blocks_homo(minibatch, dgl_minibatch.blocks)
    if mode == "neg_graph" or mode == "neg_src":
        assert torch.equal(
            dgl_minibatch.negative_node_pairs[0],
            minibatch.compacted_negative_srcs.view(-1),
        )
    if mode == "neg_graph" or mode == "neg_dst":
        assert torch.equal(
            dgl_minibatch.negative_node_pairs[1],
            minibatch.compacted_negative_dsts.view(-1),
        )


@pytest.mark.parametrize("mode", ["neg_graph", "neg_src", "neg_dst"])
def test_to_dgl_link_predication_hetero(mode):
    # Arrange
    minibatch = create_hetero_minibatch()
    minibatch.compacted_node_pairs = {
        relation: (
            torch.tensor([1, 1]),
            torch.tensor([1, 0]),
        ),
        reverse_relation: (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        ),
    }
    if mode == "neg_graph" or mode == "neg_src":
        minibatch.compacted_negative_srcs = {
            relation: torch.tensor([[2, 0], [1, 2]]),
            reverse_relation: torch.tensor([[1, 2], [0, 2]]),
        }
    if mode == "neg_graph" or mode == "neg_dst":
        minibatch.compacted_negative_dsts = {
            relation: torch.tensor([[1, 3], [2, 1]]),
            reverse_relation: torch.tensor([[2, 1], [3, 1]]),
        }
    # Act
    dgl_minibatch = minibatch.to_dgl()

    # Assert
    assert len(dgl_minibatch.blocks) == 2
    assert minibatch.node_features is dgl_minibatch.node_features
    assert minibatch.edge_features is dgl_minibatch.edge_features
    assert minibatch.compacted_node_pairs is dgl_minibatch.positive_node_pairs
    check_dgl_blocks_hetero(minibatch, dgl_minibatch.blocks)
    if mode == "neg_graph" or mode == "neg_src":
        for etype, src in minibatch.compacted_negative_srcs.items():
            assert torch.equal(
                dgl_minibatch.negative_node_pairs[etype][0],
                src.view(-1),
            )
    if mode == "neg_graph" or mode == "neg_dst":
        for etype, dst in minibatch.compacted_negative_dsts.items():
            assert torch.equal(
                dgl_minibatch.negative_node_pairs[etype][1],
                minibatch.compacted_negative_dsts[etype].view(-1),
            )
