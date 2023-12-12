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
            gb.FusedSampledSubgraphImpl(
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
            gb.FusedSampledSubgraphImpl(
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


def test_minibatch_representation_homo():
    csc_formats = [
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 3, 5, 6]),
            indices=torch.tensor([0, 1, 2, 2, 1, 2]),
        ),
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 3]),
            indices=torch.tensor([1, 2, 0]),
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
    node_features = {"x": torch.tensor([5, 0, 2, 1])}
    edge_features = [
        {"x": torch.tensor([9, 0, 1, 1, 7, 4])},
        {"x": torch.tensor([0, 2, 2])},
    ]
    subgraphs = []
    for i in range(2):
        subgraphs.append(
            gb.SampledSubgraphImpl(
                node_pairs=csc_formats[i],
                original_column_node_ids=original_column_node_ids[i],
                original_row_node_ids=original_row_node_ids[i],
                original_edge_ids=original_edge_ids[i],
            )
        )
    negative_srcs = torch.tensor([[8], [1], [6]])
    negative_dsts = torch.tensor([[2], [8], [8]])
    input_nodes = torch.tensor([8, 1, 6, 5, 9, 0, 2, 4])
    compacted_csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 2, 3]), indices=torch.tensor([3, 4, 5])
    )
    compacted_negative_srcs = torch.tensor([[0], [1], [2]])
    compacted_negative_dsts = torch.tensor([[6], [0], [0]])
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
        node_pairs=csc_formats,
        sampled_subgraphs=subgraphs,
        labels=labels,
        node_features=node_features,
        edge_features=edge_features,
        negative_srcs=negative_srcs,
        negative_dsts=negative_dsts,
        compacted_node_pairs=compacted_csc_formats,
        input_nodes=input_nodes,
        compacted_negative_srcs=compacted_negative_srcs,
        compacted_negative_dsts=compacted_negative_dsts,
    )
    expect_result = str(
        """MiniBatch(seed_nodes=None,
          sampled_subgraphs=[SampledSubgraphImpl(original_row_node_ids=tensor([10, 11, 12, 13]),
                                               original_edge_ids=tensor([19, 20, 21, 22, 25, 30]),
                                               original_column_node_ids=tensor([10, 11, 12, 13]),
                                               node_pairs=CSCFormatBase(indptr=tensor([0, 1, 3, 5, 6]),
                                                                        indices=tensor([0, 1, 2, 2, 1, 2]),
                                                          ),
                            ),
                            SampledSubgraphImpl(original_row_node_ids=tensor([10, 11, 12]),
                                               original_edge_ids=tensor([10, 15, 17]),
                                               original_column_node_ids=tensor([10, 11]),
                                               node_pairs=CSCFormatBase(indptr=tensor([0, 2, 3]),
                                                                        indices=tensor([1, 2, 0]),
                                                          ),
                            )],
          node_pairs=[CSCFormatBase(indptr=tensor([0, 1, 3, 5, 6]),
                                   indices=tensor([0, 1, 2, 2, 1, 2]),
                     ),
                     CSCFormatBase(indptr=tensor([0, 2, 3]),
                                   indices=tensor([1, 2, 0]),
                     )],
          node_features={'x': tensor([5, 0, 2, 1])},
          negative_srcs=tensor([[8],
                                [1],
                                [6]]),
          negative_dsts=tensor([[2],
                                [8],
                                [8]]),
          labels=tensor([0., 1., 2.]),
          input_nodes=tensor([8, 1, 6, 5, 9, 0, 2, 4]),
          edge_features=[{'x': tensor([9, 0, 1, 1, 7, 4])},
                        {'x': tensor([0, 2, 2])}],
          compacted_node_pairs=CSCFormatBase(indptr=tensor([0, 2, 3]),
                                             indices=tensor([3, 4, 5]),
                               ),
          compacted_negative_srcs=tensor([[0],
                                          [1],
                                          [2]]),
          compacted_negative_dsts=tensor([[6],
                                          [0],
                                          [0]]),
       )"""
    )
    result = str(minibatch)
    assert result == expect_result, print(expect_result, result)


def test_minibatch_representation_hetero():
    csc_formats = [
        {
            relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 1, 2, 3]),
                indices=torch.tensor([0, 1, 1]),
            ),
            reverse_relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 0, 0, 1, 2]),
                indices=torch.tensor([1, 0]),
            ),
        },
        {
            relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 1, 2]), indices=torch.tensor([1, 0])
            )
        },
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
        ("A", "x"): torch.tensor([6, 4, 0, 1]),
    }
    edge_features = [
        {(relation, "x"): torch.tensor([4, 2, 4])},
        {(relation, "x"): torch.tensor([0, 6])},
    ]
    subgraphs = []
    for i in range(2):
        subgraphs.append(
            gb.SampledSubgraphImpl(
                node_pairs=csc_formats[i],
                original_column_node_ids=original_column_node_ids[i],
                original_row_node_ids=original_row_node_ids[i],
                original_edge_ids=original_edge_ids[i],
            )
        )
    negative_srcs = {"B": torch.tensor([[8], [1], [6]])}
    negative_dsts = {"B": torch.tensor([[2], [8], [8]])}
    compacted_csc_formats = {
        relation: gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 2, 3]), indices=torch.tensor([3, 4, 5])
        ),
        reverse_relation: gb.CSCFormatBase(
            indptr=torch.tensor([0, 0, 0, 1, 2]), indices=torch.tensor([0, 1])
        ),
    }
    compacted_negative_srcs = {relation: torch.tensor([[0], [1], [2]])}
    compacted_negative_dsts = {relation: torch.tensor([[6], [0], [0]])}
    # Test dglminibatch with all attributes.
    minibatch = gb.MiniBatch(
        seed_nodes={"B": torch.tensor([10, 15])},
        node_pairs=csc_formats,
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
        labels={"B": torch.tensor([2, 5])},
        negative_srcs=negative_srcs,
        negative_dsts=negative_dsts,
        compacted_node_pairs=compacted_csc_formats,
        input_nodes={
            "A": torch.tensor([5, 7, 9, 11]),
            "B": torch.tensor([10, 11, 12]),
        },
        compacted_negative_srcs=compacted_negative_srcs,
        compacted_negative_dsts=compacted_negative_dsts,
    )
    expect_result = str(
        """MiniBatch(seed_nodes={'B': tensor([10, 15])},
          sampled_subgraphs=[SampledSubgraphImpl(original_row_node_ids={'A': tensor([ 5,  7,  9, 11]), 'B': tensor([10, 11, 12])},
                                               original_edge_ids={'A:r:B': tensor([19, 20, 21]), 'B:rr:A': tensor([23, 26])},
                                               original_column_node_ids={'B': tensor([10, 11, 12]), 'A': tensor([ 5,  7,  9, 11])},
                                               node_pairs={'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3]),
                                                                        indices=tensor([0, 1, 1]),
                                                          ), 'B:rr:A': CSCFormatBase(indptr=tensor([0, 0, 0, 1, 2]),
                                                                        indices=tensor([1, 0]),
                                                          )},
                            ),
                            SampledSubgraphImpl(original_row_node_ids={'A': tensor([5, 7]), 'B': tensor([10, 11])},
                                               original_edge_ids={'A:r:B': tensor([10, 12])},
                                               original_column_node_ids={'B': tensor([10, 11])},
                                               node_pairs={'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2]),
                                                                        indices=tensor([1, 0]),
                                                          )},
                            )],
          node_pairs=[{'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3]),
                                   indices=tensor([0, 1, 1]),
                     ), 'B:rr:A': CSCFormatBase(indptr=tensor([0, 0, 0, 1, 2]),
                                   indices=tensor([1, 0]),
                     )},
                     {'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2]),
                                   indices=tensor([1, 0]),
                     )}],
          node_features={('A', 'x'): tensor([6, 4, 0, 1])},
          negative_srcs={'B': tensor([[8],
                                [1],
                                [6]])},
          negative_dsts={'B': tensor([[2],
                                [8],
                                [8]])},
          labels={'B': tensor([2, 5])},
          input_nodes={'A': tensor([ 5,  7,  9, 11]), 'B': tensor([10, 11, 12])},
          edge_features=[{('A:r:B', 'x'): tensor([4, 2, 4])},
                        {('A:r:B', 'x'): tensor([0, 6])}],
          compacted_node_pairs={'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3]),
                                             indices=tensor([3, 4, 5]),
                               ), 'B:rr:A': CSCFormatBase(indptr=tensor([0, 0, 0, 1, 2]),
                                             indices=tensor([0, 1]),
                               )},
          compacted_negative_srcs={'A:r:B': tensor([[0],
                                          [1],
                                          [2]])},
          compacted_negative_dsts={'A:r:B': tensor([[6],
                                          [0],
                                          [0]])},
       )"""
    )
    result = str(minibatch)
    assert result == expect_result, print(result)


def test_get_dgl_blocks_homo():
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
            gb.FusedSampledSubgraphImpl(
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
    compacted_negative_srcs = torch.tensor([[0], [1], [2]])
    compacted_negative_dsts = torch.tensor([[6], [0], [0]])
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
    dgl_blocks = minibatch.get_dgl_blocks()
    expect_result = str(
        """[Block(num_src_nodes=4, num_dst_nodes=4, num_edges=6), Block(num_src_nodes=3, num_dst_nodes=2, num_edges=3)]"""
    )
    result = str(dgl_blocks)
    assert result == expect_result, print(result)


def test_get_dgl_blocks_hetero():
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
        ("A", "x"): torch.tensor([6, 4, 0, 1]),
    }
    edge_features = [
        {(relation, "x"): torch.tensor([4, 2, 4])},
        {(relation, "x"): torch.tensor([0, 6])},
    ]
    subgraphs = []
    for i in range(2):
        subgraphs.append(
            gb.FusedSampledSubgraphImpl(
                node_pairs=node_pairs[i],
                original_column_node_ids=original_column_node_ids[i],
                original_row_node_ids=original_row_node_ids[i],
                original_edge_ids=original_edge_ids[i],
            )
        )
    negative_srcs = {"B": torch.tensor([[8], [1], [6]])}
    negative_dsts = {"B": torch.tensor([[2], [8], [8]])}
    compacted_node_pairs = {
        relation: (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])),
        reverse_relation: (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])),
    }
    compacted_negative_srcs = {relation: torch.tensor([[0], [1], [2]])}
    compacted_negative_dsts = {relation: torch.tensor([[6], [0], [0]])}
    # Test dglminibatch with all attributes.
    minibatch = gb.MiniBatch(
        seed_nodes={"B": torch.tensor([10, 15])},
        node_pairs=node_pairs,
        sampled_subgraphs=subgraphs,
        node_features=node_features,
        edge_features=edge_features,
        labels={"B": torch.tensor([2, 5])},
        negative_srcs=negative_srcs,
        negative_dsts=negative_dsts,
        compacted_node_pairs=compacted_node_pairs,
        input_nodes={
            "A": torch.tensor([5, 7, 9, 11]),
            "B": torch.tensor([10, 11, 12]),
        },
        compacted_negative_srcs=compacted_negative_srcs,
        compacted_negative_dsts=compacted_negative_dsts,
    )
    dgl_blocks = minibatch.get_dgl_blocks()
    expect_result = str(
        """[Block(num_src_nodes={'A': 4, 'B': 3},
      num_dst_nodes={'A': 4, 'B': 3},
      num_edges={('A', 'r', 'B'): 3, ('B', 'rr', 'A'): 2},
      metagraph=[('A', 'B', 'r'), ('B', 'A', 'rr')]), Block(num_src_nodes={'A': 2, 'B': 2},
      num_dst_nodes={'B': 2},
      num_edges={('A', 'r', 'B'): 2},
      metagraph=[('A', 'B', 'r')])]"""
    )
    result = str(dgl_blocks)
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


def test_get_dgl_blocks_node_classification_without_feature():
    # Arrange
    minibatch = create_homo_minibatch()
    minibatch.node_features = None
    minibatch.labels = None
    minibatch.seed_nodes = torch.tensor([10, 15])
    # Act
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    assert minibatch.node_features is None
    assert minibatch.labels is None
    check_dgl_blocks_homo(minibatch, dgl_blocks)


def test_get_dgl_blocks_node_classification_homo():
    # Arrange
    minibatch = create_homo_minibatch()
    minibatch.seed_nodes = torch.tensor([10, 15])
    minibatch.labels = torch.tensor([2, 5])
    # Act
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_homo(minibatch, dgl_blocks)


def test_to_dgl_node_classification_hetero():
    minibatch = create_hetero_minibatch()
    minibatch.labels = {"B": torch.tensor([2, 5])}
    minibatch.seed_nodes = {"B": torch.tensor([10, 15])}
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_hetero(minibatch, dgl_blocks)


@pytest.mark.parametrize("mode", ["neg_graph", "neg_src", "neg_dst"])
def test_dgl_link_predication_homo(mode):
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
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_homo(minibatch, dgl_blocks)
    if mode == "neg_graph" or mode == "neg_src":
        assert torch.equal(
            minibatch.get_negative_node_pairs()[0],
            minibatch.compacted_negative_srcs.view(-1),
        )
    if mode == "neg_graph" or mode == "neg_dst":
        assert torch.equal(
            minibatch.get_negative_node_pairs()[1],
            minibatch.compacted_negative_dsts.view(-1),
        )
    (
        training_node_pairs,
        training_labels,
    ) = minibatch.get_training_node_pair_and_labels()
    if mode == "neg_src":
        expect_training_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 1, 0, 0]),
        )
    else:
        expect_training_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 0, 0, 1]),
        )
    expect_training_labels = torch.tensor([1, 1, 0, 0, 0, 0]).float()
    assert torch.equal(training_node_pairs[0], expect_training_node_pairs[0])
    assert torch.equal(training_node_pairs[1], expect_training_node_pairs[1])
    assert torch.equal(training_labels, expect_training_labels)


@pytest.mark.parametrize("mode", ["neg_graph", "neg_src", "neg_dst"])
def test_dgl_link_predication_hetero(mode):
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
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_hetero(minibatch, dgl_blocks)
    if mode == "neg_graph" or mode == "neg_src":
        for etype, src in minibatch.compacted_negative_srcs.items():
            assert torch.equal(
                minibatch.get_negative_node_pairs()[etype][0],
                src.view(-1),
            )
    if mode == "neg_graph" or mode == "neg_dst":
        for etype, dst in minibatch.compacted_negative_dsts.items():
            assert torch.equal(
                minibatch.get_negative_node_pairs()[etype][1],
                minibatch.compacted_negative_dsts[etype].view(-1),
            )


def create_homo_minibatch_csc_format():
    csc_formats = [
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 3, 5, 6]),
            indices=torch.tensor([0, 1, 2, 2, 1, 2]),
        ),
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 3]),
            indices=torch.tensor([1, 2, 0]),
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
                node_pairs=csc_formats[i],
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


def create_hetero_minibatch_csc_format():
    node_pairs = [
        {
            relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 1, 2, 3]),
                indices=torch.tensor([0, 1, 1]),
            ),
            reverse_relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 0, 0, 1, 2]),
                indices=torch.tensor([1, 0]),
            ),
        },
        {
            relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 1, 2]), indices=torch.tensor([1, 0])
            )
        },
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


def check_dgl_blocks_hetero_csc_format(minibatch, blocks):
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
        dst_ndoes = torch.arange(
            0, len(node_pairs[i][relation].indptr) - 1
        ).repeat_interleave(
            node_pairs[i][relation].indptr[1:]
            - node_pairs[i][relation].indptr[:-1]
        )
        assert torch.equal(edges[0], node_pairs[i][relation].indices)
        assert torch.equal(edges[1], dst_ndoes)
        assert torch.equal(
            block.edges[etype].data[dgl.EID], original_edge_ids[i][relation]
        )
    edges = blocks[0].edges(etype=gb.etype_str_to_tuple(reverse_relation))
    dst_ndoes = torch.arange(
        0, len(node_pairs[0][reverse_relation].indptr) - 1
    ).repeat_interleave(
        node_pairs[0][reverse_relation].indptr[1:]
        - node_pairs[0][reverse_relation].indptr[:-1]
    )
    assert torch.equal(edges[0], node_pairs[0][reverse_relation].indices)
    assert torch.equal(edges[1], dst_ndoes)
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["A"], original_row_node_ids[0]["A"]
    )
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["B"], original_row_node_ids[0]["B"]
    )


def check_dgl_blocks_homo_csc_format(minibatch, blocks):
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
        dst_ndoes = torch.arange(
            0, len(node_pairs[i].indptr) - 1
        ).repeat_interleave(
            node_pairs[i].indptr[1:] - node_pairs[i].indptr[:-1]
        )
        assert torch.equal(block.edges()[0], node_pairs[i].indices), print(
            block.edges()
        )
        assert torch.equal(block.edges()[1], dst_ndoes), print(block.edges())
        assert torch.equal(block.edata[dgl.EID], original_edge_ids[i]), print(
            block.edata[dgl.EID]
        )
    assert torch.equal(
        blocks[0].srcdata[dgl.NID], original_row_node_ids[0]
    ), print(blocks[0].srcdata[dgl.NID])


def test_dgl_node_classification_without_feature_csc_format():
    # Arrange
    minibatch = create_homo_minibatch_csc_format()
    minibatch.node_features = None
    minibatch.labels = None
    minibatch.seed_nodes = torch.tensor([10, 15])
    # Act
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    assert minibatch.node_features is None
    assert minibatch.labels is None
    check_dgl_blocks_homo_csc_format(minibatch, dgl_blocks)


def test_dgl_node_classification_homo_csc_format():
    # Arrange
    minibatch = create_homo_minibatch_csc_format()
    minibatch.seed_nodes = torch.tensor([10, 15])
    minibatch.labels = torch.tensor([2, 5])
    # Act
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_homo_csc_format(minibatch, dgl_blocks)


def test_dgl_node_classification_hetero_csc_format():
    minibatch = create_hetero_minibatch_csc_format()
    minibatch.labels = {"B": torch.tensor([2, 5])}
    minibatch.seed_nodes = {"B": torch.tensor([10, 15])}
    # Act
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_hetero_csc_format(minibatch, dgl_blocks)


@pytest.mark.parametrize("mode", ["neg_graph", "neg_src", "neg_dst"])
def test_dgl_link_predication_homo_csc_format(mode):
    # Arrange
    minibatch = create_homo_minibatch_csc_format()
    minibatch.compacted_node_pairs = (
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )
    if mode == "neg_graph" or mode == "neg_src":
        minibatch.compacted_negative_srcs = torch.tensor([[0, 0], [1, 1]])
    if mode == "neg_graph" or mode == "neg_dst":
        minibatch.compacted_negative_dsts = torch.tensor([[1, 0], [0, 1]])
    # Act
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_homo_csc_format(minibatch, dgl_blocks)
    if mode == "neg_graph" or mode == "neg_src":
        assert torch.equal(
            minibatch.get_negative_node_pairs()[0],
            minibatch.compacted_negative_srcs.view(-1),
        )
    if mode == "neg_graph" or mode == "neg_dst":
        assert torch.equal(
            minibatch.get_negative_node_pairs()[1],
            minibatch.compacted_negative_dsts.view(-1),
        )
    (
        training_node_pairs,
        training_labels,
    ) = minibatch.get_training_node_pair_and_labels()
    if mode == "neg_src":
        expect_training_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 1, 0, 0]),
        )
    else:
        expect_training_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 0, 0, 1]),
        )
    expect_training_labels = torch.tensor([1, 1, 0, 0, 0, 0]).float()
    assert torch.equal(training_node_pairs[0], expect_training_node_pairs[0])
    assert torch.equal(training_node_pairs[1], expect_training_node_pairs[1])
    assert torch.equal(training_labels, expect_training_labels)


@pytest.mark.parametrize("mode", ["neg_graph", "neg_src", "neg_dst"])
def test_dgl_link_predication_hetero_csc_format(mode):
    # Arrange
    minibatch = create_hetero_minibatch_csc_format()
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
    dgl_blocks = minibatch.get_dgl_blocks()

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_hetero_csc_format(minibatch, dgl_blocks)
    if mode == "neg_graph" or mode == "neg_src":
        for etype, src in minibatch.compacted_negative_srcs.items():
            assert torch.equal(
                minibatch.get_negative_node_pairs()[etype][0],
                src.view(-1),
            )
    if mode == "neg_graph" or mode == "neg_dst":
        for etype, dst in minibatch.compacted_negative_dsts.items():
            assert torch.equal(
                minibatch.get_negative_node_pairs()[etype][1],
                minibatch.compacted_negative_dsts[etype].view(-1),
            )
