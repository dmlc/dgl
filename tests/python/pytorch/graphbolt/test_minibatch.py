import dgl
import dgl.graphbolt as gb
import pytest
import torch


relation = "A:r:B"
reverse_relation = "B:rr:A"


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_minibatch_representation_homo(indptr_dtype, indices_dtype):
    csc_formats = [
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 3, 5, 6], dtype=indptr_dtype),
            indices=torch.tensor([0, 1, 2, 2, 1, 2], dtype=indices_dtype),
        ),
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 3], dtype=indptr_dtype),
            indices=torch.tensor([1, 2, 0], dtype=indices_dtype),
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
                sampled_csc=csc_formats[i],
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
        """MiniBatch(seeds=None,
          seed_nodes=None,
          sampled_subgraphs=None,
          positive_node_pairs=None,
          node_pairs_with_labels=None,
          node_pairs=None,
          node_features=None,
          negative_srcs=None,
          negative_node_pairs=None,
          negative_dsts=None,
          labels=None,
          input_nodes=None,
          indexes=None,
          edge_features=None,
          compacted_seeds=None,
          compacted_node_pairs=None,
          compacted_negative_srcs=None,
          compacted_negative_dsts=None,
          blocks=None,
       )"""
    )
    result = str(minibatch)
    assert result == expect_result, print(expect_result, result)
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
        """MiniBatch(seeds=None,
          seed_nodes=None,
          sampled_subgraphs=[SampledSubgraphImpl(sampled_csc=CSCFormatBase(indptr=tensor([0, 1, 3, 5, 6], dtype=torch.int32),
                                                                         indices=tensor([0, 1, 2, 2, 1, 2], dtype=torch.int32),
                                                           ),
                                               original_row_node_ids=tensor([10, 11, 12, 13]),
                                               original_edge_ids=tensor([19, 20, 21, 22, 25, 30]),
                                               original_column_node_ids=tensor([10, 11, 12, 13]),
                            ),
                            SampledSubgraphImpl(sampled_csc=CSCFormatBase(indptr=tensor([0, 2, 3], dtype=torch.int32),
                                                                         indices=tensor([1, 2, 0], dtype=torch.int32),
                                                           ),
                                               original_row_node_ids=tensor([10, 11, 12]),
                                               original_edge_ids=tensor([10, 15, 17]),
                                               original_column_node_ids=tensor([10, 11]),
                            )],
          positive_node_pairs=CSCFormatBase(indptr=tensor([0, 2, 3]),
                                            indices=tensor([3, 4, 5]),
                              ),
          node_pairs_with_labels=(CSCFormatBase(indptr=tensor([0, 2, 3]),
                                               indices=tensor([3, 4, 5]),
                                 ),
                                 tensor([0., 1., 2.])),
          node_pairs=[CSCFormatBase(indptr=tensor([0, 1, 3, 5, 6], dtype=torch.int32),
                                   indices=tensor([0, 1, 2, 2, 1, 2], dtype=torch.int32),
                     ),
                     CSCFormatBase(indptr=tensor([0, 2, 3], dtype=torch.int32),
                                   indices=tensor([1, 2, 0], dtype=torch.int32),
                     )],
          node_features={'x': tensor([5, 0, 2, 1])},
          negative_srcs=tensor([[8],
                                [1],
                                [6]]),
          negative_node_pairs=(tensor([[0],
                                      [1],
                                      [2]]),
                              tensor([[6],
                                      [0],
                                      [0]])),
          negative_dsts=tensor([[2],
                                [8],
                                [8]]),
          labels=tensor([0., 1., 2.]),
          input_nodes=tensor([8, 1, 6, 5, 9, 0, 2, 4]),
          indexes=None,
          edge_features=[{'x': tensor([9, 0, 1, 1, 7, 4])},
                        {'x': tensor([0, 2, 2])}],
          compacted_seeds=None,
          compacted_node_pairs=CSCFormatBase(indptr=tensor([0, 2, 3]),
                                             indices=tensor([3, 4, 5]),
                               ),
          compacted_negative_srcs=tensor([[0],
                                          [1],
                                          [2]]),
          compacted_negative_dsts=tensor([[6],
                                          [0],
                                          [0]]),
          blocks=[Block(num_src_nodes=4, num_dst_nodes=4, num_edges=6),
                 Block(num_src_nodes=3, num_dst_nodes=2, num_edges=3)],
       )"""
    )
    result = str(minibatch)
    assert result == expect_result, print(expect_result, result)


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_minibatch_representation_hetero(indptr_dtype, indices_dtype):
    csc_formats = [
        {
            relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 1, 2, 3], dtype=indptr_dtype),
                indices=torch.tensor([0, 1, 1], dtype=indices_dtype),
            ),
            reverse_relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 0, 0, 1, 2], dtype=indptr_dtype),
                indices=torch.tensor([1, 0], dtype=indices_dtype),
            ),
        },
        {
            relation: gb.CSCFormatBase(
                indptr=torch.tensor([0, 1, 2], dtype=indptr_dtype),
                indices=torch.tensor([1, 0], dtype=indices_dtype),
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
                sampled_csc=csc_formats[i],
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
    # Test minibatch with all attributes.
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
        """MiniBatch(seeds=None,
          seed_nodes={'B': tensor([10, 15])},
          sampled_subgraphs=[SampledSubgraphImpl(sampled_csc={'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3], dtype=torch.int32),
                                                                         indices=tensor([0, 1, 1], dtype=torch.int32),
                                                           ), 'B:rr:A': CSCFormatBase(indptr=tensor([0, 0, 0, 1, 2], dtype=torch.int32),
                                                                         indices=tensor([1, 0], dtype=torch.int32),
                                                           )},
                                               original_row_node_ids={'A': tensor([ 5,  7,  9, 11]), 'B': tensor([10, 11, 12])},
                                               original_edge_ids={'A:r:B': tensor([19, 20, 21]), 'B:rr:A': tensor([23, 26])},
                                               original_column_node_ids={'B': tensor([10, 11, 12]), 'A': tensor([ 5,  7,  9, 11])},
                            ),
                            SampledSubgraphImpl(sampled_csc={'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2], dtype=torch.int32),
                                                                         indices=tensor([1, 0], dtype=torch.int32),
                                                           )},
                                               original_row_node_ids={'A': tensor([5, 7]), 'B': tensor([10, 11])},
                                               original_edge_ids={'A:r:B': tensor([10, 12])},
                                               original_column_node_ids={'B': tensor([10, 11])},
                            )],
          positive_node_pairs={'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3]),
                                            indices=tensor([3, 4, 5]),
                              ), 'B:rr:A': CSCFormatBase(indptr=tensor([0, 0, 0, 1, 2]),
                                            indices=tensor([0, 1]),
                              )},
          node_pairs_with_labels=({'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3]),
                                               indices=tensor([3, 4, 5]),
                                 ), 'B:rr:A': CSCFormatBase(indptr=tensor([0, 0, 0, 1, 2]),
                                               indices=tensor([0, 1]),
                                 )},
                                 {'B': tensor([2, 5])}),
          node_pairs=[{'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3], dtype=torch.int32),
                                   indices=tensor([0, 1, 1], dtype=torch.int32),
                     ), 'B:rr:A': CSCFormatBase(indptr=tensor([0, 0, 0, 1, 2], dtype=torch.int32),
                                   indices=tensor([1, 0], dtype=torch.int32),
                     )},
                     {'A:r:B': CSCFormatBase(indptr=tensor([0, 1, 2], dtype=torch.int32),
                                   indices=tensor([1, 0], dtype=torch.int32),
                     )}],
          node_features={('A', 'x'): tensor([6, 4, 0, 1])},
          negative_srcs={'B': tensor([[8],
                                [1],
                                [6]])},
          negative_node_pairs={'A:r:B': (tensor([[0],
                                      [1],
                                      [2]]), tensor([[6],
                                      [0],
                                      [0]]))},
          negative_dsts={'B': tensor([[2],
                                [8],
                                [8]])},
          labels={'B': tensor([2, 5])},
          input_nodes={'A': tensor([ 5,  7,  9, 11]), 'B': tensor([10, 11, 12])},
          indexes=None,
          edge_features=[{('A:r:B', 'x'): tensor([4, 2, 4])},
                        {('A:r:B', 'x'): tensor([0, 6])}],
          compacted_seeds=None,
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
          blocks=[Block(num_src_nodes={'A': 4, 'B': 3},
                       num_dst_nodes={'A': 4, 'B': 3},
                       num_edges={('A', 'r', 'B'): 3, ('B', 'rr', 'A'): 2},
                       metagraph=[('A', 'B', 'r'), ('B', 'A', 'rr')]),
                 Block(num_src_nodes={'A': 2, 'B': 2},
                       num_dst_nodes={'B': 2},
                       num_edges={('A', 'r', 'B'): 2},
                       metagraph=[('A', 'B', 'r')])],
       )"""
    )
    result = str(minibatch)
    assert result == expect_result, print(expect_result, result)


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_get_dgl_blocks_homo(indptr_dtype, indices_dtype):
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
    csc_formats = [
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 3, 5, 6], dtype=indptr_dtype),
            indices=torch.tensor([0, 1, 2, 2, 1, 2], dtype=indices_dtype),
        ),
        gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 3], dtype=indptr_dtype),
            indices=torch.tensor([0, 1, 2], dtype=indices_dtype),
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
                sampled_csc=csc_formats[i],
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
    dgl_blocks = minibatch.blocks
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
                sampled_csc=csc_formats[i],
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
    # Test minibatch with all attributes.
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
    dgl_blocks = minibatch.blocks
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


@pytest.mark.parametrize(
    "mode", ["neg_graph", "neg_src", "neg_dst", "edge_classification"]
)
def test_minibatch_node_pairs_with_labels(mode):
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
    if mode == "edge_classification":
        minibatch.labels = torch.tensor([0, 1]).long()
    # Act
    node_pairs, labels = minibatch.node_pairs_with_labels

    # Assert
    if mode == "neg_src":
        expect_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 1, 0, 0]),
        )
        expect_labels = torch.tensor([1, 1, 0, 0, 0, 0]).float()
    elif mode != "edge_classification":
        expect_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 0, 0, 1]),
        )
        expect_labels = torch.tensor([1, 1, 0, 0, 0, 0]).float()
    else:
        expect_node_pairs = (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        )
        expect_labels = torch.tensor([0, 1]).long()
    assert torch.equal(node_pairs[0], expect_node_pairs[0])
    assert torch.equal(node_pairs[1], expect_node_pairs[1])
    assert torch.equal(labels, expect_labels)


def create_homo_minibatch():
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
                sampled_csc=csc_formats[i],
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
    sampled_csc = [
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
                sampled_csc=sampled_csc[i],
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


def check_dgl_blocks_hetero(minibatch, blocks):
    etype = gb.etype_str_to_tuple(relation)
    sampled_csc = [
        subgraph.sampled_csc for subgraph in minibatch.sampled_subgraphs
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
            0, len(sampled_csc[i][relation].indptr) - 1
        ).repeat_interleave(sampled_csc[i][relation].indptr.diff())
        assert torch.equal(edges[0], sampled_csc[i][relation].indices)
        assert torch.equal(edges[1], dst_ndoes)
        assert torch.equal(
            block.edges[etype].data[dgl.EID], original_edge_ids[i][relation]
        )
    edges = blocks[0].edges(etype=gb.etype_str_to_tuple(reverse_relation))
    dst_ndoes = torch.arange(
        0, len(sampled_csc[0][reverse_relation].indptr) - 1
    ).repeat_interleave(sampled_csc[0][reverse_relation].indptr.diff())
    assert torch.equal(edges[0], sampled_csc[0][reverse_relation].indices)
    assert torch.equal(edges[1], dst_ndoes)
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["A"], original_row_node_ids[0]["A"]
    )
    assert torch.equal(
        blocks[0].srcdata[dgl.NID]["B"], original_row_node_ids[0]["B"]
    )


def check_dgl_blocks_homo(minibatch, blocks):
    sampled_csc = [
        subgraph.sampled_csc for subgraph in minibatch.sampled_subgraphs
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
            0, len(sampled_csc[i].indptr) - 1
        ).repeat_interleave(sampled_csc[i].indptr.diff())
        assert torch.equal(block.edges()[0], sampled_csc[i].indices), print(
            block.edges()
        )
        assert torch.equal(block.edges()[1], dst_ndoes), print(block.edges())
        assert torch.equal(block.edata[dgl.EID], original_edge_ids[i]), print(
            block.edata[dgl.EID]
        )
    assert torch.equal(
        blocks[0].srcdata[dgl.NID], original_row_node_ids[0]
    ), print(blocks[0].srcdata[dgl.NID])


def test_dgl_node_classification_without_feature():
    # Arrange
    minibatch = create_homo_minibatch()
    minibatch.node_features = None
    minibatch.labels = None
    minibatch.seed_nodes = torch.tensor([10, 15])
    # Act
    dgl_blocks = minibatch.blocks

    # Assert
    assert len(dgl_blocks) == 2
    assert minibatch.node_features is None
    assert minibatch.labels is None
    check_dgl_blocks_homo(minibatch, dgl_blocks)


def test_dgl_node_classification_homo():
    # Arrange
    minibatch = create_homo_minibatch()
    minibatch.seed_nodes = torch.tensor([10, 15])
    minibatch.labels = torch.tensor([2, 5])
    # Act
    dgl_blocks = minibatch.blocks

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_homo(minibatch, dgl_blocks)


def test_dgl_node_classification_hetero():
    minibatch = create_hetero_minibatch()
    minibatch.labels = {"B": torch.tensor([2, 5])}
    minibatch.seed_nodes = {"B": torch.tensor([10, 15])}
    # Act
    dgl_blocks = minibatch.blocks

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
    dgl_blocks = minibatch.blocks

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_homo(minibatch, dgl_blocks)
    if mode == "neg_graph" or mode == "neg_src":
        assert torch.equal(
            minibatch.negative_node_pairs[0],
            minibatch.compacted_negative_srcs,
        )
    if mode == "neg_graph" or mode == "neg_dst":
        assert torch.equal(
            minibatch.negative_node_pairs[1],
            minibatch.compacted_negative_dsts,
        )
    (
        node_pairs,
        labels,
    ) = minibatch.node_pairs_with_labels
    if mode == "neg_src":
        expect_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 1, 0, 0]),
        )
    else:
        expect_node_pairs = (
            torch.tensor([0, 1, 0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 0, 0, 1]),
        )
    expect_labels = torch.tensor([1, 1, 0, 0, 0, 0]).float()
    assert torch.equal(node_pairs[0], expect_node_pairs[0])
    assert torch.equal(node_pairs[1], expect_node_pairs[1])
    assert torch.equal(labels, expect_labels)


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
    dgl_blocks = minibatch.blocks

    # Assert
    assert len(dgl_blocks) == 2
    check_dgl_blocks_hetero(minibatch, dgl_blocks)
    if mode == "neg_graph" or mode == "neg_src":
        for etype, src in minibatch.compacted_negative_srcs.items():
            assert torch.equal(
                minibatch.negative_node_pairs[etype][0],
                src,
            )
    if mode == "neg_graph" or mode == "neg_dst":
        for etype, dst in minibatch.compacted_negative_dsts.items():
            assert torch.equal(
                minibatch.negative_node_pairs[etype][1],
                minibatch.compacted_negative_dsts[etype],
            )


def test_to_pyg_data_original():
    test_minibatch = create_homo_minibatch()
    test_minibatch.seed_nodes = torch.tensor([0, 1])
    test_minibatch.labels = torch.tensor([7, 8])

    expected_edge_index = torch.tensor(
        [[0, 0, 1, 1, 1, 2, 2, 2, 2], [0, 1, 0, 1, 2, 0, 1, 2, 3]]
    )
    expected_node_features = next(iter(test_minibatch.node_features.values()))
    expected_labels = torch.tensor([7, 8])
    expected_batch_size = 2
    expected_n_id = torch.tensor([10, 11, 12, 13])

    pyg_data = test_minibatch.to_pyg_data()
    pyg_data.validate()
    assert torch.equal(pyg_data.edge_index, expected_edge_index)
    assert torch.equal(pyg_data.x, expected_node_features)
    assert torch.equal(pyg_data.y, expected_labels)
    assert pyg_data.batch_size == expected_batch_size
    assert torch.equal(pyg_data.n_id, expected_n_id)

    subgraph = test_minibatch.sampled_subgraphs[0]
    # Test with sampled_csc as None.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=None,
        node_features={"feat": expected_node_features},
        labels=expected_labels,
    )
    pyg_data = test_minibatch.to_pyg_data()
    assert pyg_data.edge_index is None, "Edge index should be none."

    # Test with node_features as None.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features=None,
        labels=expected_labels,
    )
    pyg_data = test_minibatch.to_pyg_data()
    assert pyg_data.x is None, "Node features should be None."

    # Test with labels as None.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features={"feat": expected_node_features},
        labels=None,
    )
    pyg_data = test_minibatch.to_pyg_data()
    assert pyg_data.y is None, "Labels should be None."

    # Test with multiple features.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features={
            "feat": expected_node_features,
            "extra_feat": torch.tensor([[3], [4]]),
        },
        labels=expected_labels,
    )
    try:
        pyg_data = test_minibatch.to_pyg_data()
        assert (
            pyg_data.x is None
        ), "Multiple features case should raise an error."
    except AssertionError as e:
        assert (
            str(e)
            == "`to_pyg_data` only supports single feature homogeneous graph."
        )


def test_to_pyg_data():
    test_minibatch = create_homo_minibatch()
    test_minibatch.seeds = torch.tensor([0, 1])
    test_minibatch.labels = torch.tensor([7, 8])

    expected_edge_index = torch.tensor(
        [[0, 0, 1, 1, 1, 2, 2, 2, 2], [0, 1, 0, 1, 2, 0, 1, 2, 3]]
    )
    expected_node_features = next(iter(test_minibatch.node_features.values()))
    expected_labels = torch.tensor([7, 8])
    expected_batch_size = 2
    expected_n_id = torch.tensor([10, 11, 12, 13])

    pyg_data = test_minibatch.to_pyg_data()
    pyg_data.validate()
    assert torch.equal(pyg_data.edge_index, expected_edge_index)
    assert torch.equal(pyg_data.x, expected_node_features)
    assert torch.equal(pyg_data.y, expected_labels)
    assert pyg_data.batch_size == expected_batch_size
    assert torch.equal(pyg_data.n_id, expected_n_id)

    test_minibatch.seeds = torch.tensor([[0, 1], [2, 3]])
    assert pyg_data.batch_size == expected_batch_size

    test_minibatch.seeds = {"A": torch.tensor([0, 1])}
    assert pyg_data.batch_size == expected_batch_size

    test_minibatch.seeds = {"A": torch.tensor([[0, 1], [2, 3]])}
    assert pyg_data.batch_size == expected_batch_size

    subgraph = test_minibatch.sampled_subgraphs[0]
    # Test with sampled_csc as None.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=None,
        node_features={"feat": expected_node_features},
        labels=expected_labels,
    )
    pyg_data = test_minibatch.to_pyg_data()
    assert pyg_data.edge_index is None, "Edge index should be none."

    # Test with node_features as None.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features=None,
        labels=expected_labels,
    )
    pyg_data = test_minibatch.to_pyg_data()
    assert pyg_data.x is None, "Node features should be None."

    # Test with labels as None.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features={"feat": expected_node_features},
        labels=None,
    )
    pyg_data = test_minibatch.to_pyg_data()
    assert pyg_data.y is None, "Labels should be None."

    # Test with multiple features.
    test_minibatch = gb.MiniBatch(
        sampled_subgraphs=[subgraph],
        node_features={
            "feat": expected_node_features,
            "extra_feat": torch.tensor([[3], [4]]),
        },
        labels=expected_labels,
    )
    try:
        pyg_data = test_minibatch.to_pyg_data()
        assert (
            pyg_data.x is None
        ), "Multiple features case should raise an error."
    except AssertionError as e:
        assert (
            str(e)
            == "`to_pyg_data` only supports single feature homogeneous graph."
        )
