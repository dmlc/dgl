import re

import dgl
import pytest
import torch
from dgl import graphbolt as gb
from torch.testing import assert_close


def test_ItemSet_names():
    # ItemSet with single name.
    item_set = gb.ItemSet(torch.arange(0, 5), names="seed_nodes")
    assert item_set.names == ("seed_nodes",)

    # ItemSet with multiple names.
    item_set = gb.ItemSet(
        (torch.arange(0, 5), torch.arange(5, 10)),
        names=("seed_nodes", "labels"),
    )
    assert item_set.names == ("seed_nodes", "labels")

    # ItemSet without name.
    item_set = gb.ItemSet(torch.arange(0, 5))
    assert item_set.names is None

    # ItemSet with mismatched items and names.
    with pytest.raises(
        AssertionError,
        match=re.escape("Number of items (1) and names (2) must match."),
    ):
        _ = gb.ItemSet(torch.arange(0, 5), names=("seed_nodes", "labels"))


def test_ItemSet_length():
    # Single iterable with valid length.
    ids = torch.arange(0, 5)
    item_set = gb.ItemSet(ids)
    assert len(item_set) == 5
    # Test __iter__ method. Same as below.
    for i, item in enumerate(item_set):
        assert i == item.item()

    # Tuple of iterables with valid length.
    item_set = gb.ItemSet((torch.arange(0, 5), torch.arange(5, 10)))
    assert len(item_set) == 5
    for i, (item1, item2) in enumerate(item_set):
        assert i == item1.item()
        assert i + 5 == item2.item()

    class InvalidLength:
        def __iter__(self):
            return iter([0, 1, 2])

    # Single iterable with invalid length.
    item_set = gb.ItemSet(InvalidLength())
    with pytest.raises(TypeError):
        _ = len(item_set)
    for i, item in enumerate(item_set):
        assert i == item

    # Tuple of iterables with invalid length.
    item_set = gb.ItemSet((InvalidLength(), InvalidLength()))
    with pytest.raises(TypeError):
        _ = len(item_set)
    for i, (item1, item2) in enumerate(item_set):
        assert i == item1
        assert i == item2


def test_ItemSet_iteration_seed_nodes():
    # Node IDs.
    item_set = gb.ItemSet(torch.arange(0, 5), names="seed_nodes")
    assert item_set.names == ("seed_nodes",)
    for i, item in enumerate(item_set):
        assert i == item.item()


def test_ItemSet_iteration_seed_nodes_labels():
    # Node IDs and labels.
    seed_nodes = torch.arange(0, 5)
    labels = torch.randint(0, 3, (5,))
    item_set = gb.ItemSet((seed_nodes, labels), names=("seed_nodes", "labels"))
    assert item_set.names == ("seed_nodes", "labels")
    for i, (seed_node, label) in enumerate(item_set):
        assert seed_node == seed_nodes[i]
        assert label == labels[i]


def test_ItemSet_iteration_node_pairs():
    # Node pairs.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    item_set = gb.ItemSet(node_pairs, names="node_pairs")
    assert item_set.names == ("node_pairs",)
    for i, (src, dst) in enumerate(item_set):
        assert node_pairs[i][0] == src
        assert node_pairs[i][1] == dst


def test_ItemSet_iteration_node_pairs_labels():
    # Node pairs and labels
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    labels = torch.randint(0, 3, (5,))
    item_set = gb.ItemSet((node_pairs, labels), names=("node_pairs", "labels"))
    assert item_set.names == ("node_pairs", "labels")
    for i, (node_pair, label) in enumerate(item_set):
        assert torch.equal(node_pairs[i], node_pair)
        assert labels[i] == label


def test_ItemSet_iteration_node_pairs_neg_dsts():
    # Node pairs and negative destinations.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    neg_dsts = torch.arange(10, 25).reshape(-1, 3)
    item_set = gb.ItemSet(
        (node_pairs, neg_dsts), names=("node_pairs", "negative_dsts")
    )
    assert item_set.names == ("node_pairs", "negative_dsts")
    for i, (node_pair, neg_dst) in enumerate(item_set):
        assert torch.equal(node_pairs[i], node_pair)
        assert torch.equal(neg_dsts[i], neg_dst)


def test_ItemSet_iteration_graphs():
    # Graphs.
    graphs = [dgl.rand_graph(10, 20) for _ in range(5)]
    item_set = gb.ItemSet(graphs)
    assert item_set.names is None
    for i, item in enumerate(item_set):
        assert graphs[i] == item


def test_ItemSetDict_names():
    # ItemSetDict with single name.
    item_set = gb.ItemSetDict(
        {
            "user": gb.ItemSet(torch.arange(0, 5), names="seed_nodes"),
            "item": gb.ItemSet(torch.arange(5, 10), names="seed_nodes"),
        }
    )
    assert item_set.names == ("seed_nodes",)

    # ItemSetDict with multiple names.
    item_set = gb.ItemSetDict(
        {
            "user": gb.ItemSet(
                (torch.arange(0, 5), torch.arange(5, 10)),
                names=("seed_nodes", "labels"),
            ),
            "item": gb.ItemSet(
                (torch.arange(5, 10), torch.arange(10, 15)),
                names=("seed_nodes", "labels"),
            ),
        }
    )
    assert item_set.names == ("seed_nodes", "labels")

    # ItemSetDict with no name.
    item_set = gb.ItemSetDict(
        {
            "user": gb.ItemSet(torch.arange(0, 5)),
            "item": gb.ItemSet(torch.arange(5, 10)),
        }
    )
    assert item_set.names is None

    # ItemSetDict with mismatched items and names.
    with pytest.raises(
        AssertionError,
        match=re.escape("All itemsets must have the same names."),
    ):
        _ = gb.ItemSetDict(
            {
                "user": gb.ItemSet(
                    (torch.arange(0, 5), torch.arange(5, 10)),
                    names=("seed_nodes", "labels"),
                ),
                "item": gb.ItemSet(
                    (torch.arange(5, 10),), names=("seed_nodes",)
                ),
            }
        )


def test_ItemSetDict_length():
    # Single iterable with valid length.
    user_ids = torch.arange(0, 5)
    item_ids = torch.arange(0, 5)
    item_set = gb.ItemSetDict(
        {
            "user": gb.ItemSet(user_ids),
            "item": gb.ItemSet(item_ids),
        }
    )
    assert len(item_set) == len(user_ids) + len(item_ids)

    # Tuple of iterables with valid length.
    node_pairs_like = torch.arange(0, 10).reshape(-1, 2)
    neg_dsts_like = torch.arange(10, 20).reshape(-1, 2)
    node_pairs_follow = torch.arange(0, 10).reshape(-1, 2)
    neg_dsts_follow = torch.arange(10, 20).reshape(-1, 2)
    item_set = gb.ItemSetDict(
        {
            "user:like:item": gb.ItemSet((node_pairs_like, neg_dsts_like)),
            "user:follow:user": gb.ItemSet(
                (node_pairs_follow, neg_dsts_follow)
            ),
        }
    )
    assert len(item_set) == node_pairs_like.size(0) + node_pairs_follow.size(0)

    class InvalidLength:
        def __iter__(self):
            return iter([0, 1, 2])

    # Single iterable with invalid length.
    item_set = gb.ItemSetDict(
        {
            "user": gb.ItemSet(InvalidLength()),
            "item": gb.ItemSet(InvalidLength()),
        }
    )
    with pytest.raises(TypeError):
        _ = len(item_set)

    # Tuple of iterables with invalid length.
    item_set = gb.ItemSetDict(
        {
            "user:like:item": gb.ItemSet((InvalidLength(), InvalidLength())),
            "user:follow:user": gb.ItemSet((InvalidLength(), InvalidLength())),
        }
    )
    with pytest.raises(TypeError):
        _ = len(item_set)


def test_ItemSetDict_iteration_seed_nodes():
    # Node IDs.
    user_ids = torch.arange(0, 5)
    item_ids = torch.arange(5, 10)
    ids = {
        "user": gb.ItemSet(user_ids, names="seed_nodes"),
        "item": gb.ItemSet(item_ids, names="seed_nodes"),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids)
    assert item_set.names == ("seed_nodes",)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert chained_ids[i][0] in item
        assert item[chained_ids[i][0]] == chained_ids[i][1]


def test_ItemSetDict_iteration_seed_nodes_labels():
    # Node IDs and labels.
    user_ids = torch.arange(0, 5)
    user_labels = torch.randint(0, 3, (5,))
    item_ids = torch.arange(5, 10)
    item_labels = torch.randint(0, 3, (5,))
    ids_labels = {
        "user": gb.ItemSet(
            (user_ids, user_labels), names=("seed_nodes", "labels")
        ),
        "item": gb.ItemSet(
            (item_ids, item_labels), names=("seed_nodes", "labels")
        ),
    }
    chained_ids = []
    for key, value in ids_labels.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids_labels)
    assert item_set.names == ("seed_nodes", "labels")
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert chained_ids[i][0] in item
        assert item[chained_ids[i][0]] == chained_ids[i][1]


def test_ItemSetDict_iteration_node_pairs():
    # Node pairs.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    node_pairs_dict = {
        "user:like:item": gb.ItemSet(node_pairs, names="node_pairs"),
        "user:follow:user": gb.ItemSet(node_pairs, names="node_pairs"),
    }
    expected_data = []
    for key, value in node_pairs_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.ItemSetDict(node_pairs_dict)
    assert item_set.names == ("node_pairs",)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert torch.equal(item[expected_data[i][0]], expected_data[i][1])


def test_ItemSetDict_iteration_node_pairs_labels():
    # Node pairs and labels
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    labels = torch.randint(0, 3, (5,))
    node_pairs_labels = {
        "user:like:item": gb.ItemSet(
            (node_pairs, labels), names=("node_pairs", "labels")
        ),
        "user:follow:user": gb.ItemSet(
            (node_pairs, labels), names=("node_pairs", "labels")
        ),
    }
    expected_data = []
    for key, value in node_pairs_labels.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.ItemSetDict(node_pairs_labels)
    assert item_set.names == ("node_pairs", "labels")
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        key, value = expected_data[i]
        assert key in item
        assert torch.equal(item[key][0], value[0])
        assert item[key][1] == value[1]


def test_ItemSetDict_iteration_node_pairs_neg_dsts():
    # Node pairs and negative destinations.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    neg_dsts = torch.arange(10, 25).reshape(-1, 3)
    node_pairs_neg_dsts = {
        "user:like:item": gb.ItemSet(
            (node_pairs, neg_dsts), names=("node_pairs", "negative_dsts")
        ),
        "user:follow:user": gb.ItemSet(
            (node_pairs, neg_dsts), names=("node_pairs", "negative_dsts")
        ),
    }
    expected_data = []
    for key, value in node_pairs_neg_dsts.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.ItemSetDict(node_pairs_neg_dsts)
    assert item_set.names == ("node_pairs", "negative_dsts")
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        key, value = expected_data[i]
        assert key in item
        assert torch.equal(item[key][0], value[0])
        assert torch.equal(item[key][1], value[1])
