import re

import dgl
import pytest
import torch
from dgl import graphbolt as gb


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

    # Integer-initiated ItemSet with excessive names.
    with pytest.raises(
        AssertionError,
        match=re.escape("Number of items (1) and names (2) must match."),
    ):
        _ = gb.ItemSet(5, names=("seed_nodes", "labels"))

    # ItemSet with mismatched items and names.
    with pytest.raises(
        AssertionError,
        match=re.escape("Number of items (1) and names (2) must match."),
    ):
        _ = gb.ItemSet(torch.arange(0, 5), names=("seed_nodes", "labels"))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ItemSet_scalar_dtype(dtype):
    item_set = gb.ItemSet(torch.tensor(5, dtype=dtype), names="seed_nodes")
    for i, item in enumerate(item_set):
        assert i == item
        assert item.dtype == dtype
    assert item_set[2] == torch.tensor(2, dtype=dtype)
    assert torch.equal(
        item_set[slice(1, 4, 2)], torch.arange(1, 4, 2, dtype=dtype)
    )


def test_ItemSet_length():
    # Integer with valid length
    num = 10
    item_set = gb.ItemSet(num)
    assert len(item_set) == 10
    # Test __iter__() method. Same as below.
    for i, item in enumerate(item_set):
        assert i == item

    # Single iterable with valid length.
    ids = torch.arange(0, 5)
    item_set = gb.ItemSet(ids)
    assert len(item_set) == 5
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
    with pytest.raises(
        TypeError, match="ItemSet instance doesn't have valid length."
    ):
        _ = len(item_set)
    with pytest.raises(
        TypeError, match="ItemSet instance doesn't support indexing."
    ):
        _ = item_set[0]
    for i, item in enumerate(item_set):
        assert i == item

    # Tuple of iterables with invalid length.
    item_set = gb.ItemSet((InvalidLength(), InvalidLength()))
    with pytest.raises(
        TypeError, match="ItemSet instance doesn't have valid length."
    ):
        _ = len(item_set)
    with pytest.raises(
        TypeError, match="ItemSet instance doesn't support indexing."
    ):
        _ = item_set[0]
    for i, (item1, item2) in enumerate(item_set):
        assert i == item1
        assert i == item2


def test_ItemSet_seed_nodes():
    # Node IDs with tensor.
    item_set = gb.ItemSet(torch.arange(0, 5), names="seed_nodes")
    assert item_set.names == ("seed_nodes",)
    # Iterating over ItemSet and indexing one by one.
    for i, item in enumerate(item_set):
        assert i == item.item()
        assert i == item_set[i]
    # Indexing with a slice.
    assert torch.equal(item_set[:], torch.arange(0, 5))
    # Indexing with an Iterable.
    assert torch.equal(item_set[torch.arange(0, 5)], torch.arange(0, 5))

    # Node IDs with single integer.
    item_set = gb.ItemSet(5, names="seed_nodes")
    assert item_set.names == ("seed_nodes",)
    # Iterating over ItemSet and indexing one by one.
    for i, item in enumerate(item_set):
        assert i == item.item()
        assert i == item_set[i]
    # Indexing with a slice.
    assert torch.equal(item_set[:], torch.arange(0, 5))
    # Indexing with an integer.
    assert item_set[0] == 0
    assert item_set[-1] == 4
    # Indexing that is out of range.
    with pytest.raises(IndexError, match="ItemSet index out of range."):
        _ = item_set[5]
    with pytest.raises(IndexError, match="ItemSet index out of range."):
        _ = item_set[-10]
    # Indexing with tensor.
    with pytest.raises(
        TypeError, match="ItemSet indices must be integer or slice."
    ):
        _ = item_set[torch.arange(3)]


def test_ItemSet_seed_nodes_labels():
    # Node IDs and labels.
    seed_nodes = torch.arange(0, 5)
    labels = torch.randint(0, 3, (5,))
    item_set = gb.ItemSet((seed_nodes, labels), names=("seed_nodes", "labels"))
    assert item_set.names == ("seed_nodes", "labels")
    # Iterating over ItemSet and indexing one by one.
    for i, (seed_node, label) in enumerate(item_set):
        assert seed_node == seed_nodes[i]
        assert label == labels[i]
        assert seed_node == item_set[i][0]
        assert label == item_set[i][1]
    # Indexing with a slice.
    assert torch.equal(item_set[:][0], seed_nodes)
    assert torch.equal(item_set[:][1], labels)
    # Indexing with an Iterable.
    assert torch.equal(item_set[torch.arange(0, 5)][0], seed_nodes)
    assert torch.equal(item_set[torch.arange(0, 5)][1], labels)


def test_ItemSet_node_pairs():
    # Node pairs.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    item_set = gb.ItemSet(node_pairs, names="node_pairs")
    assert item_set.names == ("node_pairs",)
    # Iterating over ItemSet and indexing one by one.
    for i, (src, dst) in enumerate(item_set):
        assert node_pairs[i][0] == src
        assert node_pairs[i][1] == dst
        assert node_pairs[i][0] == item_set[i][0]
        assert node_pairs[i][1] == item_set[i][1]
    # Indexing with a slice.
    assert torch.equal(item_set[:], node_pairs)
    # Indexing with an Iterable.
    assert torch.equal(item_set[torch.arange(0, 5)], node_pairs)


def test_ItemSet_node_pairs_labels():
    # Node pairs and labels
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    labels = torch.randint(0, 3, (5,))
    item_set = gb.ItemSet((node_pairs, labels), names=("node_pairs", "labels"))
    assert item_set.names == ("node_pairs", "labels")
    # Iterating over ItemSet and indexing one by one.
    for i, (node_pair, label) in enumerate(item_set):
        assert torch.equal(node_pairs[i], node_pair)
        assert labels[i] == label
        assert torch.equal(node_pairs[i], item_set[i][0])
        assert labels[i] == item_set[i][1]
    # Indexing with a slice.
    assert torch.equal(item_set[:][0], node_pairs)
    assert torch.equal(item_set[:][1], labels)
    # Indexing with an Iterable.
    assert torch.equal(item_set[torch.arange(0, 5)][0], node_pairs)
    assert torch.equal(item_set[torch.arange(0, 5)][1], labels)


def test_ItemSet_node_pairs_neg_dsts():
    # Node pairs and negative destinations.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    neg_dsts = torch.arange(10, 25).reshape(-1, 3)
    item_set = gb.ItemSet(
        (node_pairs, neg_dsts), names=("node_pairs", "negative_dsts")
    )
    assert item_set.names == ("node_pairs", "negative_dsts")
    # Iterating over ItemSet and indexing one by one.
    for i, (node_pair, neg_dst) in enumerate(item_set):
        assert torch.equal(node_pairs[i], node_pair)
        assert torch.equal(neg_dsts[i], neg_dst)
        assert torch.equal(node_pairs[i], item_set[i][0])
        assert torch.equal(neg_dsts[i], item_set[i][1])
    # Indexing with a slice.
    assert torch.equal(item_set[:][0], node_pairs)
    assert torch.equal(item_set[:][1], neg_dsts)
    # Indexing with an Iterable.
    assert torch.equal(item_set[torch.arange(0, 5)][0], node_pairs)
    assert torch.equal(item_set[torch.arange(0, 5)][1], neg_dsts)


def test_ItemSet_graphs():
    # Graphs.
    graphs = [dgl.rand_graph(10, 20) for _ in range(5)]
    item_set = gb.ItemSet(graphs)
    assert item_set.names is None
    # Iterating over ItemSet and indexing one by one.
    for i, item in enumerate(item_set):
        assert graphs[i] == item
        assert graphs[i] == item_set[i]
    # Indexing with a slice.
    assert item_set[:] == graphs


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
    with pytest.raises(
        TypeError, match="ItemSet instance doesn't have valid length."
    ):
        _ = len(item_set)
    with pytest.raises(
        TypeError, match="ItemSetDict instance doesn't support indexing."
    ):
        _ = item_set[0]

    # Tuple of iterables with invalid length.
    item_set = gb.ItemSetDict(
        {
            "user:like:item": gb.ItemSet((InvalidLength(), InvalidLength())),
            "user:follow:user": gb.ItemSet((InvalidLength(), InvalidLength())),
        }
    )
    with pytest.raises(
        TypeError, match="ItemSet instance doesn't have valid length."
    ):
        _ = len(item_set)
    with pytest.raises(
        TypeError, match="ItemSetDict instance doesn't support indexing."
    ):
        _ = item_set[0]


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
    # Iterating over ItemSetDict and indexing one by one.
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert chained_ids[i][0] in item
        assert item[chained_ids[i][0]] == chained_ids[i][1]
        assert item_set[i] == item
        assert item_set[i - len(item_set)] == item
    # Indexing all with a slice.
    assert torch.equal(item_set[:]["user"], user_ids)
    assert torch.equal(item_set[:]["item"], item_ids)
    # Indexing partial with a slice.
    partial_data = item_set[:3]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["user"], user_ids[:3])
    partial_data = item_set[7:]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["item"], item_ids[2:])
    partial_data = item_set[3:7]
    assert len(list(partial_data.keys())) == 2
    assert torch.equal(partial_data["user"], user_ids[3:5])
    assert torch.equal(partial_data["item"], item_ids[:2])

    # Exception cases.
    with pytest.raises(AssertionError, match="Step must be 1."):
        _ = item_set[::2]
    with pytest.raises(
        AssertionError, match="Start must be smaller than stop."
    ):
        _ = item_set[5:3]
    with pytest.raises(
        AssertionError, match="Start must be smaller than stop."
    ):
        _ = item_set[-1:3]
    with pytest.raises(IndexError, match="ItemSetDict index out of range."):
        _ = item_set[20]
    with pytest.raises(IndexError, match="ItemSetDict index out of range."):
        _ = item_set[-20]
    with pytest.raises(
        TypeError, match="ItemSetDict indices must be int or slice."
    ):
        _ = item_set[torch.arange(3)]


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
    # Iterating over ItemSetDict and indexing one by one.
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert chained_ids[i][0] in item
        assert item[chained_ids[i][0]] == chained_ids[i][1]
        assert item_set[i] == item
    # Indexing with a slice.
    assert torch.equal(item_set[:]["user"][0], user_ids)
    assert torch.equal(item_set[:]["user"][1], user_labels)
    assert torch.equal(item_set[:]["item"][0], item_ids)
    assert torch.equal(item_set[:]["item"][1], item_labels)


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
    # Iterating over ItemSetDict and indexing one by one.
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert torch.equal(item[expected_data[i][0]], expected_data[i][1])
        assert item_set[i].keys() == item.keys()
        key = list(item.keys())[0]
        assert torch.equal(item_set[i][key], item[key])
    # Indexing with a slice.
    assert torch.equal(item_set[:]["user:like:item"], node_pairs)
    assert torch.equal(item_set[:]["user:follow:user"], node_pairs)


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
    # Iterating over ItemSetDict and indexing one by one.
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        key, value = expected_data[i]
        assert key in item
        assert torch.equal(item[key][0], value[0])
        assert item[key][1] == value[1]
        assert item_set[i].keys() == item.keys()
        key = list(item.keys())[0]
        assert torch.equal(item_set[i][key][0], item[key][0])
        assert torch.equal(item_set[i][key][1], item[key][1])
    # Indexing with a slice.
    assert torch.equal(item_set[:]["user:like:item"][0], node_pairs)
    assert torch.equal(item_set[:]["user:like:item"][1], labels)
    assert torch.equal(item_set[:]["user:follow:user"][0], node_pairs)
    assert torch.equal(item_set[:]["user:follow:user"][1], labels)


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
    # Iterating over ItemSetDict and indexing one by one.
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        key, value = expected_data[i]
        assert key in item
        assert torch.equal(item[key][0], value[0])
        assert torch.equal(item[key][1], value[1])
        assert item_set[i].keys() == item.keys()
        key = list(item.keys())[0]
        assert torch.equal(item_set[i][key][0], item[key][0])
        assert torch.equal(item_set[i][key][1], item[key][1])
    # Indexing with a slice.
    assert torch.equal(item_set[:]["user:like:item"][0], node_pairs)
    assert torch.equal(item_set[:]["user:like:item"][1], neg_dsts)
    assert torch.equal(item_set[:]["user:follow:user"][0], node_pairs)
    assert torch.equal(item_set[:]["user:follow:user"][1], neg_dsts)


def test_ItemSet_repr():
    # ItemSet with single name.
    item_set = gb.ItemSet(torch.arange(0, 5), names="seed_nodes")
    expected_str = (
        "ItemSet(\n"
        "    items=(tensor([0, 1, 2, 3, 4]),),\n"
        "    names=('seed_nodes',),\n"
        ")"
    )

    assert str(item_set) == expected_str, item_set

    # ItemSet with multiple names.
    item_set = gb.ItemSet(
        (torch.arange(0, 5), torch.arange(5, 10)),
        names=("seed_nodes", "labels"),
    )
    expected_str = (
        "ItemSet(\n"
        "    items=(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9])),\n"
        "    names=('seed_nodes', 'labels'),\n"
        ")"
    )
    assert str(item_set) == expected_str, item_set


def test_ItemSetDict_repr():
    # ItemSetDict with single name.
    item_set = gb.ItemSetDict(
        {
            "user": gb.ItemSet(torch.arange(0, 5), names="seed_nodes"),
            "item": gb.ItemSet(torch.arange(5, 10), names="seed_nodes"),
        }
    )
    expected_str = (
        "ItemSetDict(\n"
        "    itemsets={'user': ItemSet(\n"
        "                 items=(tensor([0, 1, 2, 3, 4]),),\n"
        "                 names=('seed_nodes',),\n"
        "             ), 'item': ItemSet(\n"
        "                 items=(tensor([5, 6, 7, 8, 9]),),\n"
        "                 names=('seed_nodes',),\n"
        "             )},\n"
        "    names=('seed_nodes',),\n"
        ")"
    )
    assert str(item_set) == expected_str, item_set

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
    expected_str = (
        "ItemSetDict(\n"
        "    itemsets={'user': ItemSet(\n"
        "                 items=(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9])),\n"
        "                 names=('seed_nodes', 'labels'),\n"
        "             ), 'item': ItemSet(\n"
        "                 items=(tensor([5, 6, 7, 8, 9]), tensor([10, 11, 12, 13, 14])),\n"
        "                 names=('seed_nodes', 'labels'),\n"
        "             )},\n"
        "    names=('seed_nodes', 'labels'),\n"
        ")"
    )
    assert str(item_set) == expected_str, item_set
