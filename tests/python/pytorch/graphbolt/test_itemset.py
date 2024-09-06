import re

import dgl
import pytest
import torch
from dgl import graphbolt as gb


def test_ItemSet_names():
    # ItemSet with single name.
    item_set = gb.ItemSet(torch.arange(0, 5), names="seeds")
    assert item_set.names == ("seeds",)

    # ItemSet with multiple names.
    item_set = gb.ItemSet(
        (torch.arange(0, 5), torch.arange(5, 10)),
        names=("seeds", "labels"),
    )
    assert item_set.names == ("seeds", "labels")

    # ItemSet without name.
    item_set = gb.ItemSet(torch.arange(0, 5))
    assert item_set.names is None

    # Integer-initiated ItemSet with excessive names.
    with pytest.raises(
        AssertionError,
        match=re.escape("Number of items (1) and names (2) don't match."),
    ):
        _ = gb.ItemSet(5, names=("seeds", "labels"))

    # ItemSet with mismatched items and names.
    with pytest.raises(
        AssertionError,
        match=re.escape("Number of items (1) and names (2) don't match."),
    ):
        _ = gb.ItemSet(torch.arange(0, 5), names=("seeds", "labels"))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ItemSet_scalar_dtype(dtype):
    item_set = gb.ItemSet(torch.tensor(5, dtype=dtype), names="seeds")
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
    with pytest.raises(
        TypeError, match="object of type 'InvalidLength' has no len()"
    ):
        item_set = gb.ItemSet(InvalidLength())

    # Tuple of iterables with invalid length.
    with pytest.raises(
        TypeError, match="object of type 'InvalidLength' has no len()"
    ):
        item_set = gb.ItemSet((InvalidLength(), InvalidLength()))


def test_ItemSet_seed_nodes():
    # Node IDs with tensor.
    item_set = gb.ItemSet(torch.arange(0, 5), names="seeds")
    assert item_set.names == ("seeds",)
    # Iterating over ItemSet and indexing one by one.
    for i, item in enumerate(item_set):
        assert i == item.item()
        assert i == item_set[i]
    # Indexing with a slice.
    assert torch.equal(item_set[::2], torch.tensor([0, 2, 4]))
    # Indexing with an Iterable.
    assert torch.equal(item_set[torch.arange(0, 5)], torch.arange(0, 5))

    # Node IDs with single integer.
    item_set = gb.ItemSet(5, names="seeds")
    assert item_set.names == ("seeds",)
    # Iterating over ItemSet and indexing one by one.
    for i, item in enumerate(item_set):
        assert i == item.item()
        assert i == item_set[i]
    # Indexing with a slice.
    assert torch.equal(item_set[::2], torch.tensor([0, 2, 4]))
    assert torch.equal(item_set[torch.arange(0, 5)], torch.arange(0, 5))
    # Indexing with an integer.
    assert item_set[0] == 0
    assert item_set[-1] == 4
    # Indexing that is out of range.
    with pytest.raises(IndexError, match="ItemSet index out of range."):
        _ = item_set[5]
    with pytest.raises(IndexError, match="ItemSet index out of range."):
        _ = item_set[-10]
    # Indexing with invalid input type.
    with pytest.raises(
        TypeError,
        match="ItemSet indices must be int, slice, or torch.Tensor, not <class 'float'>.",
    ):
        _ = item_set[1.5]


def test_ItemSet_seed_nodes_labels():
    # Node IDs and labels.
    seed_nodes = torch.arange(0, 5)
    labels = torch.randint(0, 3, (5,))
    item_set = gb.ItemSet((seed_nodes, labels), names=("seeds", "labels"))
    assert item_set.names == ("seeds", "labels")
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
    item_set = gb.ItemSet(node_pairs, names="seeds")
    assert item_set.names == ("seeds",)
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
    item_set = gb.ItemSet((node_pairs, labels), names=("seeds", "labels"))
    assert item_set.names == ("seeds", "labels")
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


def test_ItemSet_node_pairs_labels_indexes():
    # Node pairs and negative destinations.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    labels = torch.tensor([1, 1, 0, 0, 0])
    indexes = torch.tensor([0, 1, 0, 0, 1])
    item_set = gb.ItemSet(
        (node_pairs, labels, indexes), names=("seeds", "labels", "indexes")
    )
    assert item_set.names == ("seeds", "labels", "indexes")
    # Iterating over ItemSet and indexing one by one.
    for i, (node_pair, label, index) in enumerate(item_set):
        assert torch.equal(node_pairs[i], node_pair)
        assert torch.equal(labels[i], label)
        assert torch.equal(indexes[i], index)
        assert torch.equal(node_pairs[i], item_set[i][0])
        assert torch.equal(labels[i], item_set[i][1])
        assert torch.equal(indexes[i], item_set[i][2])
    # Indexing with a slice.
    assert torch.equal(item_set[:][0], node_pairs)
    assert torch.equal(item_set[:][1], labels)
    assert torch.equal(item_set[:][2], indexes)
    # Indexing with an Iterable.
    assert torch.equal(item_set[torch.arange(0, 5)][0], node_pairs)
    assert torch.equal(item_set[torch.arange(0, 5)][1], labels)
    assert torch.equal(item_set[torch.arange(0, 5)][2], indexes)


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


def test_HeteroItemSet_names():
    # HeteroItemSet with single name.
    item_set = gb.HeteroItemSet(
        {
            "user": gb.ItemSet(torch.arange(0, 5), names="seeds"),
            "item": gb.ItemSet(torch.arange(5, 10), names="seeds"),
        }
    )
    assert item_set.names == ("seeds",)

    # HeteroItemSet with multiple names.
    item_set = gb.HeteroItemSet(
        {
            "user": gb.ItemSet(
                (torch.arange(0, 5), torch.arange(5, 10)),
                names=("seeds", "labels"),
            ),
            "item": gb.ItemSet(
                (torch.arange(5, 10), torch.arange(10, 15)),
                names=("seeds", "labels"),
            ),
        }
    )
    assert item_set.names == ("seeds", "labels")

    # HeteroItemSet with no name.
    item_set = gb.HeteroItemSet(
        {
            "user": gb.ItemSet(torch.arange(0, 5)),
            "item": gb.ItemSet(torch.arange(5, 10)),
        }
    )
    assert item_set.names is None

    # HeteroItemSet with mismatched items and names.
    with pytest.raises(
        AssertionError,
        match=re.escape("All itemsets must have the same names."),
    ):
        _ = gb.HeteroItemSet(
            {
                "user": gb.ItemSet(
                    (torch.arange(0, 5), torch.arange(5, 10)),
                    names=("seeds", "labels"),
                ),
                "item": gb.ItemSet((torch.arange(5, 10),), names=("seeds",)),
            }
        )


def test_HeteroItemSet_length():
    # Single iterable with valid length.
    user_ids = torch.arange(0, 5)
    item_ids = torch.arange(0, 5)
    item_set = gb.HeteroItemSet(
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
    item_set = gb.HeteroItemSet(
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
    with pytest.raises(
        TypeError, match="object of type 'InvalidLength' has no len()"
    ):
        item_set = gb.HeteroItemSet(
            {
                "user": gb.ItemSet(InvalidLength()),
                "item": gb.ItemSet(InvalidLength()),
            }
        )

    # Tuple of iterables with invalid length.
    with pytest.raises(
        TypeError, match="object of type 'InvalidLength' has no len()"
    ):
        item_set = gb.HeteroItemSet(
            {
                "user:like:item": gb.ItemSet(
                    (InvalidLength(), InvalidLength())
                ),
                "user:follow:user": gb.ItemSet(
                    (InvalidLength(), InvalidLength())
                ),
            }
        )


def test_HeteroItemSet_iteration_seed_nodes():
    # Node IDs.
    user_ids = torch.arange(0, 5)
    item_ids = torch.arange(5, 10)
    ids = {
        "user": gb.ItemSet(user_ids, names="seeds"),
        "item": gb.ItemSet(item_ids, names="seeds"),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.HeteroItemSet(ids)
    assert item_set.names == ("seeds",)
    # Iterating over HeteroItemSet and indexing one by one.
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
    partial_data = item_set[3:8:2]
    assert len(list(partial_data.keys())) == 2
    assert torch.equal(partial_data["user"], user_ids[3:-1:2])
    assert torch.equal(partial_data["item"], item_ids[0:3:2])
    # Indexing with an iterable of int.
    partial_data = item_set[torch.tensor([1, 0, 4])]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["user"], torch.tensor([1, 0, 4]))
    partial_data = item_set[torch.tensor([9, 8, 5])]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["item"], torch.tensor([9, 8, 5]))
    partial_data = item_set[torch.tensor([8, 1, 0, 9, 7, 5])]
    assert len(list(partial_data.keys())) == 2
    assert torch.equal(partial_data["user"], torch.tensor([1, 0]))
    assert torch.equal(partial_data["item"], torch.tensor([8, 9, 7, 5]))

    # Exception cases.
    with pytest.raises(
        AssertionError, match="Start must be smaller than stop."
    ):
        _ = item_set[5:3]
    with pytest.raises(
        AssertionError, match="Start must be smaller than stop."
    ):
        _ = item_set[-1:3]
    with pytest.raises(IndexError, match="HeteroItemSet index out of range."):
        _ = item_set[20]
    with pytest.raises(IndexError, match="HeteroItemSet index out of range."):
        _ = item_set[-20]
    with pytest.raises(
        TypeError,
        match="HeteroItemSet indices must be int, slice, or iterable of int, not <class 'float'>.",
    ):
        _ = item_set[1.5]


def test_HeteroItemSet_iteration_seed_nodes_labels():
    # Node IDs and labels.
    user_ids = torch.arange(0, 5)
    user_labels = torch.randint(0, 3, (5,))
    item_ids = torch.arange(5, 10)
    item_labels = torch.randint(0, 3, (5,))
    ids_labels = {
        "user": gb.ItemSet((user_ids, user_labels), names=("seeds", "labels")),
        "item": gb.ItemSet((item_ids, item_labels), names=("seeds", "labels")),
    }
    chained_ids = []
    for key, value in ids_labels.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.HeteroItemSet(ids_labels)
    assert item_set.names == ("seeds", "labels")
    # Iterating over HeteroItemSet and indexing one by one.
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


def test_HeteroItemSet_iteration_node_pairs():
    # Node pairs.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    node_pairs_dict = {
        "user:like:item": gb.ItemSet(node_pairs, names="seeds"),
        "user:follow:user": gb.ItemSet(node_pairs, names="seeds"),
    }
    expected_data = []
    for key, value in node_pairs_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.HeteroItemSet(node_pairs_dict)
    assert item_set.names == ("seeds",)
    # Iterating over HeteroItemSet and indexing one by one.
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


def test_HeteroItemSet_iteration_node_pairs_labels():
    # Node pairs and labels
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    labels = torch.randint(0, 3, (5,))
    node_pairs_labels = {
        "user:like:item": gb.ItemSet(
            (node_pairs, labels), names=("seeds", "labels")
        ),
        "user:follow:user": gb.ItemSet(
            (node_pairs, labels), names=("seeds", "labels")
        ),
    }
    expected_data = []
    for key, value in node_pairs_labels.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.HeteroItemSet(node_pairs_labels)
    assert item_set.names == ("seeds", "labels")
    # Iterating over HeteroItemSet and indexing one by one.
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


def test_HeteroItemSet_iteration_node_pairs_labels_indexes():
    # Node pairs and negative destinations.
    node_pairs = torch.arange(0, 10).reshape(-1, 2)
    labels = torch.tensor([1, 1, 0, 0, 0])
    indexes = torch.tensor([0, 1, 0, 0, 1])
    node_pairs_neg_dsts = {
        "user:like:item": gb.ItemSet(
            (node_pairs, labels, indexes), names=("seeds", "labels", "indexes")
        ),
        "user:follow:user": gb.ItemSet(
            (node_pairs, labels, indexes), names=("seeds", "labels", "indexes")
        ),
    }
    expected_data = []
    for key, value in node_pairs_neg_dsts.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.HeteroItemSet(node_pairs_neg_dsts)
    assert item_set.names == ("seeds", "labels", "indexes")
    # Iterating over HeteroItemSet and indexing one by one.
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        key, value = expected_data[i]
        assert key in item
        assert torch.equal(item[key][0], value[0])
        assert torch.equal(item[key][1], value[1])
        assert torch.equal(item[key][2], value[2])
        assert item_set[i].keys() == item.keys()
        key = list(item.keys())[0]
        assert torch.equal(item_set[i][key][0], item[key][0])
        assert torch.equal(item_set[i][key][1], item[key][1])
        assert torch.equal(item_set[i][key][2], item[key][2])
    # Indexing with a slice.
    assert torch.equal(item_set[:]["user:like:item"][0], node_pairs)
    assert torch.equal(item_set[:]["user:like:item"][1], labels)
    assert torch.equal(item_set[:]["user:like:item"][2], indexes)
    assert torch.equal(item_set[:]["user:follow:user"][0], node_pairs)
    assert torch.equal(item_set[:]["user:follow:user"][1], labels)
    assert torch.equal(item_set[:]["user:follow:user"][2], indexes)


def test_ItemSet_repr():
    # ItemSet with single name.
    item_set = gb.ItemSet(torch.arange(0, 5), names="seeds")
    expected_str = (
        "ItemSet(\n"
        "    items=(tensor([0, 1, 2, 3, 4]),),\n"
        "    names=('seeds',),\n"
        ")"
    )

    assert str(item_set) == expected_str, item_set

    # ItemSet with multiple names.
    item_set = gb.ItemSet(
        (torch.arange(0, 5), torch.arange(5, 10)),
        names=("seeds", "labels"),
    )
    expected_str = (
        "ItemSet(\n"
        "    items=(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9])),\n"
        "    names=('seeds', 'labels'),\n"
        ")"
    )
    assert str(item_set) == expected_str, item_set


def test_HeteroItemSet_repr():
    # HeteroItemSet with single name.
    item_set = gb.HeteroItemSet(
        {
            "user": gb.ItemSet(torch.arange(0, 5), names="seeds"),
            "item": gb.ItemSet(torch.arange(5, 10), names="seeds"),
        }
    )
    expected_str = (
        "HeteroItemSet(\n"
        "    itemsets={'user': ItemSet(\n"
        "                 items=(tensor([0, 1, 2, 3, 4]),),\n"
        "                 names=('seeds',),\n"
        "             ), 'item': ItemSet(\n"
        "                 items=(tensor([5, 6, 7, 8, 9]),),\n"
        "                 names=('seeds',),\n"
        "             )},\n"
        "    names=('seeds',),\n"
        ")"
    )
    assert str(item_set) == expected_str, item_set

    # HeteroItemSet with multiple names.
    item_set = gb.HeteroItemSet(
        {
            "user": gb.ItemSet(
                (torch.arange(0, 5), torch.arange(5, 10)),
                names=("seeds", "labels"),
            ),
            "item": gb.ItemSet(
                (torch.arange(5, 10), torch.arange(10, 15)),
                names=("seeds", "labels"),
            ),
        }
    )
    expected_str = (
        "HeteroItemSet(\n"
        "    itemsets={'user': ItemSet(\n"
        "                 items=(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9])),\n"
        "                 names=('seeds', 'labels'),\n"
        "             ), 'item': ItemSet(\n"
        "                 items=(tensor([5, 6, 7, 8, 9]), tensor([10, 11, 12, 13, 14])),\n"
        "                 names=('seeds', 'labels'),\n"
        "             )},\n"
        "    names=('seeds', 'labels'),\n"
        ")"
    )
    assert str(item_set) == expected_str, item_set


def test_deprecation_alias():
    """Test `ItemSetDict` as the alias for `HeteroItemSet`."""

    user_ids = torch.arange(0, 5)
    item_ids = torch.arange(5, 10)
    ids = {
        "user": gb.ItemSet(user_ids, names="seeds"),
        "item": gb.ItemSet(item_ids, names="seeds"),
    }
    with pytest.warns(
        DeprecationWarning,
        match="ItemSetDict is deprecated and will be removed in the future. Please use HeteroItemSet instead.",
    ):
        item_set_dict = gb.ItemSetDict(ids)
    hetero_item_set = gb.HeteroItemSet(ids)
    assert len(item_set_dict) == len(hetero_item_set)
    assert item_set_dict.names == hetero_item_set.names
    assert item_set_dict._keys == hetero_item_set._keys
    assert torch.equal(item_set_dict._offsets, hetero_item_set._offsets)
    assert (
        repr(item_set_dict)[len("ItemSetDict") :]
        == repr(hetero_item_set)[len("HeteroItemSet") :]
    )
    # Indexing all with a slice.
    assert torch.equal(item_set_dict[:]["user"], hetero_item_set[:]["user"])
    assert torch.equal(item_set_dict[:]["item"], hetero_item_set[:]["item"])
    # Indexing partial with a slice.
    partial_data = item_set_dict[:3]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["user"], hetero_item_set[:3]["user"])
    partial_data = item_set_dict[7:]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["item"], hetero_item_set[7:]["item"])
    partial_data = item_set_dict[3:8:2]
    assert len(list(partial_data.keys())) == 2
    assert torch.equal(partial_data["user"], hetero_item_set[3:8:2]["user"])
    assert torch.equal(partial_data["item"], hetero_item_set[3:8:2]["item"])
    # Indexing with an iterable of int.
    partial_data = item_set_dict[torch.tensor([1, 0, 4])]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["user"], hetero_item_set[1, 0, 4]["user"])
    partial_data = item_set_dict[torch.tensor([9, 8, 5])]
    assert len(list(partial_data.keys())) == 1
    assert torch.equal(partial_data["item"], hetero_item_set[9, 8, 5]["item"])
    partial_data = item_set_dict[torch.tensor([8, 1, 0, 9, 7, 5])]
    assert len(list(partial_data.keys())) == 2
    assert torch.equal(partial_data["user"], hetero_item_set[1, 0]["user"])
    assert torch.equal(
        partial_data["item"], hetero_item_set[8, 9, 7, 5]["item"]
    )
