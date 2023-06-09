import dgl
import pytest
import torch
from dgl import graphbolt as gb
from torch.testing import assert_close


def test_mismatch_size_in_tuple():
    # Size mismatch.
    node_pairs = (torch.arange(0, 5), torch.arange(5, 11))
    with pytest.raises(AssertionError):
        _ = gb.ItemSet(node_pairs)


def test_ItemSet_node_edge_ids():
    # Node or edge IDs.
    item_set = gb.ItemSet(torch.arange(0, 5))
    for i, item in enumerate(item_set):
        assert i == item.item()


def test_ItemSet_graphs():
    # Graphs.
    graphs = [dgl.rand_graph(10, 20) for _ in range(5)]
    item_set = gb.ItemSet(graphs)
    for i, item in enumerate(item_set):
        assert graphs[i] == item


def test_ItemSet_node_pairs():
    # Node pairs.
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    item_set = gb.ItemSet(node_pairs)
    for i, (src, dst) in enumerate(item_set):
        assert node_pairs[0][i] == src
        assert node_pairs[1][i] == dst


def test_ItemSet_node_pairs_labels():
    # Node pairs and labels
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    labels = torch.randint(0, 3, (5,))
    item_set = gb.ItemSet((node_pairs[0], node_pairs[1], labels))
    for i, (src, dst, label) in enumerate(item_set):
        assert node_pairs[0][i] == src
        assert node_pairs[1][i] == dst
        assert labels[i] == label


def test_ItemSet_head_tail_neg_tails():
    # Head, tail and negative tails.
    heads = torch.arange(0, 5)
    tails = torch.arange(5, 10)
    neg_tails = torch.arange(10, 20).reshape(5, 2)
    item_set = gb.ItemSet((heads, tails, neg_tails))
    for i, (head, tail, negs) in enumerate(item_set):
        assert heads[i] == head
        assert tails[i] == tail
        assert_close(neg_tails[i], negs)


def test_ItemSetDict_node_edge_ids():
    # Node or edge IDs
    ids = {
        ("user", "like", "item"): gb.ItemSet(torch.arange(0, 5)),
        ("user", "follow", "user"): gb.ItemSet(torch.arange(0, 5)),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = gb.ItemSetDict(ids)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert chained_ids[i][0] in item
        assert item[chained_ids[i][0]] == chained_ids[i][1]


def test_ItemSetDict_node_pairs():
    # Node pairs.
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    node_pairs_dict = {
        ("user", "like", "item"): gb.ItemSet(node_pairs),
        ("user", "follow", "user"): gb.ItemSet(node_pairs),
    }
    expected_data = []
    for key, value in node_pairs_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.ItemSetDict(node_pairs_dict)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert item[expected_data[i][0]] == expected_data[i][1]


def test_ItemSetDict_node_pairs_labels():
    # Node pairs and labels
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    labels = torch.randint(0, 3, (5,))
    node_pairs_dict = {
        ("user", "like", "item"): gb.ItemSet(
            (node_pairs[0], node_pairs[1], labels)
        ),
        ("user", "follow", "user"): gb.ItemSet(
            (node_pairs[0], node_pairs[1], labels)
        ),
    }
    expected_data = []
    for key, value in node_pairs_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.ItemSetDict(node_pairs_dict)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert item[expected_data[i][0]] == expected_data[i][1]


def test_ItemSetDict_head_tail_neg_tails():
    # Head, tail and negative tails.
    heads = torch.arange(0, 5)
    tails = torch.arange(5, 10)
    neg_tails = torch.arange(10, 20).reshape(5, 2)
    item_set = gb.ItemSet((heads, tails, neg_tails))
    data_dict = {
        ("user", "like", "item"): gb.ItemSet((heads, tails, neg_tails)),
        ("user", "follow", "user"): gb.ItemSet((heads, tails, neg_tails)),
    }
    expected_data = []
    for key, value in data_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = gb.ItemSetDict(data_dict)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert_close(item[expected_data[i][0]], expected_data[i][1])
