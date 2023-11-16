import re
import unittest

import backend as F

import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch


@unittest.skipIf(F._default_context_str == "cpu", "CopyTo needs GPU to test")
def test_CopyTo():
    item_sampler = gb.ItemSampler(gb.ItemSet(torch.randn(20)), 4)

    # Invoke CopyTo via class constructor.
    dp = gb.CopyTo(item_sampler, "cuda")
    for data in dp:
        assert data.device.type == "cuda"

    # Invoke CopyTo via functional form.
    dp = item_sampler.copy_to("cuda")
    for data in dp:
        assert data.device.type == "cuda"


@unittest.skipIf(F._default_context_str == "cpu", "CopyTo needs GPU to test")
def test_CopyToWithMiniBatches():
    N = 16
    B = 2
    itemset = gb.ItemSet(torch.arange(N), names="seed_nodes")
    graph = gb_test_utils.rand_csc_graph(100, 0.15)

    features = {}
    keys = [("node", None, "a"), ("node", None, "b")]
    features[keys[0]] = gb.TorchBasedFeature(torch.randn(200, 4))
    features[keys[1]] = gb.TorchBasedFeature(torch.randn(200, 4))
    feature_store = gb.BasicFeatureStore(features)

    datapipe = gb.ItemSampler(itemset, batch_size=B)
    datapipe = gb.NeighborSampler(
        datapipe,
        graph,
        fanouts=[torch.LongTensor([2]) for _ in range(2)],
    )
    datapipe = gb.FeatureFetcher(
        datapipe,
        feature_store,
        ["a"],
    )

    def test_data_device(datapipe):
        for data in datapipe:
            for attr in dir(data):
                var = getattr(data, attr)
                if (
                    not callable(var)
                    and not attr.startswith("__")
                    and hasattr(var, "device")
                ):
                    assert var.device.type == "cuda"

    # Invoke CopyTo via class constructor.
    test_data_device(gb.CopyTo(datapipe, "cuda"))

    # Invoke CopyTo via functional form.
    test_data_device(datapipe.copy_to("cuda"))

    # Test for DGLMiniBatch.
    datapipe = gb.DGLMiniBatchConverter(datapipe)

    # Invoke CopyTo via class constructor.
    test_data_device(gb.CopyTo(datapipe, "cuda"))

    # Invoke CopyTo via functional form.
    test_data_device(datapipe.copy_to("cuda"))


def test_etype_tuple_to_str():
    """Convert etype from tuple to string."""
    # Test for expected input.
    c_etype = ("user", "like", "item")
    c_etype_str = gb.etype_tuple_to_str(c_etype)
    assert c_etype_str == "user:like:item"

    # Test for unexpected input: not a tuple.
    c_etype = "user:like:item"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Passed-in canonical etype should be in format of (str, str, str). "
            "But got user:like:item."
        ),
    ):
        _ = gb.etype_tuple_to_str(c_etype)

    # Test for unexpected input: tuple with wrong length.
    c_etype = ("user", "like")
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Passed-in canonical etype should be in format of (str, str, str). "
            "But got ('user', 'like')."
        ),
    ):
        _ = gb.etype_tuple_to_str(c_etype)


def test_etype_str_to_tuple():
    """Convert etype from string to tuple."""
    # Test for expected input.
    c_etype_str = "user:like:item"
    c_etype = gb.etype_str_to_tuple(c_etype_str)
    assert c_etype == ("user", "like", "item")

    # Test for unexpected input: string with wrong format.
    c_etype_str = "user:like"
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Passed-in canonical etype should be in format of 'str:str:str'. "
            "But got user:like."
        ),
    ):
        _ = gb.etype_str_to_tuple(c_etype_str)


def test_isin():
    elements = torch.tensor([2, 3, 5, 5, 20, 13, 11])
    test_elements = torch.tensor([2, 5])
    res = gb.isin(elements, test_elements)
    expected = torch.tensor([True, False, True, True, False, False, False])
    assert torch.equal(res, expected)


def test_isin_big_data():
    elements = torch.randint(0, 10000, (10000000,))
    test_elements = torch.randint(0, 10000, (500000,))
    res = gb.isin(elements, test_elements)
    expected = torch.isin(elements, test_elements)
    assert torch.equal(res, expected)


def test_isin_non_1D_dim():
    elements = torch.tensor([[2, 3], [5, 5], [20, 13]])
    test_elements = torch.tensor([2, 5])
    with pytest.raises(Exception):
        gb.isin(elements, test_elements)
    elements = torch.tensor([2, 3, 5, 5, 20, 13])
    test_elements = torch.tensor([[2, 5]])
    with pytest.raises(Exception):
        gb.isin(elements, test_elements)
