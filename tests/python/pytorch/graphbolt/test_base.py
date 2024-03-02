import re
import unittest
from collections.abc import Iterable, Mapping

import backend as F

import dgl.graphbolt as gb
import pytest
import torch
from torch.testing._internal.optests import opcheck

from . import gb_test_utils


@unittest.skipIf(F._default_context_str == "cpu", "CopyTo needs GPU to test")
def test_CopyTo():
    item_sampler = gb.ItemSampler(
        gb.ItemSet(torch.arange(20), names="seed_nodes"), 4
    )

    # Invoke CopyTo via class constructor.
    dp = gb.CopyTo(item_sampler, "cuda")
    for data in dp:
        assert data.seed_nodes.device.type == "cuda"

    # Invoke CopyTo via functional form.
    dp = item_sampler.copy_to("cuda")
    for data in dp:
        assert data.seed_nodes.device.type == "cuda"


@pytest.mark.parametrize(
    "task",
    [
        "node_classification",
        "node_inference",
        "link_prediction",
        "edge_classification",
        "extra_attrs",
        "other",
    ],
)
@unittest.skipIf(F._default_context_str == "cpu", "CopyTo needs GPU to test")
def test_CopyToWithMiniBatches(task):
    N = 16
    B = 2
    if task == "node_classification" or task == "extra_attrs":
        itemset = gb.ItemSet(
            (torch.arange(N), torch.arange(N)), names=("seed_nodes", "labels")
        )
    elif task == "node_inference":
        itemset = gb.ItemSet(torch.arange(N), names="seed_nodes")
    elif task == "link_prediction":
        itemset = gb.ItemSet(
            (
                torch.arange(2 * N).reshape(-1, 2),
                torch.arange(3 * N).reshape(-1, 3),
            ),
            names=("node_pairs", "negative_dsts"),
        )
    elif task == "edge_classification":
        itemset = gb.ItemSet(
            (torch.arange(2 * N).reshape(-1, 2), torch.arange(N)),
            names=("node_pairs", "labels"),
        )
    else:
        itemset = gb.ItemSet(
            (torch.arange(2 * N).reshape(-1, 2), torch.arange(N)),
            names=("node_pairs", "seed_nodes"),
        )
    graph = gb_test_utils.rand_csc_graph(100, 0.15, bidirection_edge=True)

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
    if task != "node_inference":
        datapipe = gb.FeatureFetcher(
            datapipe,
            feature_store,
            ["a"],
        )

    if task == "node_classification":
        copied_attrs = [
            "node_features",
            "edge_features",
            "sampled_subgraphs",
            "labels",
            "blocks",
        ]
    elif task == "node_inference":
        copied_attrs = [
            "seed_nodes",
            "sampled_subgraphs",
            "blocks",
            "labels",
        ]
    elif task == "link_prediction":
        copied_attrs = [
            "compacted_node_pairs",
            "node_features",
            "edge_features",
            "sampled_subgraphs",
            "compacted_negative_srcs",
            "compacted_negative_dsts",
            "blocks",
            "positive_node_pairs",
            "negative_node_pairs",
            "node_pairs_with_labels",
        ]
    elif task == "edge_classification":
        copied_attrs = [
            "compacted_node_pairs",
            "node_features",
            "edge_features",
            "sampled_subgraphs",
            "labels",
            "blocks",
            "positive_node_pairs",
            "negative_node_pairs",
            "node_pairs_with_labels",
        ]
    elif task == "extra_attrs":
        copied_attrs = [
            "node_features",
            "edge_features",
            "sampled_subgraphs",
            "labels",
            "blocks",
            "seed_nodes",
        ]

    def test_data_device(datapipe):
        for data in datapipe:
            for attr in dir(data):
                var = getattr(data, attr)
                if isinstance(var, Mapping):
                    var = var[next(iter(var))]
                elif isinstance(var, Iterable):
                    var = next(iter(var))
                if (
                    not callable(var)
                    and not attr.startswith("__")
                    and hasattr(var, "device")
                    and var is not None
                ):
                    if task == "other":
                        assert var.device.type == "cuda"
                    else:
                        if attr in copied_attrs:
                            assert var.device.type == "cuda"
                        else:
                            assert var.device.type == "cpu"

    if task == "extra_attrs":
        extra_attrs = ["seed_nodes"]
    else:
        extra_attrs = None

    # Invoke CopyTo via class constructor.
    test_data_device(gb.CopyTo(datapipe, "cuda", extra_attrs))

    # Invoke CopyTo via functional form.
    test_data_device(datapipe.copy_to("cuda", extra_attrs))


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
    elements = torch.tensor([2, 3, 5, 5, 20, 13, 11], device=F.ctx())
    test_elements = torch.tensor([2, 5], device=F.ctx())
    res = gb.isin(elements, test_elements)
    expected = torch.tensor(
        [True, False, True, True, False, False, False], device=F.ctx()
    )
    assert torch.equal(res, expected)


def test_isin_big_data():
    elements = torch.randint(0, 10000, (10000000,), device=F.ctx())
    test_elements = torch.randint(0, 10000, (500000,), device=F.ctx())
    res = gb.isin(elements, test_elements)
    expected = torch.isin(elements, test_elements)
    assert torch.equal(res, expected)


def test_isin_non_1D_dim():
    elements = torch.tensor([[2, 3], [5, 5], [20, 13]], device=F.ctx())
    test_elements = torch.tensor([2, 5], device=F.ctx())
    with pytest.raises(Exception):
        gb.isin(elements, test_elements)
    elements = torch.tensor([2, 3, 5, 5, 20, 13], device=F.ctx())
    test_elements = torch.tensor([[2, 5]], device=F.ctx())
    with pytest.raises(Exception):
        gb.isin(elements, test_elements)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pinned", [False, True])
def test_index_select(dtype, idtype, pinned):
    if F._default_context_str != "gpu" and pinned:
        pytest.skip("Pinned tests are available only on GPU.")
    tensor = torch.tensor([[2, 3], [5, 5], [20, 13]], dtype=dtype)
    tensor = tensor.pin_memory() if pinned else tensor.to(F.ctx())
    index = torch.tensor([0, 2], dtype=idtype, device=F.ctx())
    gb_result = gb.index_select(tensor, index)
    torch_result = tensor.to(F.ctx())[index.long()]
    assert torch.equal(torch_result, gb_result)


def torch_expand_indptr(indptr, dtype, nodes=None):
    if nodes is None:
        nodes = torch.arange(len(indptr) - 1, dtype=dtype, device=indptr.device)
    return nodes.to(dtype).repeat_interleave(indptr.diff())


@pytest.mark.parametrize("nodes", [None, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_expand_indptr(nodes, dtype):
    if nodes:
        nodes = torch.tensor([1, 7, 3, 4, 5, 8], dtype=dtype, device=F.ctx())
    indptr = torch.tensor([0, 2, 2, 7, 10, 12, 20], device=F.ctx())
    torch_result = torch_expand_indptr(indptr, dtype, nodes)
    gb_result = gb.expand_indptr(indptr, dtype, nodes)
    assert torch.equal(torch_result, gb_result)
    gb_result = gb.expand_indptr(indptr, dtype, nodes, indptr[-1].item())
    assert torch.equal(torch_result, gb_result)

    # Tests torch.compile compatibility
    for output_size in [None, indptr[-1].item()]:
        kwargs = {"node_ids": nodes, "output_size": output_size}
        opcheck(
            torch.ops.graphbolt.expand_indptr,
            (indptr, dtype),
            kwargs,
            test_utils=[
                "test_schema",
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ],
            raise_exception=True,
        )


def test_csc_format_base_representation():
    csc_format_base = gb.CSCFormatBase(
        indptr=torch.tensor([0, 2, 4]),
        indices=torch.tensor([4, 5, 6, 7]),
    )
    expected_result = str(
        """CSCFormatBase(indptr=tensor([0, 2, 4]),
              indices=tensor([4, 5, 6, 7]),
)"""
    )
    assert str(csc_format_base) == expected_result, print(csc_format_base)


def test_csc_format_base_incorrect_indptr():
    indptr = torch.tensor([0, 2, 4, 6, 7, 11])
    indices = torch.tensor([2, 3, 1, 4, 5, 2, 5, 1, 4, 4])
    with pytest.raises(AssertionError):
        # The value of last element in indptr is not corresponding to indices.
        csc_formats = gb.CSCFormatBase(indptr=indptr, indices=indices)
