import os
import re
import unittest
from collections.abc import Iterable, Mapping

import backend as F

import dgl.graphbolt as gb
import pytest
import torch
from torch.torch_version import TorchVersion

from . import gb_test_utils


def test_pytorch_cuda_allocator_conf():
    env = os.getenv("PYTORCH_CUDA_ALLOC_CONF")
    assert env is not None
    config_list = env.split(",")
    assert "expandable_segments:True" in config_list


@unittest.skipIf(F._default_context_str != "gpu", "CopyTo needs GPU to test")
@pytest.mark.parametrize("non_blocking", [False, True])
def test_CopyTo(non_blocking):
    item_sampler = gb.ItemSampler(
        gb.ItemSet(torch.arange(20), names="seeds"), 4
    )
    if non_blocking:
        item_sampler = item_sampler.transform(lambda x: x.pin_memory())

    # Invoke CopyTo via class constructor.
    dp = gb.CopyTo(item_sampler, "cuda")
    for data in dp:
        assert data.seeds.device.type == "cuda"

    dp = gb.CopyTo(item_sampler, "cuda", non_blocking)
    for data in dp:
        assert data.seeds.device.type == "cuda"

    # Invoke CopyTo via functional form.
    dp = item_sampler.copy_to("cuda", non_blocking)
    for data in dp:
        assert data.seeds.device.type == "cuda"


@pytest.mark.parametrize(
    "task",
    [
        "node_classification",
        "node_inference",
        "link_prediction",
        "edge_classification",
    ],
)
@unittest.skipIf(F._default_context_str == "cpu", "CopyTo needs GPU to test")
def test_CopyToWithMiniBatches(task):
    N = 16
    B = 2
    if task == "node_classification":
        itemset = gb.ItemSet(
            (torch.arange(N), torch.arange(N)), names=("seeds", "labels")
        )
    elif task == "node_inference":
        itemset = gb.ItemSet(torch.arange(N), names="seeds")
    elif task == "link_prediction":
        itemset = gb.ItemSet(
            (
                torch.arange(2 * N).reshape(-1, 2),
                torch.arange(N),
            ),
            names=("seeds", "labels"),
        )
    elif task == "edge_classification":
        itemset = gb.ItemSet(
            (torch.arange(2 * N).reshape(-1, 2), torch.arange(N)),
            names=("seeds", "labels"),
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

    copied_attrs = [
        "labels",
        "compacted_seeds",
        "sampled_subgraphs",
        "indexes",
        "node_features",
        "edge_features",
        "blocks",
        "seeds",
        "input_nodes",
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
                    if attr in copied_attrs:
                        assert var.device.type == "cuda", attr
                    else:
                        assert var.device.type == "cpu", attr

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


def test_seed_type_str_to_ntypes():
    """Convert etype from string to tuple."""
    # Test for node pairs.
    seed_type_str = "user:like:item"
    seed_size = 2
    node_type = gb.seed_type_str_to_ntypes(seed_type_str, seed_size)
    assert node_type == ["user", "item"]

    # Test for node pairs.
    seed_type_str = "user:item:user"
    seed_size = 3
    node_type = gb.seed_type_str_to_ntypes(seed_type_str, seed_size)
    assert node_type == ["user", "item", "user"]

    # Test for unexpected input: list.
    seed_type_str = ["user", "item"]
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Passed-in seed type should be string, but got <class 'list'>"
        ),
    ):
        _ = gb.seed_type_str_to_ntypes(seed_type_str, 2)


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
    if pinned:
        gb_result = gb.index_select(tensor.cpu(), index.cpu().pin_memory())
        assert torch.equal(torch_result.cpu(), gb_result)
        assert gb_result.is_pinned()

    # Test the internal async API
    future = torch.ops.graphbolt.index_select_async(tensor.cpu(), index.cpu())
    assert torch.equal(torch_result.cpu(), future.wait())


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
def test_scatter_async(dtype, idtype):
    input = torch.tensor([[2, 3], [5, 5], [20, 13]], dtype=dtype)
    index = torch.ones([1], dtype=idtype)
    res = torch.ops.graphbolt.scatter_async(input, index, input[2:3])
    assert torch.equal(
        torch.tensor([[2, 3], [20, 13], [20, 13]], dtype=dtype), res.wait()
    )


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

    if TorchVersion(torch.__version__) >= TorchVersion("2.2.0a0"):
        import torch._dynamo as dynamo
        from torch.testing._internal.optests import opcheck

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

            explanation = dynamo.explain(gb.expand_indptr)(
                indptr, dtype, nodes, output_size
            )
            expected_breaks = -1 if output_size is None else 0
            assert explanation.graph_break_count == expected_breaks


@unittest.skipIf(
    F._default_context_str != "gpu", "Only GPU implementation is available."
)
@pytest.mark.parametrize("offset", [None, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_indptr_edge_ids(offset, dtype):
    indptr = torch.tensor([0, 2, 2, 7, 10, 12], device=F.ctx())
    if offset:
        offset = indptr[:-1]
        ref_result = torch.arange(
            0, indptr[-1].item(), dtype=dtype, device=F.ctx()
        )
    else:
        ref_result = torch.tensor(
            [0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 0, 1], dtype=dtype, device=F.ctx()
        )
    gb_result = gb.indptr_edge_ids(indptr, dtype, offset)
    assert torch.equal(ref_result, gb_result)
    gb_result = gb.indptr_edge_ids(indptr, dtype, offset, indptr[-1].item())
    assert torch.equal(ref_result, gb_result)

    if TorchVersion(torch.__version__) >= TorchVersion("2.2.0a0"):
        import torch._dynamo as dynamo
        from torch.testing._internal.optests import opcheck

        # Tests torch.compile compatibility
        for output_size in [None, indptr[-1].item()]:
            kwargs = {"offset": offset, "output_size": output_size}
            opcheck(
                torch.ops.graphbolt.indptr_edge_ids,
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

            explanation = dynamo.explain(gb.indptr_edge_ids)(
                indptr, dtype, offset, output_size
            )
            expected_breaks = -1 if output_size is None else 0
            assert explanation.graph_break_count == expected_breaks


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
