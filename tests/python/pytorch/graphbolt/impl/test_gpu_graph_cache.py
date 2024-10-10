import unittest

import backend as F

import dgl.graphbolt as gb

import pytest
import torch


@unittest.skipIf(
    F._default_context_str != "gpu"
    or torch.cuda.get_device_capability()[0] < 7,
    reason="GPUCachedFeature tests are available only when testing the GPU backend."
    if F._default_context_str != "gpu"
    else "GPUCachedFeature requires a Volta or later generation NVIDIA GPU.",
)
@pytest.mark.parametrize(
    "indptr_dtype",
    [
        torch.int32,
        torch.int64,
    ],
)
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
@pytest.mark.parametrize("cache_size", [4, 9, 11])
@pytest.mark.parametrize("with_edge_ids", [True, False])
def test_gpu_graph_cache(indptr_dtype, dtype, cache_size, with_edge_ids):
    indices_dtype = torch.int32
    indptr = torch.tensor([0, 3, 6, 10], dtype=indptr_dtype, pin_memory=True)
    indices = torch.arange(0, indptr[-1], dtype=indices_dtype, pin_memory=True)
    probs_or_mask = indices.to(dtype).pin_memory()
    edge_tensors = [indices, probs_or_mask]

    g = gb.GPUGraphCache(
        cache_size,
        2,
        indptr.dtype,
        [e.dtype for e in edge_tensors],
        not with_edge_ids,
    )

    for i in range(10):
        keys = (
            torch.arange(2, dtype=indices_dtype, device=F.ctx()) + i * 2
        ) % (indptr.size(0) - 1)
        missing_keys, replace = g.query(keys)
        (
            missing_indptr,
            missing_edge_tensors,
        ) = torch.ops.graphbolt.index_select_csc_batched(
            indptr, edge_tensors, missing_keys, with_edge_ids, None
        )
        output_indptr, output_edge_tensors = replace(
            missing_indptr, missing_edge_tensors
        )

        (
            reference_indptr,
            reference_edge_tensors,
        ) = torch.ops.graphbolt.index_select_csc_batched(
            indptr, edge_tensors, keys, with_edge_ids, None
        )

        assert torch.equal(output_indptr, reference_indptr)
        assert len(output_edge_tensors) == len(reference_edge_tensors)
        for e, ref in zip(output_edge_tensors, reference_edge_tensors):
            assert torch.equal(e, ref)
