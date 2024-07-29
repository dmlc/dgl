import backend as F

import pytest
import torch

from dgl import graphbolt as gb


@pytest.mark.parametrize("offsets", [False, True])
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
@pytest.mark.parametrize("feature_size", [2, 16])
@pytest.mark.parametrize("num_parts", [1, 2, None])
@pytest.mark.parametrize("policy", ["s3-fifo", "sieve", "lru", "clock"])
def test_feature_cache(offsets, dtype, feature_size, num_parts, policy):
    cache_size = 32 * (
        torch.get_num_threads() if num_parts is None else num_parts
    )
    a = torch.randint(0, 2, [1024, feature_size], dtype=dtype)
    cache = gb.impl.CPUFeatureCache(
        (cache_size,) + a.shape[1:], a.dtype, policy, num_parts
    )

    keys = torch.tensor([0, 1])
    values, missing_index, missing_keys, missing_offsets = cache.query(keys)
    if not offsets:
        missing_offsets = None
    assert torch.equal(
        missing_keys.flip([0]) if num_parts == 1 else missing_keys.sort()[0],
        keys,
    )

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])

    pin_memory = F._default_context_str == "gpu"

    keys = torch.arange(1, 33, pin_memory=pin_memory)
    values, missing_index, missing_keys, missing_offsets = cache.query(keys)
    if not offsets:
        missing_offsets = None
    assert torch.equal(
        missing_keys.flip([0]) if num_parts == 1 else missing_keys.sort()[0],
        torch.arange(2, 33),
    )
    assert not pin_memory or values.is_pinned()

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])

    values, missing_index, missing_keys, missing_offsets = cache.query(keys)
    if not offsets:
        missing_offsets = None
    assert torch.equal(missing_keys.flip([0]), torch.tensor([]))

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])

    values, missing_index, missing_keys, missing_offsets = cache.query(keys)
    if not offsets:
        missing_offsets = None
    assert torch.equal(missing_keys.flip([0]), torch.tensor([]))

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])

    raw_feature_cache = torch.ops.graphbolt.feature_cache(
        (cache_size,) + a.shape[1:], a.dtype, pin_memory
    )
    idx = torch.tensor([0, 1, 2])
    raw_feature_cache.replace(idx, a[idx])
    val = raw_feature_cache.index_select(idx)
    assert torch.equal(val, a[idx])
    if pin_memory:
        val = raw_feature_cache.index_select(idx.to(F.ctx()))
        assert torch.equal(val, a[idx].to(F.ctx()))
