import backend as F

import pytest
import torch

from dgl import graphbolt as gb


def _test_query_and_replace(policy1, policy2, keys, offset):
    # Testing query_and_replace equivalence to query and then replace.
    (
        _,
        index,
        pointers,
        missing_keys,
        found_offsets,
        missing_offsets,
    ) = policy1.query_and_replace(keys, offset)
    found_cnt = keys.size(0) - missing_keys.size(0)
    found_pointers = pointers[:found_cnt]
    policy1.reading_completed(found_pointers, found_offsets)
    missing_pointers = pointers[found_cnt:]
    policy1.writing_completed(missing_pointers, missing_offsets)

    (
        _,
        index2,
        missing_keys2,
        found_pointers2,
        found_offsets2,
        missing_offsets2,
    ) = policy2.query(keys + offset, 0)
    policy2.reading_completed(found_pointers2, found_offsets2)
    (_, missing_pointers2, missing_offsets2) = policy2.replace(
        missing_keys2, missing_offsets2, 0
    )
    policy2.writing_completed(missing_pointers2, missing_offsets2)

    assert torch.equal(index, index2)
    assert torch.equal(missing_keys, missing_keys2 - offset)


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
@pytest.mark.parametrize("offset", [0, 1111111])
def test_feature_cache(offsets, dtype, feature_size, num_parts, policy, offset):
    cache_size = 32 * (
        torch.get_num_threads() if num_parts is None else num_parts
    )
    a = torch.randint(0, 2, [1024, feature_size], dtype=dtype)
    cache = gb.impl.CPUFeatureCache(
        (cache_size,) + a.shape[1:], a.dtype, policy, num_parts
    )
    cache2 = gb.impl.CPUFeatureCache(
        (cache_size,) + a.shape[1:], a.dtype, policy, num_parts
    )
    policy1 = gb.impl.CPUFeatureCache(
        (cache_size,) + a.shape[1:], a.dtype, policy, num_parts
    )._policy
    policy2 = gb.impl.CPUFeatureCache(
        (cache_size,) + a.shape[1:], a.dtype, policy, num_parts
    )._policy
    reader_fn = lambda keys: a[keys]

    keys = torch.tensor([0, 1])
    values, missing_index, missing_keys, missing_offsets = cache.query(
        keys, offset
    )
    if not offsets:
        missing_offsets = None
    assert torch.equal(
        missing_keys.flip([0]) if num_parts == 1 else missing_keys.sort()[0],
        keys,
    )

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets, offset)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])
    assert torch.equal(
        cache2.query_and_replace(keys, reader_fn, offset), a[keys]
    )

    _test_query_and_replace(policy1, policy2, keys, offset)

    pin_memory = F._default_context_str == "gpu"

    keys = torch.arange(1, 33, pin_memory=pin_memory)
    values, missing_index, missing_keys, missing_offsets = cache.query(
        keys, offset
    )
    if not offsets:
        missing_offsets = None
    assert torch.equal(
        missing_keys.flip([0]) if num_parts == 1 else missing_keys.sort()[0],
        torch.arange(2, 33),
    )
    assert not pin_memory or values.is_pinned()

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets, offset)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])
    assert torch.equal(
        cache2.query_and_replace(keys, reader_fn, offset), a[keys]
    )

    _test_query_and_replace(policy1, policy2, keys, offset)

    values, missing_index, missing_keys, missing_offsets = cache.query(
        keys, offset
    )
    if not offsets:
        missing_offsets = None
    assert torch.equal(missing_keys.flip([0]), torch.tensor([]))

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets, offset)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])
    assert torch.equal(
        cache2.query_and_replace(keys, reader_fn, offset), a[keys]
    )

    _test_query_and_replace(policy1, policy2, keys, offset)

    values, missing_index, missing_keys, missing_offsets = cache.query(
        keys, offset
    )
    if not offsets:
        missing_offsets = None
    assert torch.equal(missing_keys.flip([0]), torch.tensor([]))

    missing_values = a[missing_keys]
    cache.replace(missing_keys, missing_values, missing_offsets, offset)
    values[missing_index] = missing_values
    assert torch.equal(values, a[keys])
    assert torch.equal(
        cache2.query_and_replace(keys, reader_fn, offset), a[keys]
    )

    _test_query_and_replace(policy1, policy2, keys, offset)

    assert cache.miss_rate == cache2.miss_rate

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
