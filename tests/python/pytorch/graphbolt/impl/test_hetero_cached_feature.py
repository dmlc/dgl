import backend as F

import pytest
import torch

from dgl import graphbolt as gb


@pytest.mark.parametrize(
    "cached_feature_type", [gb.cpu_cached_feature, gb.gpu_cached_feature]
)
def test_hetero_cached_feature(cached_feature_type):
    if cached_feature_type == gb.gpu_cached_feature and (
        F._default_context_str != "gpu"
        or torch.cuda.get_device_capability()[0] < 7
    ):
        pytest.skip(
            "GPUCachedFeature tests are available only when testing the GPU backend."
            if F._default_context_str != "gpu"
            else "GPUCachedFeature requires a Volta or later generation NVIDIA GPU."
        )
    device = F.ctx() if cached_feature_type == gb.gpu_cached_feature else None
    pin_memory = cached_feature_type == gb.gpu_cached_feature

    a = {
        ("node", str(i), "feat"): gb.TorchBasedFeature(
            torch.randn([(i + 1) * 10, 5], pin_memory=pin_memory)
        )
        for i in range(75)
    }
    cached_a = cached_feature_type(a, 2**18)

    for i in range(1024):
        etype = i % len(a)
        ids = torch.randint(
            0, (etype + 1) * 10 - 1, ((etype + 1) * 4,), device=device
        )
        feature_key = ("node", str(etype), "feat")
        ref = a[feature_key].read(ids)
        val = cached_a[feature_key].read(ids)
        torch.testing.assert_close(ref, val, rtol=0, atol=0)
    assert cached_a[feature_key].miss_rate < 0.69
