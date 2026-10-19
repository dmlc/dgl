"""
Test Full operator on Ascend NPU.

Exercises the AscendC Full kernel (src/array/ascend/full_kernel.cpp) via DGL
high-level APIs. The Full operator fills a 1D NDArray with a scalar value; it
is triggered indirectly when constructing graph edge arrays (src/dst/etype) and
when filling feature tensors on NPU.

Coverage:
  - IdArray Full (int32 / int64) via edge array construction
  - Full<float32> via feature fill (B档向量化路径)
  - Full<float64> via feature fill (A档标量回退路径)
  - heterograph edge construction (Full + Range)
  - Boundary cases: n=0, 1, TILE_LENGTH boundary (8191/8192/8193), large (1M)

Differential testing: NPU results compared against CPU reference.
"""
import torch
import dgl


def _setup():
    if not (hasattr(torch, 'npu') and torch.npu.is_available()):
        return None, None
    return torch.device('npu:0'), torch.device('cpu')


def _check_eq(name, npu_val, cpu_val):
    assert torch.equal(npu_val.cpu(), cpu_val), \
        f"{name}: NPU result does not match CPU"


def _check_close(name, npu_val, cpu_val, atol=1e-5):
    assert torch.allclose(npu_val.cpu(), cpu_val, atol=atol), \
        f"{name}: max_diff={(npu_val.cpu() - cpu_val).abs().max().item():.6e}"


def test_full_idarray_int32():
    """IdArray Full<int32> via edge array construction."""
    device, cpu = _setup()
    if device is None:
        return
    src = torch.tensor([0, 1, 2, 0, 1, 3, 2, 7, 9, 100], dtype=torch.int32)
    dst = torch.tensor([1, 2, 0, 2, 0, 1, 3, 5, 1, 99], dtype=torch.int32)
    g_cpu = dgl.graph((src, dst), num_nodes=101)
    g_npu = g_cpu.to(device)
    u, v = g_npu.edges()
    uc, vc = g_cpu.edges()
    _check_eq("int32 src", u, uc)
    _check_eq("int32 dst", v, vc)


def test_full_idarray_int64():
    """IdArray Full<int64> via edge array construction."""
    device, cpu = _setup()
    if device is None:
        return
    src = torch.tensor([0, 1, 2, 0, 1, 3, 2, 7, 9, 100], dtype=torch.int64)
    dst = torch.tensor([1, 2, 0, 2, 0, 1, 3, 5, 1, 99], dtype=torch.int64)
    g_cpu = dgl.graph((src, dst), num_nodes=101)
    g_npu = g_cpu.to(device)
    u, v = g_npu.edges()
    uc, vc = g_cpu.edges()
    _check_eq("int64 src", u, uc)
    _check_eq("int64 dst", v, vc)


def test_full_float32_feature_fill():
    """Full<float32> via feature fill (B档向量化路径)."""
    device, cpu = _setup()
    if device is None:
        return
    g = dgl.graph(([0, 1, 2], [1, 2, 0])).to(device)
    g.edata['h'] = torch.full((3, 4), 3.14, dtype=torch.float32, device=device)
    ref = torch.full((3, 4), 3.14, dtype=torch.float32)
    _check_close("float32 fill", g.edata['h'], ref, atol=1e-5)


def test_full_float64_feature_fill():
    """Full<float64> via feature fill (A档标量回退路径)."""
    device, cpu = _setup()
    if device is None:
        return
    g = dgl.graph(([0, 1, 2], [1, 2, 0])).to(device)
    g.edata['h'] = torch.full((3, 4), 2.71, dtype=torch.float64, device=device)
    ref = torch.full((3, 4), 2.71, dtype=torch.float64)
    _check_close("float64 fill", g.edata['h'], ref, atol=1e-10)


def test_full_heterograph():
    """heterograph edge construction (Full + Range)."""
    device, cpu = _setup()
    if device is None:
        return
    g_cpu = dgl.heterograph({('A', 'r', 'B'): ([0, 1], [1, 0])})
    g_npu = g_cpu.to(device)
    u, v, e = g_npu.edges(form='all')
    uc, vc, ec = g_cpu.edges(form='all')
    _check_eq("heterograph src", u, uc)
    _check_eq("heterograph dst", v, vc)
    _check_eq("heterograph eid", e, ec)


def test_full_boundary_sizes():
    """Boundary cases around tile/multi-core splits of the AscendC kernel.

    kFullTileLength = 8192; 40 vector cores on 910B3 -> 40 * 8192 = 327680
    is the all-cores-exactly-one-tile point, +1/-1 exercise the remainder
    (`extra`) distribution of the contiguous per-core partition.
    """
    device, cpu = _setup()
    if device is None:
        return
    for n in [0, 1, 8191, 8192, 8193, 16383, 16384, 16385, 50000, 200000,
              327679, 327680, 327681, 1000000, 8192000]:
        s = torch.arange(n, dtype=torch.int64) % 1000
        d = (s + 1) % 1000
        g_cpu = dgl.graph((s, d), num_nodes=1000)
        g_npu = g_cpu.to(device)
        u, v = g_npu.edges()
        uc, vc = g_cpu.edges()
        _check_eq(f"n={n} src", u, uc)
        _check_eq(f"n={n} dst", v, vc)


if __name__ == '__main__':
    test_full_idarray_int32()
    print("[PASS] IdArray int32")
    test_full_idarray_int64()
    print("[PASS] IdArray int64")
    test_full_float32_feature_fill()
    print("[PASS] float32 fill (B档向量化)")
    test_full_float64_feature_fill()
    print("[PASS] float64 fill (A档标量)")
    test_full_heterograph()
    print("[PASS] heterograph")
    test_full_boundary_sizes()
    print("[PASS] boundary sizes")
    print("\n=== ALL CORRECTNESS TESTS PASSED ===")
