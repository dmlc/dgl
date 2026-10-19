"""
Test Range operator on Ascend NPU.

Verifies that the AscendC Range kernel (range_i32/range_i64) produces
correct integer sequences, matching CPU results element-wise. Also
benchmarks NPU kernel performance against the CPU fallback path.
"""
import ctypes
import time

import torch

try:
    import torch_npu  # noqa: F401
    if not torch.npu.is_available():
        pytest.fail("❌ NPU UT 强制失败：torch_npu 未安装")
except (ImportError, AttributeError, RuntimeError) as e:
    pytest.fail("❌ NPU UT 强制失败：NPU 驱动/硬件不可用")


def _setup():
    # 加载 dgl C++ 库，使 torch.arange 路由到 AscendC Range kernel
    # 优先 import dgl，失败则 ctypes 加载编译产物
    try:
        import dgl  # noqa: F401
    except ImportError:
        import os
        dgl_so = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "build", "libdgl.so")
        ctypes.CDLL(dgl_so)
    dev = torch_npu.npu.current_device()
    torch.npu.set_device(dev)
    return dev


# ---------------------------------------------------------------------------
# 测试 1: 功能测试 - int32 Range (NPU)
# ---------------------------------------------------------------------------
CASES_I32 = [(0, 10), (0, 100), (0, 1000), (5, 15), (100, 200), (0, 1)]
CASES_I64 = [(0, 10), (0, 1000), (5, 25), (0, 1), (1000000, 1000010)]
SIZES = [1000, 10000, 100000, 1000000, 10000000]


def test_range_i32_basic():
    dev = _setup()
    if dev is None:
        return
    for low, high in CASES_I32:
        npu = torch.arange(low, high, dtype=torch.int32, device="npu:0")
        cpu = torch.arange(low, high, dtype=torch.int32)
        assert torch.equal(npu.cpu(), cpu), f"Range({low},{high}) int32 mismatch"


def test_range_i64_basic():
    dev = _setup()
    if dev is None:
        return
    for low, high in CASES_I64:
        npu = torch.arange(low, high, dtype=torch.int64, device="npu:0")
        cpu = torch.arange(low, high, dtype=torch.int64)
        assert torch.equal(npu.cpu(), cpu), f"Range({low},{high}) int64 mismatch"


def test_range_empty():
    dev = _setup()
    if dev is None:
        return
    npu = torch.arange(5, 5, dtype=torch.int32, device="npu:0")
    assert npu.shape[0] == 0, "empty array should have length 0"


def test_range_single_element():
    dev = _setup()
    if dev is None:
        return
    npu = torch.arange(42, 43, dtype=torch.int32, device="npu:0")
    assert npu.cpu().item() == 42, "single element should be 42"


def test_range_i32_precision():
    dev = _setup()
    if dev is None:
        return
    npu = torch.arange(0, 10000, dtype=torch.int32, device="npu:0")
    cpu = torch.arange(0, 10000, dtype=torch.int32)
    assert torch.equal(npu.cpu(), cpu), "int32 precision mismatch"


def test_range_i64_precision():
    dev = _setup()
    if dev is None:
        return
    npu = torch.arange(0, 10000, dtype=torch.int64, device="npu:0")
    cpu = torch.arange(0, 10000, dtype=torch.int64)
    assert torch.equal(npu.cpu(), cpu), "int64 precision mismatch"


def test_range_perf():
    dev = _setup()
    if dev is None:
        return
    for size in SIZES:
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            torch.arange(0, size, dtype=torch.int32, device="npu:0")
        torch.npu.synchronize()
        npu_ms = (time.perf_counter() - t0) / 20 * 1000

        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            cpu = torch.arange(0, size, dtype=torch.int32)
            cpu.to("npu:0")
        torch.npu.synchronize()
        cpu_ms = (time.perf_counter() - t0) / 20 * 1000

        speedup = cpu_ms / npu_ms if npu_ms > 0 else float("inf")
        print(f"  size={size:>10,d}: NPU={npu_ms:.3f}ms, CPU+copy={cpu_ms:.3f}ms, speedup={speedup:.2f}x")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("DGL Ascend Range 算子测试")
    print("=" * 60)

    dev = _setup()
    if dev is not None:
        print(f"[OK] dgl 库已加载")
        print(f"[INFO] NPU 设备: {dev}")

    all_pass = True

    # 测试 1: 功能测试
    print("\n" + "=" * 60)
    print("测试 1: 功能测试 - int32 Range (NPU)")
    print("=" * 60)
    for low, high in CASES_I32:
        npu = torch.arange(low, high, dtype=torch.int32, device="npu:0")
        cpu = torch.arange(low, high, dtype=torch.int32)
        match = torch.equal(npu.cpu(), cpu)
        print(f"  [{'PASS' if match else 'FAIL'}] Range({low}, {high}) int32, len={high-low}")
        if not match:
            all_pass = False

    # 测试 2: 功能测试
    print("\n" + "=" * 60)
    print("测试 2: 功能测试 - int64 Range (NPU)")
    print("=" * 60)
    for low, high in CASES_I64:
        npu = torch.arange(low, high, dtype=torch.int64, device="npu:0")
        cpu = torch.arange(low, high, dtype=torch.int64)
        match = torch.equal(npu.cpu(), cpu)
        print(f"  [{'PASS' if match else 'FAIL'}] Range({low}, {high}) int64, len={high-low}")
        if not match:
            all_pass = False

    # 测试 3: 精度测试
    print("\n" + "=" * 60)
    print("测试 3: 精度测试 - 大范围逐元素对比")
    print("=" * 60)
    for dtype_name, dtype in [("int32", torch.int32), ("int64", torch.int64)]:
        npu = torch.arange(0, 10000, dtype=dtype, device="npu:0")
        cpu = torch.arange(0, 10000, dtype=dtype)
        npu_cpu = npu.cpu()
        exact_match = torch.equal(npu_cpu, cpu)
        max_diff = (npu_cpu.to(torch.int64) - cpu.to(torch.int64)).abs().max().item()
        print(f"  [{dtype_name}] Range(0, 10000):")
        print(f"    精确匹配: {'YES' if exact_match else 'NO'}")
        print(f"    最大差异: {max_diff}")
        print(f"    前5个值 (NPU): {npu_cpu[:5].tolist()}")
        print(f"    前5个值 (CPU): {cpu[:5].tolist()}")
        print(f"    后5个值 (NPU): {npu_cpu[-5:].tolist()}")
        print(f"    后5个值 (CPU): {cpu[-5:].tolist()}")
        if not exact_match:
            all_pass = False

    # 测试 4: 边界测试
    print("\n" + "=" * 60)
    print("测试 4: 边界测试")
    print("=" * 60)
    try:
        npu = torch.arange(5, 5, dtype=torch.int32, device="npu:0")
        print(f"  [PASS] 空数组 Range(5, 5), len={npu.shape[0]}")
    except Exception as e:
        print(f"  [FAIL] 空数组 Range(5, 5): {e}")
        all_pass = False

    try:
        npu = torch.arange(42, 43, dtype=torch.int32, device="npu:0")
        val = npu.cpu().item()
        match = (val == 42)
        print(f"  [{'PASS' if match else 'FAIL'}] 单元素 Range(42, 43), val={val}")
        if not match:
            all_pass = False
    except Exception as e:
        print(f"  [FAIL] 单元素 Range(42, 43): {e}")
        all_pass = False

    # 测试 5: 性能测试
    print("\n" + "=" * 60)
    print("测试 5: 性能测试 - NPU kernel vs CPU 生成+拷贝")
    print("=" * 60)
    for size in SIZES:
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            torch.arange(0, size, dtype=torch.int32, device="npu:0")
        torch.npu.synchronize()
        npu_ms = (time.perf_counter() - t0) / 20 * 1000

        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            cpu = torch.arange(0, size, dtype=torch.int32)
            cpu.to("npu:0")
        torch.npu.synchronize()
        cpu_ms = (time.perf_counter() - t0) / 20 * 1000

        speedup = cpu_ms / npu_ms if npu_ms > 0 else float("inf")
        print(f"  size={size:>10,d}: NPU kernel={npu_ms:.3f}ms, CPU+copy={cpu_ms:.3f}ms, speedup={speedup:.2f}x")

    # 总结
    print("\n" + "=" * 60)
    if all_pass:
        print("总结: 全部测试通过 ✅")
    else:
        print("总结: 存在失败项 ❌")
    print("=" * 60)
