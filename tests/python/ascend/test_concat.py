"""
concat 算子 Ascend NPU 测试。

C++ 入口 `aten::ConcatND`（src/array/array.cc），Ascend 侧 AscendC kernel
（src/array/ascend/concat_kernel.cpp + concat_tiling.h + concat.cc），
golden 为 torch.cat（concat_golden.py），Python 封装见 concat_impl.py。

覆盖：
- 典型配置：功能_P0 / 性能_P0 / 多输入_P1
- 泛化：1D~4D、任意 dim（含负数）、非等长输入、输入个数 1~128
- dtype：fp32 / fp16 走 AscendC kernel；int32 / int64 / float64 走通用路径
- 分块与多核：单段跨多 UB chunk、恰为 chunk 边界、段数远超核数、单 chunk 单核
- 值域：大值、零值、NaN/Inf 逐位透传、整型极值
- 边界：拼接轴长度为 0、非拼接轴长度为 0、(1,1,1) 最小形状、非对齐切片输入
- 稳定性：重复调用、与其他算子交替、流序、参数非法时报错
"""
import pytest
import torch
import torch_npu
import dgl

from concat_golden import concat_golden
from concat_impl import concat_impl

dev = torch.device("npu:0")
torch.npu.set_device(dev)

# 与 src/array/ascend/concat_tiling.h 的 CONCAT_CHUNK_BYTES 保持一致：
# 单个 UB 缓冲 97280 字节。段元素数超过 chunk_elems 才会触发分块搬运。
CHUNK_BYTES = 97280
FP32_CHUNK_ELEMS = CHUNK_BYTES // 4
FP16_CHUNK_ELEMS = CHUNK_BYTES // 2
# Ascend 910B 的 Vector Core 数，段数超过它才会出现每核多段。
VECTOR_CORE_NUM = 48


def randn(*shape, dtype=torch.float32, device=dev, seed=None):
    """可复现的随机张量。"""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, device=device)


def check(tensors, dim):
    """concat_impl 结果与 golden 逐位一致，并返回结果供调用方断言 shape。"""
    got = concat_impl(tensors, dim)
    want = concat_golden(tensors, dim)
    shapes = [tuple(t.shape) for t in tensors]
    assert tuple(got.shape) == tuple(want.shape), \
        f"dim={dim} shapes={shapes}: {tuple(got.shape)} != {tuple(want.shape)}"
    assert got.dtype == want.dtype, f"dim={dim}: {got.dtype} != {want.dtype}"
    assert got.device.type == want.device.type, \
        f"dim={dim}: {got.device} != {want.device}"
    # 纯数据搬运，必须逐位一致；equal_nan 让 NaN 透传用例也能比较。
    torch.testing.assert_close(
        got.cpu(), want.cpu(), rtol=0, atol=0, equal_nan=True,
        msg=f"dim={dim} shapes={shapes}")
    return got


class TestConcatTypical:
    """规格文档中的典型配置。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_functional_p0(self):
        """功能_P0：2×(256,1024) dim=0 → (512,1024)。"""
        ts = [randn(256, 1024, seed=42), randn(256, 1024, seed=43)]
        assert tuple(check(ts, 0).shape) == (512, 1024)

    def test_performance_p0(self):
        """性能_P0：2×(1024,512) dim=-1 → (1024,1024)。"""
        ts = [randn(1024, 512, seed=44), randn(1024, 512, seed=45)]
        assert tuple(check(ts, -1).shape) == (1024, 1024)

    def test_multi_input_p1(self):
        """多输入_P1：4×(128,64) dim=0 → (512,64)。"""
        ts = [randn(128, 64, seed=46 + i) for i in range(4)]
        assert tuple(check(ts, 0).shape) == (512, 64)


class TestConcatDims:
    """泛化：各 ndim、各 dim（含负数）、非等长。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_1d(self):
        ts = [randn(7, seed=1), randn(11, seed=2), randn(3, seed=3)]
        assert tuple(check(ts, 0).shape) == (21,)

    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
    def test_3d_all_dims(self, dim):
        ts = [randn(4, 8, 16, seed=10 + i) for i in range(2)]
        check(ts, dim)

    def test_4d_middle_axis(self):
        ts = [randn(2, 3, 4, 5, seed=40 + i) for i in range(3)]
        assert tuple(check(ts, 2).shape) == (2, 3, 12, 5)

    def test_middle_axis_uneven(self):
        """dim=1 中间轴且长度不等：outer=4 段，每段 len_i*inner 元素。"""
        ts = [randn(4, 1, 32, seed=20), randn(4, 2, 32, seed=21),
              randn(4, 5, 32, seed=22)]
        assert tuple(check(ts, 1).shape) == (4, 8, 32)

    def test_last_axis_inner_one(self):
        """dim=last 且 inner=1：每段仅 2 个元素，段数远超核数。"""
        ts = [randn(1024, 3, seed=30), randn(1024, 5, seed=31)]
        assert tuple(check(ts, 1).shape) == (1024, 8)

    def test_uneven_lengths_dim0(self):
        ts = [randn(1, 64, seed=50), randn(17, 64, seed=51),
              randn(3, 64, seed=52), randn(256, 64, seed=53)]
        assert tuple(check(ts, 0).shape) == (277, 64)


class TestConcatInputCount:
    """输入个数 N：1、2、48（等于核数）、超过核数。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_single_input(self):
        ts = [randn(64, 32, seed=70)]
        assert tuple(check(ts, 0).shape) == (64, 32)

    @pytest.mark.parametrize("n", [2, 3, VECTOR_CORE_NUM, 128])
    def test_many_inputs(self, n):
        ts = [randn(8, 16, seed=100 + i) for i in range(n)]
        assert tuple(check(ts, 0).shape) == (8 * n, 16)


class TestConcatChunking:
    """UB 分块搬运与多核切分覆盖。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_multi_chunk_fp32(self):
        """单段元素数跨 3 个以上 UB chunk。"""
        n = FP32_CHUNK_ELEMS * 3 + 17
        check([randn(n, seed=60), randn(n, seed=61)], 0)

    def test_multi_chunk_fp16(self):
        n = FP16_CHUNK_ELEMS * 2 + 5
        check([randn(n, dtype=torch.float16, seed=62),
               randn(n, dtype=torch.float16, seed=63)], 0)

    def test_exact_chunk_boundary(self):
        """段元素数恰为一个 chunk：无尾块。"""
        check([randn(FP32_CHUNK_ELEMS, seed=64),
               randn(FP32_CHUNK_ELEMS, seed=65)], 0)

    def test_chunk_plus_one_element(self):
        """段元素数为 chunk+1：尾块仅 1 个元素。"""
        check([randn(FP32_CHUNK_ELEMS + 1, seed=66),
               randn(FP32_CHUNK_ELEMS + 1, seed=67)], 0)

    def test_segments_far_exceed_cores(self):
        """dim=last，段数 4096 ≫ 48 核，每核串行处理多段。"""
        ts = [randn(4096, 2, seed=68), randn(4096, 2, seed=69)]
        assert tuple(check(ts, 1).shape) == (4096, 4)

    def test_single_chunk_single_core(self):
        """totalChunks=1 → blockDim=1，单核单 chunk。"""
        check([randn(2, 3, seed=70), randn(2, 3, seed=71)], 0)

    def test_large_2d(self):
        ts = [randn(1024, 1024, seed=72 + i) for i in range(4)]
        assert tuple(check(ts, 0).shape) == (4096, 1024)


class TestConcatDtype:
    """fp32/fp16 走 AscendC kernel，其余 dtype 走通用 CopyDataFromTo 路径。"""

    def setup_method(self):
        torch.npu.synchronize()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_fast_path_dtypes(self, dtype):
        ts = [randn(32, 64, dtype=dtype, seed=80),
              randn(16, 64, dtype=dtype, seed=81)]
        got = check(ts, 0)
        assert got.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64, torch.float64])
    def test_generic_path_dtypes(self, dtype):
        a = torch.arange(3 * 5, dtype=dtype, device=dev).reshape(3, 5)
        b = torch.arange(2 * 5, dtype=dtype, device=dev).reshape(2, 5) + 100
        got = check([a, b], 0)
        assert got.dtype == dtype
        assert tuple(got.shape) == (5, 5)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_generic_path_dim1(self, dtype):
        a = torch.arange(4 * 3, dtype=dtype, device=dev).reshape(4, 3)
        b = torch.arange(4 * 5, dtype=dtype, device=dev).reshape(4, 5) + 100
        assert tuple(check([a, b], 1).shape) == (4, 8)

    def test_fp16_vs_fp32_same_bits(self):
        """同一份数据在 fp16/fp32 下都逐位透传。"""
        base = randn(64, 32, seed=90)
        got32 = check([base, base.clone()], 0)
        got16 = check([base.half(), base.half()], 0)
        torch.testing.assert_close(
            got32.half().cpu(), got16.cpu(), rtol=0, atol=0)


class TestConcatValues:
    """值域：concat 不做任何算术，任何值都必须逐位透传。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_large_values(self):
        ts = [torch.full((4, 8), 1e30, device=dev),
              torch.full((4, 8), -1e30, device=dev)]
        got = check(ts, 0)
        assert not torch.isnan(got).any() and not torch.isinf(got).any()

    def test_small_values(self):
        ts = [torch.full((4, 8), 1e-30, device=dev),
              torch.full((4, 8), -1e-30, device=dev)]
        check(ts, 0)

    def test_zeros(self):
        ts = [torch.zeros(3, 7, device=dev), torch.zeros(2, 7, device=dev)]
        assert (check(ts, 0) == 0).all()

    def test_nan_inf_passthrough(self):
        a = torch.tensor(
            [[float("nan"), float("inf")], [-float("inf"), 0.0]], device=dev)
        b = torch.tensor(
            [[1.0, float("nan")], [float("inf"), -float("inf")]], device=dev)
        got = check([a, b], 0)
        assert torch.isnan(got).sum().item() == 2
        assert torch.isinf(got).sum().item() == 4

    def test_int_extremes(self):
        a = torch.tensor(
            [[2 ** 63 - 1, -(2 ** 63)], [0, -1]],
            dtype=torch.int64, device=dev)
        b = torch.tensor([[7, 8], [9, 10]], dtype=torch.int64, device=dev)
        check([a, b], 0)


class TestConcatEdgeCases:
    """空张量、最小形状、非对齐切片。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_zero_len_on_concat_axis(self):
        """拼接轴长度为 0 → 输出该轴仍为 0，且不发起 kernel。"""
        ts = [randn(0, 64, seed=110), randn(0, 64, seed=111)]
        assert tuple(check(ts, 0).shape) == (0, 64)

    def test_zero_len_mixed(self):
        """空与非空混合：空输入贡献 0 个元素。"""
        ts = [randn(0, 64, seed=112), randn(3, 64, seed=113),
              randn(0, 64, seed=114)]
        assert tuple(check(ts, 0).shape) == (3, 64)

    def test_zero_len_on_other_axis(self):
        """非拼接轴长度为 0 → 输出保留该 0 维。"""
        ts = [randn(0, 64, seed=115), randn(0, 64, seed=116)]
        assert tuple(check(ts, 1).shape) == (0, 128)

    def test_minimal_shape(self):
        check([randn(1, 1, 1, seed=117), randn(1, 1, 1, seed=118)], 0)

    def test_unaligned_slice_input(self):
        """切片输入：data_ptr 非 32B 对齐，应回退通用路径且结果正确。"""
        full = randn(64, 100, seed=119)  # 行 400 字节，t[1:] 偏移 400
        a = full[1:5]
        b = full[7:9]
        assert a.data_ptr() % 32 != 0, "用例前提：切片应非 32B 对齐"
        assert tuple(check([a, b], 0).shape) == (6, 100)

    def test_non_contiguous_input(self):
        """转置产生的非连续张量（封装内部 .contiguous()）。"""
        a = randn(8, 32, seed=120).t()
        b = randn(8, 32, seed=121).t()
        assert not a.is_contiguous()
        assert tuple(check([a, b], 1).shape) == (32, 16)

    def test_sliced_contiguous_input(self):
        """连续切片：data_ptr 偏移为行字节数整数倍，仍为 32B 对齐。"""
        full = randn(64, 128, seed=122)  # 行 512 字节
        a = full[0:16]
        b = full[16:32]
        assert a.data_ptr() % 32 == 0
        assert tuple(check([a, b], 0).shape) == (32, 128)


class TestConcatCpu:
    """CPU 设备走通用路径，与 NPU 结果一致。"""

    def setup_method(self):
        torch.npu.synchronize()

    @pytest.mark.parametrize("dim", [0, 1])
    def test_cpu(self, dim):
        torch.manual_seed(130)
        if dim == 0:
            ts = [torch.randn(6, 8), torch.randn(4, 8)]
        else:
            ts = [torch.randn(6, 8), torch.randn(6, 4)]
        got = concat_impl(ts, dim)
        want = concat_golden(ts, dim)
        assert got.device.type == "cpu"
        torch.testing.assert_close(got, want, rtol=0, atol=0)


class TestConcatStability:
    """重复调用、流序、与其他算子交替。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_repeated_calls(self):
        ts = [randn(64, 128, seed=140), randn(64, 128, seed=141)]
        want = concat_golden(ts, 0).cpu()
        for i in range(5):
            got = concat_impl(ts, 0).cpu()
            torch.testing.assert_close(
                got, want, rtol=0, atol=0, msg=f"第 {i} 次调用结果漂移")

    def test_interleaved_with_other_ops(self):
        """concat 与其它 NPU 算子交替，验证流序与 buffer 复用安全。"""
        a = randn(32, 64, seed=150)
        b = randn(32, 64, seed=151)
        for _ in range(3):
            a = a.mul(1.01)
            b = b.add(0.5)
            got = concat_impl([a, b], 0)
            want = concat_golden([a, b], 0)
            torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)

    def test_stream_ordering(self):
        """前置 NPU 异步写必须在 concat 前可见。"""
        a = randn(256, 1024, seed=160)
        a.mul_(2.0)  # 异步下发到当前流
        got = concat_impl([a, a.clone()], 0)
        want = concat_golden([a, a], 0)
        torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)

    def test_result_is_independent_of_inputs(self):
        """输出是新分配内存，改输入不应影响已返回结果。"""
        a = randn(16, 32, seed=170)
        got = concat_impl([a, a.clone()], 0)
        a.fill_(0.0)
        assert got.abs().max().item() > 0


class TestConcatErrors:
    """非法参数应在 C++ 侧报错而非静默出错。"""

    def setup_method(self):
        torch.npu.synchronize()

    def test_empty_list(self):
        with pytest.raises(dgl.DGLError):
            concat_impl([], 0)

    def test_dim_out_of_range(self):
        ts = [randn(4, 8, seed=180), randn(4, 8, seed=181)]
        with pytest.raises(dgl.DGLError):
            concat_impl(ts, 2)
        with pytest.raises(dgl.DGLError):
            concat_impl(ts, -3)

    def test_mismatched_shape(self):
        with pytest.raises(dgl.DGLError):
            concat_impl([randn(4, 8, seed=182), randn(4, 9, seed=183)], 0)

    def test_mismatched_dtype(self):
        with pytest.raises(dgl.DGLError):
            concat_impl([randn(4, 8, seed=184),
                         randn(4, 8, seed=185).half()], 0)

    def test_mismatched_device(self):
        with pytest.raises(dgl.DGLError):
            concat_impl([randn(4, 8, seed=186),
                         torch.randn(4, 8)], 0)
