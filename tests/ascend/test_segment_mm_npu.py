"""SegmentMM / SegmentMMBackwardB Ascend NPU functional tests.

模式 B（无评测集）自建 golden + 用例。golden 为 torch CPU FP32 参考实现，
Per-segment GEMM per math definition，不复用被测 NPU kernel，保证比对独立性。

测试框架：pytest，单文件组织 L0/L1/L2 + 白盒补充共 61 用例
（SegmentMM: L0 5 + L1 10 + L2 func 6 + L2 异常 8 + WB 9 = 38，
 SegmentMMBackwardB: L0 4 + L1 6 + L2 func 3 + L2 异常 3 + WB 7 = 23）。
白盒用例（WB-*）由源码分支枚举驱动，覆盖 M/K/N 尾块、K-split 累加、
kFractal 循环、多核边界、尾核空闲早返回等内部分支。

精度口径：float32 混合容差（rtol=2^-10, atol=2^-16, matched_ratio>=0.99,
max_abs_error<=1e-2 固定上限，对齐 ops-precision-standard float_compute 3 FP32 列），
nan/inf 用 isnan/isinf 等价判定不套混合容差。

调用入口：
  - SegmentMM 前向：经 DGL dispatch 的 `_segment_mm`（dgl._sparse_ops），
    b_trans=false/true 两条路径；a_trans 恒 false（dispatch CHECK 约束）。
  - SegmentMMBackwardB：经 `_segment_mm_backward_B`（dgl._sparse_ops）。
  - a_trans=true 异常用例走原始 C API `_CAPI_DGLKernelSEGMENTMM`。
seglen 须在 CPU（dispatch CHECK_EQ 约束）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
import torch

import dgl
from dgl._sparse_ops import (
    _CAPI_DGLKernelSEGMENTMM,
    _segment_mm,
    _segment_mm_backward_B,
    to_dgl_nd,
    to_dgl_nd_for_write,
)

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


# ---------------------------------------------------------------------------
# 设备与同步
# ---------------------------------------------------------------------------

def get_npu_device(device_id: int = 0) -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("未检测到可用的 NPU 设备。")
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)


# ---------------------------------------------------------------------------
# Golden implementation (CPU FP32)
# ---------------------------------------------------------------------------

def segment_mm_golden(
    A: torch.Tensor,
    B: torch.Tensor,
    seglen: torch.Tensor,
    b_trans: bool,
) -> torch.Tensor:
    """SegmentMM golden：逐 etype A_seg @ B[r]（或 @ B[r].t()）。

    A: [M, K] float32 (CPU)
    B: [R, K, N] (b_trans=False) 或 [R, N, K] (b_trans=True) float32 (CPU)
    seglen: [R] int32/int64 (CPU)，seglen[r] = m_r
    返回 C: [M, N] float32 (CPU)

    边界自洽：seg[r]=0 / R=0 / K=0 / N=0 由 torch 切片与 matmul 自然处理，
    不做 early-return 特判，与公式逐字对齐。
    """
    seg = seglen.to(torch.int64).tolist()
    R = len(seg)
    K = A.shape[1]
    if R == 0:
        N = B.shape[2] if not b_trans else B.shape[1]
        M = A.shape[0]
        return torch.zeros(M, N, dtype=torch.float32)
    N = B.shape[1] if b_trans else B.shape[2]
    M = A.shape[0]
    C = torch.zeros(M, N, dtype=torch.float32)
    off = 0
    for r in range(R):
        m_r = seg[r]
        A_seg = A[off:off + m_r, :]          # [m_r, K]
        B_r = B[r]                            # [K, N] 或 [N, K]
        if b_trans:
            C[off:off + m_r, :] = A_seg @ B_r.t()   # [m_r,K]@[K,N]=[m_r,N]
        else:
            C[off:off + m_r, :] = A_seg @ B_r       # [m_r,K]@[K,N]=[m_r,N]
        off += m_r
    return C


def segment_mm_backward_b_golden(
    A: torch.Tensor,
    dC: torch.Tensor,
    seglen: torch.Tensor,
) -> torch.Tensor:
    """SegmentMMBackwardB golden：逐 etype dB[r] = A_seg.t() @ dC_seg。

    A: [K_sum, N] float32 (CPU)
    dC: [K_sum, M] float32 (CPU)
    seglen: [R] int32/int64 (CPU)，seglen[r] = k_r
    返回 dB: [R, N, M] float32 (CPU)
    """
    seg = seglen.to(torch.int64).tolist()
    R = len(seg)
    N = A.shape[1]
    M = dC.shape[1]
    dB = torch.zeros(R, N, M, dtype=torch.float32)
    off = 0
    for r in range(R):
        k_r = seg[r]
        A_seg = A[off:off + k_r, :]     # [k_r, N]
        dC_seg = dC[off:off + k_r, :]   # [k_r, M]
        dB[r] = A_seg.t() @ dC_seg      # [N, k_r] @ [k_r, M] = [N, M]
        off += k_r
    return dB


# ---------------------------------------------------------------------------
# 精度断言（混合容差 + nan/inf 等价判定）
# ---------------------------------------------------------------------------

RTOL_FP32 = 2 ** (-10)        # 9.77e-4
ATOL_FP32 = 2 ** (-16)        # 1.53e-5
REQUIRED_MATCHED_RATIO = 0.99
MAX_ABS_ERROR_LIMIT = 1e-2     # 固定上限：max(1e-2, 32*2^-23)=1e-2，对齐权威源 float_compute 3 FP32 列


def assert_mixed_tolerance(
    npu_out: torch.Tensor,
    golden: torch.Tensor,
    rtol: float = RTOL_FP32,
    atol: float = ATOL_FP32,
    required_matched_ratio: float = REQUIRED_MATCHED_RATIO,
    max_abs_error_limit: float = MAX_ABS_ERROR_LIMIT,
) -> None:
    """逐元素混合容差 |a-g|<=atol+rtol*|g| + 通过率门禁 + max_abs_error 上限。

    nan/inf 元素用等价判定（isnan/isinf 配对），不参与混合容差计算。
    空张量（numel==0）直接通过。

    max_abs_error 上限为固定值 1e-2（对齐 ops-precision-standard float_compute 3
    FP32 列：max_abs_error_limit=max(1e-2, 32·ULP@1.0)=1e-2），不随输出量级缩放。
    用例输入量级须控制在使 GEMM 结果量级不超过 ~1e3 量级（保证 FP32 舍入绝对误差
    仍落在 1e-2 上限内）；大量级输入用例已调整为常规量级（见 L1-SMM-08）。
    """
    assert npu_out.shape == golden.shape, (
        f"shape 不一致: npu {tuple(npu_out.shape)} vs golden {tuple(golden.shape)}"
    )
    total = npu_out.numel()
    if total == 0:
        return  # 空进空出，通过

    npu_f = npu_out.float()
    gld_f = golden.float()

    # 分离 nan/inf 与有限值
    npu_nan = torch.isnan(npu_f)
    gld_nan = torch.isnan(gld_f)
    npu_inf = torch.isinf(npu_f)
    gld_inf = torch.isinf(gld_f)
    finite_mask = ~(npu_nan | gld_nan | npu_inf | gld_inf)

    # nan/inf 等价判定
    nan_mismatch = (npu_nan != gld_nan)
    inf_mismatch = (npu_inf != gld_inf)
    # inf 须同号：在两者均为 inf 的同位置掩码上做符号比较，避免 1D 索引结果
    # 与 2D 掩码直接 & 导致的广播崩溃（inf 计数 ≠ 末维时 shape 不匹配）。
    both_inf = gld_inf & npu_inf
    if both_inf.any():
        sign_mismatch = both_inf & (torch.sign(npu_f) != torch.sign(gld_f))
    else:
        sign_mismatch = torch.zeros_like(gld_inf, dtype=torch.bool)
    special_mismatch = nan_mismatch | inf_mismatch | sign_mismatch

    # 有限值混合容差
    if finite_mask.any():
        abs_err = torch.abs(npu_f[finite_mask] - gld_f[finite_mask])
        threshold = atol + rtol * torch.abs(gld_f[finite_mask])
        elem_pass = abs_err <= threshold
        finite_total = finite_mask.sum().item()
        finite_pass = elem_pass.sum().item()
        max_abs_error = float(abs_err.max().item())
    else:
        finite_total = 0
        finite_pass = 0
        max_abs_error = 0.0

    special_count = special_mismatch.sum().item()
    # 通过率：有限值通过 + 特殊值匹配，除以总数
    matched = finite_pass + (total - finite_mask.sum().item() - special_count)
    matched_ratio = matched / total

    # max_abs_error 上限为固定 1e-2（对齐权威源 float_compute 3 FP32 列），
    # 不随输出量级缩放；用例输入量级已控制在结果量级不超过 ~1e3 量级。
    assert special_mismatch.sum().item() == 0, (
        f"nan/inf 等价判定失败: {special_count} 个元素 nan/inf 不匹配"
    )
    assert matched_ratio >= required_matched_ratio, (
        f"通过率不达标: matched_ratio={matched_ratio:.6f} < {required_matched_ratio}"
    )
    assert max_abs_error <= max_abs_error_limit, (
        f"max_abs_error={max_abs_error:.6g} > limit {max_abs_error_limit:.6g}"
    )


def assert_empty_or_allzero(npu_out: torch.Tensor, golden: torch.Tensor) -> None:
    """空/全零输出断言：shape 一致且为空或全零。"""
    assert npu_out.shape == golden.shape, (
        f"shape 不一致: npu {tuple(npu_out.shape)} vs golden {tuple(golden.shape)}"
    )
    if npu_out.numel() == 0:
        return
    assert torch.all(npu_out.float() == 0), "预期全零输出但存在非零元素"


# ---------------------------------------------------------------------------
# 数值生成
# ---------------------------------------------------------------------------

def gen_tensor(
    shape: Tuple[int, ...],
    kind: str = "normal",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """按 kind 生成 CPU 张量。

    kind:
      normal  [-1, 1] 均匀
      small   [1e-3, 1e-2]
      large   [1, 10]   （常规偏大量级，GEMM 结果量级 ≤ ~1e3，保证 FP32 舍入绝对误差落在固定 1e-2 上限内）
      neg     [-1, 1] 含负镜像（normal 即含负）
      zero    全零（含 ±0）
      inf     normal 基础上注入 +inf
      nan     normal 基础上注入 nan
      fmax    接近 float32 max
    """
    g = torch.Generator().manual_seed(seed)
    if kind == "normal" or kind == "neg":
        t = (torch.rand(shape, generator=g, dtype=dtype) * 2 - 1)
        return t
    if kind == "small":
        return torch.rand(shape, generator=g, dtype=dtype) * 9e-3 + 1e-3
    if kind == "large":
        return torch.rand(shape, generator=g, dtype=dtype) * (10 - 1) + 1
    if kind == "zero":
        t = torch.zeros(shape, dtype=dtype)
        if t.numel() > 0:
            t.view(-1)[0] = 0.0
            if t.numel() > 1:
                t.view(-1)[1] = -0.0
        return t
    if kind == "inf":
        t = (torch.rand(shape, generator=g, dtype=dtype) * 2 - 1)
        if t.numel() > 0:
            t.view(-1)[0] = float("inf")
        return t
    if kind == "nan":
        t = (torch.rand(shape, generator=g, dtype=dtype) * 2 - 1)
        if t.numel() > 0:
            t.view(-1)[0] = float("nan")
        return t
    if kind == "fmax":
        t = (torch.rand(shape, generator=g, dtype=dtype) * 2 - 1)
        scale = torch.finfo(torch.float32).max * 0.5
        return t * scale
    raise ValueError(f"unknown value kind: {kind}")


# ---------------------------------------------------------------------------
# NPU 调用封装
# ---------------------------------------------------------------------------

def call_segment_mm_npu(
    A_cpu: torch.Tensor,
    B_cpu: torch.Tensor,
    seglen_cpu: torch.Tensor,
    b_trans: bool,
    device: torch.device,
) -> torch.Tensor:
    """调用 SegmentMM 前向（_segment_mm，a_trans 恒 false）。seglen 留 CPU。"""
    A = A_cpu.to(device)
    B = B_cpu.to(device)
    if b_trans:
        N = B_cpu.shape[1]   # B[R, N, K]
    else:
        N = B_cpu.shape[2]   # B[R, K, N]
    M = A_cpu.shape[0]
    C = torch.empty((M, N), device=device, dtype=A_cpu.dtype)
    _segment_mm(A, B, C, seglen_cpu, b_trans=b_trans)
    synchronize(device)
    return C.detach().cpu()


def call_segment_mm_raw_npu(
    A_cpu: torch.Tensor,
    B_cpu: torch.Tensor,
    seglen_cpu: torch.Tensor,
    a_trans: bool,
    b_trans: bool,
    device: torch.device,
) -> torch.Tensor:
    """直接调用原始 C API _CAPI_DGLKernelSEGMENTMM（用于 a_trans=true 异常用例）。"""
    A = A_cpu.to(device)
    B = B_cpu.to(device)
    if b_trans:
        N = B_cpu.shape[1]
    else:
        N = B_cpu.shape[2]
    M = A_cpu.shape[0]
    C = torch.empty((M, N), device=device, dtype=A_cpu.dtype)
    _CAPI_DGLKernelSEGMENTMM(
        to_dgl_nd(A),
        to_dgl_nd(B),
        to_dgl_nd_for_write(C),
        to_dgl_nd(seglen_cpu),
        a_trans,
        b_trans,
    )
    synchronize(device)
    return C.detach().cpu()


def call_segment_mm_backward_b_npu(
    A_cpu: torch.Tensor,
    dC_cpu: torch.Tensor,
    seglen_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """调用 SegmentMMBackwardB（_segment_mm_backward_B）。seglen 留 CPU。"""
    A = A_cpu.to(device)
    dC = dC_cpu.to(device)
    R = seglen_cpu.shape[0]
    N = A_cpu.shape[1]
    M = dC_cpu.shape[1]
    dB = torch.empty((R, N, M), device=device, dtype=A_cpu.dtype)
    _segment_mm_backward_B(A, dC, dB, seglen_cpu)
    synchronize(device)
    return dB.detach().cpu()


# ---------------------------------------------------------------------------
# 用例数据定义
# ---------------------------------------------------------------------------

IDTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}


@dataclass
class FwdCase:
    """SegmentMM 前向功能用例。"""
    cid: str
    seglen: List[int]
    idtype: str
    k: int
    n: int
    b_trans: bool
    value: str = "normal"
    seed: int = 0
    desc: str = ""

    @property
    def R(self) -> int:
        return len(self.seglen)

    @property
    def M(self) -> int:
        return sum(self.seglen)

    def build(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idt = IDTYPE_MAP[self.idtype]
        seglen = torch.tensor(self.seglen, dtype=idt)
        A = gen_tensor((self.M, self.k), self.value, seed=self.seed)
        if self.b_trans:
            B = gen_tensor((self.R, self.n, self.k), self.value, seed=self.seed + 1)
        else:
            B = gen_tensor((self.R, self.k, self.n), self.value, seed=self.seed + 1)
        return A, B, seglen


@dataclass
class BwdCase:
    """SegmentMMBackwardB 功能用例。"""
    cid: str
    seglen: List[int]
    idtype: str
    n: int
    m: int
    value: str = "normal"
    seed: int = 0
    desc: str = ""

    @property
    def R(self) -> int:
        return len(self.seglen)

    @property
    def K_sum(self) -> int:
        return sum(self.seglen)

    def build(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idt = IDTYPE_MAP[self.idtype]
        seglen = torch.tensor(self.seglen, dtype=idt)
        A = gen_tensor((self.K_sum, self.n), self.value, seed=self.seed)
        dC = gen_tensor((self.K_sum, self.m), self.value, seed=self.seed + 1)
        return A, dC, seglen


# ---- L0 门槛用例 ----

L0_FWD_CASES: List[FwdCase] = [
    FwdCase("L0-SMM-01", [2, 3], "int32", 4, 6, False, "normal", 1, "int32 双段 b_trans=false"),
    FwdCase("L0-SMM-02", [2, 3], "int64", 4, 6, False, "normal", 2, "int64 seglen"),
    FwdCase("L0-SMM-03", [2, 3], "int32", 4, 6, True, "normal", 3, "b_trans=true 路径"),
    FwdCase("L0-SMM-04", [4], "int32", 4, 6, False, "normal", 4, "R=1 单段退化"),
    FwdCase("L0-SMM-05", [2, 3], "int32", 1, 6, False, "normal", 5, "K=1 外积退化"),
]

L0_BWD_CASES: List[BwdCase] = [
    BwdCase("L0-BWD-01", [2, 3], "int32", 4, 6, "normal", 10, "int32 双段"),
    BwdCase("L0-BWD-02", [2, 3], "int64", 4, 6, "normal", 11, "int64 seglen"),
    BwdCase("L0-BWD-03", [4], "int32", 4, 6, "normal", 12, "R=1 单段退化"),
    BwdCase("L0-BWD-04", [2, 3], "int32", 4, 1, "normal", 13, "M=1 退化"),
]

# ---- L1 功能用例 ----

L1_FWD_CASES: List[FwdCase] = [
    FwdCase("L1-SMM-01", [8, 8], "int32", 64, 64, False, "normal", 20, "等长双段对齐 K/N=64"),
    FwdCase("L1-SMM-02", [1, 3, 2], "int64", 5, 7, False, "normal", 21, "变长三段非对齐K"),
    # 注：方案原文 B[2,5,7] b_trans=true 存在 K 维不匹配（A.K=5, B.shape[2]=7），
    # 意图为非对齐 N=7，故 B 应为 [R,N,K]=[2,7,5]，此处修正。
    FwdCase("L1-SMM-03", [3, 3], "int32", 5, 7, True, "normal", 22, "b_trans=true 非对齐N=7"),
    FwdCase("L1-SMM-04", [2, 1, 4, 3], "int64", 4, 8, True, "normal", 23, "四段变长对齐 b_trans=true"),
    FwdCase("L1-SMM-05", [16], "int32", 64, 64, False, "normal", 24, "R=1 中规模对齐"),
    FwdCase("L1-SMM-06", [16], "int32", 64, 64, True, "normal", 25, "R=1 b_trans=true"),
    FwdCase("L1-SMM-07", [2, 3], "int32", 4, 6, False, "small", 26, "小量级 [1e-3,1e-2]"),
    FwdCase("L1-SMM-08", [2, 3], "int32", 4, 6, False, "large", 27, "偏大量级 [1,10]"),
    FwdCase("L1-SMM-09", [2, 3], "int32", 4, 6, True, "neg", 28, "负数值 b_trans=true"),
    FwdCase("L1-SMM-10", [2, 0, 3], "int32", 4, 6, False, "normal", 29, "seglen 含 0 段（中段 0）"),
]

L1_BWD_CASES: List[BwdCase] = [
    BwdCase("L1-BWD-01", [8, 8], "int32", 64, 64, "normal", 30, "等长双段对齐"),
    BwdCase("L1-BWD-02", [1, 3, 2], "int64", 5, 7, "normal", 31, "变长三段非对齐"),
    BwdCase("L1-BWD-03", [2, 1, 4, 3], "int32", 4, 8, "normal", 32, "四段对齐中规模"),
    BwdCase("L1-BWD-04", [2, 3], "int32", 1, 6, "normal", 33, "N=1 退化"),
    BwdCase("L1-BWD-05", [2, 3], "int32", 4, 6, "small", 34, "小量级数值"),
    BwdCase("L1-BWD-06", [2, 0, 3], "int32", 4, 6, "normal", 35, "seglen 含 0 段"),
]

# ---- L2 功能用例（空 tensor / 特殊值） ----

L2_FWD_FUNC_CASES: List[FwdCase] = [
    FwdCase("L2-SMM-09", [], "int32", 4, 6, False, "normal", 40, "空 tensor R=0"),
    FwdCase("L2-SMM-10", [0, 0], "int32", 4, 6, False, "normal", 41, "空 tensor M=0 全段 0"),
    FwdCase("L2-SMM-11", [2, 3], "int32", 0, 6, False, "normal", 42, "空 tensor K=0"),
    FwdCase("L2-SMM-12", [2, 3], "int32", 4, 0, False, "normal", 43, "空 tensor N=0"),
    FwdCase("L2-SMM-13", [2, 3], "int32", 4, 6, False, "inf", 44, "特殊值 +inf"),
    FwdCase("L2-SMM-14", [2, 3], "int32", 4, 6, False, "nan", 45, "特殊值 nan"),
]

L2_BWD_FUNC_CASES: List[BwdCase] = [
    BwdCase("L2-BWD-04", [], "int32", 4, 6, "normal", 50, "反向空 tensor R=0"),
    BwdCase("L2-BWD-05", [0, 0], "int32", 4, 6, "normal", 51, "反向空 tensor K_sum=0"),
    BwdCase("L2-BWD-06", [2, 3], "int32", 4, 6, "nan", 52, "反向特殊值 nan"),
]

# ---------------------------------------------------------------------------
# 白盒补充用例（源码分支驱动）
#
# 枚举依据（segment_mm_kernel.cpp / segment_mm_cube_processor.h / segment_mm.cc）：
#   WB-F1  AIV no-op 早返回            kernel.cpp:24-27 / 49-51
#   WB-F2  尾核空闲早返回              cube_processor.h:84  (mIdx >= mGrid)
#   WB-F3  空M-block范围早返回        cube_processor.h:90  (myMBlockLo >= myMBlockHi)
#   WB-F4  M-block 尾块 actualRows<16  cube_processor.h:100 (Min(BLOCK_LEN, M-mOff))
#   WB-F5  N-tile 尾块 nChunk<nTile    cube_processor.h:103 (Min(nTile, myNHi-nOff))
#   WB-F6  K-tile 尾块 kChunk<kTile    cube_processor.h:110 (Min(kTile, K-kOff))
#   WB-F7  K 分片累加 initC true→false cube_processor.h:106,123
#   WB-F8  bTrans=false SplitBTranspose cube_processor.h:119 (LoadData3D enTranspose)
#   WB-F9  bTrans=true SplitBNoTranspose cube_processor.h:117 (LoadData2D ifTranspose=false)
#   WB-F10 kFractal 循环 >1 (bTrans)   cube_processor.h:208 (for i<kFractals)
#   WB-F11 host 0 段跳过               segment_mm.cc:116 (m==0||k==0||n==0 continue)
#   WB-F12 host tiling mGrid/nGrid 切分 segment_mm.cc:49-62
#   WB-B1  ~WB-B7  反向对应分支 (cube_processor.h:325-371)
#
# 已有黑盒覆盖情况（gap 分析）：
#   - M<=16 全部：未覆盖 M>16 的多 M-block / M-tail
#   - K<=64 全部：未覆盖 K>64 的 K-split 累加
#   - K<8 (L1-SMM-02 K=5)：未覆盖 K>8 且非 8 对齐的 K-tail
#   - b_trans=true K=4：未覆盖 kFractal 循环 >1 (需 K>8)
#   - N<=64：未覆盖 N>nTile 的 N-tile 尾块
#   - 未覆盖 mBlocks 非整除 mGrid → 尾核空闲早返回
#   - 反向：未覆盖 output row tail / K-split / K-tail / col tile tail / 尾核
# ---------------------------------------------------------------------------

WB_FWD_CASES: List[FwdCase] = [
    # M-block 尾块：M=17 → 2 blocks, 尾块 actualRows=1
    FwdCase("WB-SMM-01", [17], "int32", 8, 16, False, "normal", 200,
            "M=17 M-tail(actualRows=1) + multi M-block + mGrid=2"),
    # 多 M-block 无尾块：M=32 → 2 blocks, 全 actualRows=16
    FwdCase("WB-SMM-02", [32], "int32", 8, 16, False, "normal", 201,
            "M=32 multi M-block no-tail + mGrid=2 multi-core-M"),
    # K-tail + kFractal 循环>1 (bTrans=true)：K=9 → kChunk=9, kFractals=CeilDiv(9,8)=2
    FwdCase("WB-SMM-03", [16], "int32", 9, 16, True, "normal", 202,
            "K=9 K-tail + SplitBNoTranspose kFractal loop>1 (b_trans=true)"),
    # K-split 累加 + K-tail (bTrans=false)：K=65 → kOff=0,kChunk=64,initC=true; kOff=64,kChunk=1,initC=false
    FwdCase("WB-SMM-04", [16], "int32", 65, 16, False, "normal", 203,
            "K=65 K-split(2 iter) + K-accumulate(initC true→false) + K-tail(kChunk=1)"),
    # K-split + bTrans=true：K=65 两路径同时覆盖
    FwdCase("WB-SMM-05", [16], "int32", 65, 16, True, "normal", 204,
            "K=65 K-split + b_trans=true (SplitBNoTranspose + kFractal loop per K iter)"),
    # N-tile 尾块 + full 20-core：M=320(mGrid=20), N=65(nGrid=1, nTile=64) → nChunk=64 then 1
    FwdCase("WB-SMM-06", [320], "int32", 8, 65, False, "normal", 205,
            "M=320 full 20-core + N=65 N-tile tail(nChunk=1) + nGrid=1"),
    # 尾核空闲早返回：M=336(21 blocks), mGrid=20, mBlocksPerCore=2 → core10=[20,21) 1 block, core11-19 空
    FwdCase("WB-SMM-07", [336], "int32", 8, 16, False, "normal", 206,
            "M=336 tail-core(1 block) + idle-core early-return(myMBlockLo>=myMBlockHi)"),
    # 组合尾块 (bTrans=false)：M=17 M-tail + K=9 K-tail + N=17 N partial
    FwdCase("WB-SMM-08", [17], "int32", 9, 17, False, "normal", 207,
            "Combined M-tail(actualRows=1) + K-tail(kChunk=9) + N-partial(nChunk<16)"),
    # 大规模双方向多核 + K-split 精确无尾：M=64(mGrid=4), N=64(nGrid=4), K=128(2×64 exact)
    FwdCase("WB-SMM-09", [64], "int32", 128, 64, False, "normal", 208,
            "M=64 N=64 multi-core both dirs(16 cores) + K=128 K-split exact(no K-tail)"),
]

WB_BWD_CASES: List[BwdCase] = [
    # 输出行尾块：N=17 → 2 output-row blocks, 尾块 actualRows=1
    BwdCase("WB-BWD-01", [8], "int32", 17, 16, "normal", 210,
            "N=17 output-row tail(actualRows=1) + multi M-block + mGrid=2"),
    # K-split + K-tail：K=65 → kOff=0,kChunk=64,initC=true; kOff=64,kChunk=1,initC=false
    BwdCase("WB-BWD-02", [65], "int32", 16, 16, "normal", 211,
            "K=65 K-split(2 iter) + K-accumulate + K-tail(kChunk=1)"),
    # K-tail 非 8 对齐：K=9 → kChunk=9 (SplitATranspose + SplitBTranspose 均 K-tail)
    BwdCase("WB-BWD-03", [9], "int32", 16, 16, "normal", 212,
            "K=9 K-tail(non-8-aligned) in both SplitATranspose + SplitBTranspose"),
    # full 20-core 多块：N=320(mGrid=20), M=16(nGrid=1)
    BwdCase("WB-BWD-04", [8], "int32", 320, 16, "normal", 213,
            "N=320 full 20-core + multi output-row blocks(each 1 block)"),
    # 组合尾块：N=17 row-tail + K=9 K-tail + M=17 col-tile-tail
    BwdCase("WB-BWD-05", [9], "int32", 17, 17, "normal", 214,
            "Combined N-tail(actualRows=1) + K-tail(kChunk=9) + M=17 col-partial"),
    # 尾核空闲早返回：N=336(21 blocks), mGrid=20 → core10 1 block, core11-19 空
    BwdCase("WB-BWD-06", [8], "int32", 336, 16, "normal", 215,
            "N=336 tail-core(1 block) + idle-core early-return"),
    # K-split 精确无尾：K=128 → 2×64 exact, no K-tail
    BwdCase("WB-BWD-07", [128], "int32", 16, 16, "normal", 216,
            "K=128 K-split exact(2×64, no K-tail) + K-accumulate"),
]


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npu_device():
    return get_npu_device(0)


# ---------------------------------------------------------------------------
# L0 门槛用例 —— SegmentMM 前向
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L0_FWD_CASES, ids=[c.cid for c in L0_FWD_CASES])
def test_l0_segment_mm(case: FwdCase, npu_device):
    A_cpu, B_cpu, seglen = case.build()
    golden = segment_mm_golden(A_cpu, B_cpu, seglen, case.b_trans)
    npu_out = call_segment_mm_npu(A_cpu, B_cpu, seglen, case.b_trans, npu_device)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L0 门槛用例 —— SegmentMMBackwardB
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L0_BWD_CASES, ids=[c.cid for c in L0_BWD_CASES])
def test_l0_segment_mm_backward_b(case: BwdCase, npu_device):
    A_cpu, dC_cpu, seglen = case.build()
    golden = segment_mm_backward_b_golden(A_cpu, dC_cpu, seglen)
    npu_out = call_segment_mm_backward_b_npu(A_cpu, dC_cpu, seglen, npu_device)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L1 功能用例 —— SegmentMM 前向
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L1_FWD_CASES, ids=[c.cid for c in L1_FWD_CASES])
def test_l1_segment_mm(case: FwdCase, npu_device):
    A_cpu, B_cpu, seglen = case.build()
    golden = segment_mm_golden(A_cpu, B_cpu, seglen, case.b_trans)
    npu_out = call_segment_mm_npu(A_cpu, B_cpu, seglen, case.b_trans, npu_device)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L1 功能用例 —— SegmentMMBackwardB
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L1_BWD_CASES, ids=[c.cid for c in L1_BWD_CASES])
def test_l1_segment_mm_backward_b(case: BwdCase, npu_device):
    A_cpu, dC_cpu, seglen = case.build()
    golden = segment_mm_backward_b_golden(A_cpu, dC_cpu, seglen)
    npu_out = call_segment_mm_backward_b_npu(A_cpu, dC_cpu, seglen, npu_device)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L2 空张量 / 特殊值功能用例 —— SegmentMM 前向
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L2_FWD_FUNC_CASES, ids=[c.cid for c in L2_FWD_FUNC_CASES])
def test_l2_segment_mm_func(case: FwdCase, npu_device):
    A_cpu, B_cpu, seglen = case.build()
    golden = segment_mm_golden(A_cpu, B_cpu, seglen, case.b_trans)
    npu_out = call_segment_mm_npu(A_cpu, B_cpu, seglen, case.b_trans, npu_device)
    if case.value in ("inf", "nan"):
        # nan/inf 等价判定（assert_mixed_tolerance 内部已分离特殊值路径）
        assert_mixed_tolerance(npu_out, golden)
    elif case.k == 0 or case.n == 0 or case.M == 0:
        # 空输出（K=0 全零、N=0 空列、M=0 空行）断言 shape 一致且空/全零
        assert_empty_or_allzero(npu_out, golden)
        # K=0 时输出全零但也做一次混合容差（全零应通过）
        assert_mixed_tolerance(npu_out, golden)
    else:
        assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L2 空张量 / 特殊值功能用例 —— SegmentMMBackwardB
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L2_BWD_FUNC_CASES, ids=[c.cid for c in L2_BWD_FUNC_CASES])
def test_l2_segment_mm_backward_b_func(case: BwdCase, npu_device):
    A_cpu, dC_cpu, seglen = case.build()
    golden = segment_mm_backward_b_golden(A_cpu, dC_cpu, seglen)
    npu_out = call_segment_mm_backward_b_npu(A_cpu, dC_cpu, seglen, npu_device)
    if case.value == "nan":
        assert_mixed_tolerance(npu_out, golden)
    elif case.K_sum == 0 or case.R == 0:
        assert_empty_or_allzero(npu_out, golden)
        assert_mixed_tolerance(npu_out, golden)
    else:
        assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# 白盒用例 —— SegmentMM 前向（源码分支驱动）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", WB_FWD_CASES, ids=[c.cid for c in WB_FWD_CASES])
def test_wb_segment_mm(case: FwdCase, npu_device):
    A_cpu, B_cpu, seglen = case.build()
    golden = segment_mm_golden(A_cpu, B_cpu, seglen, case.b_trans)
    npu_out = call_segment_mm_npu(A_cpu, B_cpu, seglen, case.b_trans, npu_device)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# 白盒用例 —— SegmentMMBackwardB（源码分支驱动）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", WB_BWD_CASES, ids=[c.cid for c in WB_BWD_CASES])
def test_wb_segment_mm_backward_b(case: BwdCase, npu_device):
    A_cpu, dC_cpu, seglen = case.build()
    golden = segment_mm_backward_b_golden(A_cpu, dC_cpu, seglen)
    npu_out = call_segment_mm_backward_b_npu(A_cpu, dC_cpu, seglen, npu_device)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L2 异常用例 —— SegmentMM 前向参数校验
# ---------------------------------------------------------------------------

def _build_normal_fwd_inputs(idtype="int32", dtype=torch.float32):
    seglen = torch.tensor([2, 3], dtype=IDTYPE_MAP[idtype])
    A = gen_tensor((5, 4), "normal", dtype=dtype, seed=100)
    B = gen_tensor((2, 4, 6), "normal", dtype=dtype, seed=101)
    return A, B, seglen


def test_l2_smm_01_half_dtype(npu_device):
    """L2-SMM-01: DType=float16 → LOG(FATAL)/报错。"""
    A, B, seglen = _build_normal_fwd_inputs(dtype=torch.float16)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.float16)
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    with pytest.raises(Exception):
        _segment_mm(A_npu, B_npu, C, seglen, b_trans=False)


def test_l2_smm_02_bfloat16_dtype(npu_device):
    """L2-SMM-02: DType=bfloat16 → LOG(FATAL)/报错。"""
    A, B, seglen = _build_normal_fwd_inputs(dtype=torch.bfloat16)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.bfloat16)
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    with pytest.raises(Exception):
        _segment_mm(A_npu, B_npu, C, seglen, b_trans=False)


def test_l2_smm_03_double_dtype(npu_device):
    """L2-SMM-03: DType=float64 → LOG(FATAL)/报错（Ascend 无 double GEMM）。"""
    A, B, seglen = _build_normal_fwd_inputs(dtype=torch.float64)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.float64)
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    with pytest.raises(Exception):
        _segment_mm(A_npu, B_npu, C, seglen, b_trans=False)


def test_l2_smm_04_a_trans_true(npu_device):
    """L2-SMM-04: a_trans=true → dispatch CHECK(!A_trans) 失败，报错。"""
    A, B, seglen = _build_normal_fwd_inputs()
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _CAPI_DGLKernelSEGMENTMM(
            to_dgl_nd(A_npu), to_dgl_nd(B_npu), to_dgl_nd_for_write(C),
            to_dgl_nd(seglen), True, False,
        )


def test_l2_smm_05_seglen_on_device(npu_device):
    """L2-SMM-05: seglen_A 在 Device → CHECK_EQ(ctx, kDGLCPU) 失败，报错。"""
    A, B, seglen = _build_normal_fwd_inputs()
    seglen_dev = seglen.to(npu_device)
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _segment_mm(A_npu, B_npu, C, seglen_dev, b_trans=False)


def test_l2_smm_06_b_seglen_mismatch(npu_device):
    """L2-SMM-06: B.shape[0]=3 != R=2（seglen 长度）→ CHECK 失败。"""
    seglen = torch.tensor([2, 3], dtype=torch.int32)
    A = gen_tensor((5, 4), "normal", seed=100)
    B = gen_tensor((3, 4, 6), "normal", seed=101)  # R=3 != seglen 长度 2
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _segment_mm(A_npu, B_npu, C, seglen, b_trans=False)


def test_l2_smm_07_k_mismatch_no_trans(npu_device):
    """L2-SMM-07: b_trans=false 时 A.shape[1]=4 != B.shape[1]=5 → CHECK 失败。"""
    seglen = torch.tensor([2, 3], dtype=torch.int32)
    A = gen_tensor((5, 4), "normal", seed=100)
    B = gen_tensor((2, 5, 6), "normal", seed=101)  # K=5 != A.K=4
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _segment_mm(A_npu, B_npu, C, seglen, b_trans=False)


def test_l2_smm_08_k_mismatch_trans(npu_device):
    """L2-SMM-08: b_trans=true 时 A.shape[1]=4 != B.shape[2]=5 → CHECK 失败。"""
    seglen = torch.tensor([2, 3], dtype=torch.int32)
    A = gen_tensor((5, 4), "normal", seed=100)
    # b_trans=true: B[R, N, K] = [2, 6, 5]，K=5 != A.K=4
    B = gen_tensor((2, 6, 5), "normal", seed=101)
    A_npu, B_npu = A.to(npu_device), B.to(npu_device)
    C = torch.empty((5, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _segment_mm(A_npu, B_npu, C, seglen, b_trans=True)


# ---------------------------------------------------------------------------
# L2 异常用例 —— SegmentMMBackwardB 参数校验
# ---------------------------------------------------------------------------

def test_l2_bwd_01_half_dtype(npu_device):
    """L2-BWD-01: 反向 DType=float16 → LOG(FATAL)/报错。"""
    seglen = torch.tensor([2, 3], dtype=torch.int32)
    A = gen_tensor((5, 4), "normal", dtype=torch.float16, seed=200)
    dC = gen_tensor((5, 6), "normal", dtype=torch.float16, seed=201)
    dB = torch.empty((2, 4, 6), device=npu_device, dtype=torch.float16)
    A_npu, dC_npu = A.to(npu_device), dC.to(npu_device)
    with pytest.raises(Exception):
        _segment_mm_backward_B(A_npu, dC_npu, dB, seglen)


def test_l2_bwd_02_seglen_on_device(npu_device):
    """L2-BWD-02: 反向 seglen 在 Device → CHECK 失败。"""
    seglen = torch.tensor([2, 3], dtype=torch.int32)
    A = gen_tensor((5, 4), "normal", seed=200)
    dC = gen_tensor((5, 6), "normal", seed=201)
    seglen_dev = seglen.to(npu_device)
    A_npu, dC_npu = A.to(npu_device), dC.to(npu_device)
    dB = torch.empty((2, 4, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _segment_mm_backward_B(A_npu, dC_npu, dB, seglen_dev)


def test_l2_bwd_03_ksum_mismatch(npu_device):
    """L2-BWD-03: A.shape[0]=5 != dC.shape[0]=4 → K_sum 不一致 CHECK 失败。"""
    seglen = torch.tensor([2, 3], dtype=torch.int32)
    A = gen_tensor((5, 4), "normal", seed=200)
    dC = gen_tensor((4, 6), "normal", seed=201)  # K_sum=4 != A.shape[0]=5
    A_npu, dC_npu = A.to(npu_device), dC.to(npu_device)
    dB = torch.empty((2, 4, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _segment_mm_backward_B(A_npu, dC_npu, dB, seglen)


# ---------------------------------------------------------------------------
# 用例汇总（供 pytest collection 核对，61 = L0:9 + L1:16 + L2 func:9 + L2 异常:11 + WB:16）
# ---------------------------------------------------------------------------

ALL_CASE_IDS = (
    [c.cid for c in L0_FWD_CASES]
    + [c.cid for c in L0_BWD_CASES]
    + [c.cid for c in L1_FWD_CASES]
    + [c.cid for c in L1_BWD_CASES]
    + [c.cid for c in L2_FWD_FUNC_CASES]
    + [c.cid for c in L2_BWD_FUNC_CASES]
    + [c.cid for c in WB_FWD_CASES]
    + [c.cid for c in WB_BWD_CASES]
    + [
        "L2-SMM-01", "L2-SMM-02", "L2-SMM-03", "L2-SMM-04",
        "L2-SMM-05", "L2-SMM-06", "L2-SMM-07", "L2-SMM-08",
        "L2-BWD-01", "L2-BWD-02", "L2-BWD-03",
    ]
)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
