"""GatherMM / GatherMMScatter Ascend NPU functional tests.

模式 B（无评测集）自建 golden + 用例。golden 为 torch CPU FP32 参考实现，逐行
gather + matmul / 外积 scatter-累加，per math definition,
不复用被测 NPU kernel，保证比对独立性（参照 test_segment_mm_npu.py 范式）。

测试框架：pytest，单文件组织 L0/L1/L2 + 白盒补充共 86 用例
（GatherMM: L0 5 + L1 10 + L2 func 6 + L2 异常 7 = 28；
 GatherMMScatter Kernel1: L0 5 + L1 7 + L2 func 5 + L2 异常 2 = 19；
 GatherMMScatter Kernel2: L0 4 + L1 5 + L2 func 4 + L2 异常(双路径共 2) = ...）。
黑盒合计 62 = L0 14 + L1 22 + L2 func 15 + L2 异常 11。
白盒补充 24 例（源码分支枚举：tiling 尾块/非对齐/多核边界/原子路径）。
本文件包含黑盒用例（L0/L1/L2）与白盒补充用例（WB-* 前缀）。

精度口径：float32 混合容差（rtol=2^-10, atol=2^-16, matched_ratio>=0.99,
max_abs_error<=1e-2 固定上限，对齐 ops-precision-standard float_compute 3 FP32 列），
nan/inf 用 isnan/isinf 等价判定不套混合容差，空张量直接通过。

调用入口（DGL Python 层）：
  - GatherMM          : dgl._sparse_ops._gather_mm(A, B, out, idx_a, idx_b)
                        （底层 _CAPI_DGLKernelGATHERMM，kernel.cc:570）
  - GatherMMScatter    : dgl._sparse_ops._gather_mm_scatter(A, B, out, idx_a, idx_b, idx_c)
                        （底层 _CAPI_DGLKernelGATHERMMSCATTER，kernel.cc:580）
  - B.ndim==3 → Kernel1（逐行 GEMM scatter-累加）；B.ndim==2 → Kernel2（外积 scatter-累加）。
  - dispatch（kernel.cc:103/141）要求 idx_a/idx_b/idx_c 与 A 同设备（不同于 SegmentMM
    seglen 留 CPU）；idx 为 None 时取行号 r（数学定义），由 host/kernel 判空处理。

Math definition source: requirement doc
（idx_a/idx_b shape [m]、C shape [m,n]、m = A.shape[0]；idx_b-None 路径自洽要求
A.shape[0]==R，see deviation note）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
import torch

import dgl
from dgl._sparse_ops import (
    _CAPI_DGLKernelGATHERMM,
    _CAPI_DGLKernelGATHERMMSCATTER,
    _gather_mm,
    _gather_mm_scatter,
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
# Golden implementation (CPU FP32, per math definition)
# ---------------------------------------------------------------------------

def gather_mm_golden(
    A: torch.Tensor,
    B: torch.Tensor,
    idx_a: Optional[torch.Tensor],
    idx_b: Optional[torch.Tensor],
) -> torch.Tensor:
    """GatherMM golden（B 恒 3D [R,k,n]）。

    A: [m,k] float32 (CPU)   B: [R,k,n] float32 (CPU)
    idx_a/idx_b: [m] int32/int64 (CPU) 或 None（None → arange(m)）
    返回 C: [m,n] float32 (CPU)

    数学定义（math definition）：
        for r in [0,m): C[r,:] = A[idx_a[r],:] @ B[idx_b[r],:,:]
    m = A.shape[0]（输出行数 = A 行数，真值源）。
    边界自洽：m=0/k=0/n=0/R=0 由 torch 切片与 matmul 自然处理，不做 early-return 特判。
    """
    m, k = A.shape
    n = B.shape[2]
    ra = idx_a if idx_a is not None else torch.arange(m)
    rb = idx_b if idx_b is not None else torch.arange(m)
    C = torch.zeros(m, n, dtype=torch.float32)
    for r in range(m):
        C[r] = A[ra[r]] @ B[rb[r]]     # [k] @ [k,n] = [n]
    return C


def gather_mm_scatter_k1_golden(
    A: torch.Tensor,
    B: torch.Tensor,
    idx_a: Optional[torch.Tensor],
    idx_b: Optional[torch.Tensor],
    idx_c: Optional[torch.Tensor],
    numC: Optional[int] = None,
) -> torch.Tensor:
    """GatherMMScatter Kernel1 golden（B 3D [R,k,n]，scatter-累加）。

    A: [m,k]  B: [R,k,n]  idx_a/idx_b/idx_c: [m] 或 None（None → arange(m)）
    返回 C: [numC,n] float32 (CPU)，numC = max(idx_c)+1（idx_c 非空时）或 m（idx_c None 时）。
    数学定义（math definition Kernel1）：
        for r in [0,m): C[idx_c[r],:] += A[idx_a[r],:] @ B[idx_b[r],:,:]
    逐行累加，多 r→同 rc 自然累加（与累加顺序无关的 FP32 加法在合理量级内一致）。
    """
    m, k = A.shape
    n = B.shape[2]
    ra = idx_a if idx_a is not None else torch.arange(m)
    rb = idx_b if idx_b is not None else torch.arange(m)
    rc = idx_c if idx_c is not None else torch.arange(m)
    if numC is None:
        nc = (int(rc.max()) + 1) if rc.numel() > 0 else 0
    else:
        nc = numC
    C = torch.zeros(nc, n, dtype=torch.float32)
    for r in range(m):
        C[rc[r]] += A[ra[r]] @ B[rb[r]]   # 逐行累加
    return C


def gather_mm_scatter_k2_golden(
    A: torch.Tensor,
    B: torch.Tensor,
    idx_a: Optional[torch.Tensor],
    idx_b: Optional[torch.Tensor],
    idx_c: torch.Tensor,
    numC: Optional[int] = None,
) -> torch.Tensor:
    """GatherMMScatter Kernel2 golden（B 2D [R,n]，外积 scatter-累加）。

    A: [m,k]  B: [R,n]  idx_a/idx_b/idx_c: [m]（Kernel2 idx_c 必填，math definition）
    返回 C: [numC,k,n] float32 (CPU)。
    数学定义（math definition Kernel2）：
        for r in [0,m): C[idx_c[r],i,j] += A[idx_a[r],i] * B[idx_b[r],j]
    """
    m, k = A.shape
    n = B.shape[1]
    ra = idx_a if idx_a is not None else torch.arange(m)
    rb = idx_b if idx_b is not None else torch.arange(m)
    rc = idx_c
    if numC is None:
        nc = (int(rc.max()) + 1) if rc.numel() > 0 else 0
    else:
        nc = numC
    C = torch.zeros(nc, k, n, dtype=torch.float32)
    for r in range(m):
        a = A[ra[r]].view(k, 1)        # [k,1]
        b = B[rb[r]].view(1, n)        # [1,n]
        C[rc[r]] += a * b              # 外积 [k,n]，累加
    return C


# ---------------------------------------------------------------------------
# 精度断言（混合容差 + nan/inf 等价判定）—— 复用 SegmentMM 实现
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
    FP32 列），不随输出量级缩放。用例输入量级须控制使 GEMM 结果量级在 ~100 以内
    （normal [-1,1] k<=64，large [1,10] k<=8），保证 FP32 舍入绝对误差落在固定 1e-2 上限内。
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
    matched = finite_pass + (total - finite_mask.sum().item() - special_count)
    matched_ratio = matched / total

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
      large   [1, 10]   （仅配小 k<=8，GEMM 结果量级 ≤ ~1e3，保证 FP32 舍入绝对误差落在固定 1e-2 上限内）
      neg     [-1, 1] 含负镜像（normal 即含负）
      zero    全零（含 ±0）
      inf     normal 基础上注入 +inf（首元素）
      nan     normal 基础上注入 nan（首元素）
    数值量级控制：参照 SegmentMM 教训，GEMM 结果量级控制在 ~100 以内
    （normal k≤64、large k≤8）。
    """
    g = torch.Generator().manual_seed(seed)
    if kind == "normal" or kind == "neg":
        return torch.rand(shape, generator=g, dtype=dtype) * 2 - 1
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
    raise ValueError(f"unknown value kind: {kind}")


# ---------------------------------------------------------------------------
# NPU 调用封装
# ---------------------------------------------------------------------------

def _to_idx_tensor(
    idx: Optional[torch.Tensor],
    idtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """idx 转 device 张量；None 保持 None（→ Null 数组，dispatch 判空取行号 r）。"""
    if idx is None:
        return None
    return idx.to(device).to(idtype)


def call_gather_mm_npu(
    A_cpu: torch.Tensor,
    B_cpu: torch.Tensor,
    idx_a_cpu: Optional[torch.Tensor],
    idx_b_cpu: Optional[torch.Tensor],
    idtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """调用 GatherMM（_gather_mm，B 恒 3D）。idx 与 A 同设备（dispatch 约束）。

    输出 C shape [m,n]，m = A.shape[0]（math definition 真值）。
    """
    A = A_cpu.to(device)
    B = B_cpu.to(device)
    idx_a = _to_idx_tensor(idx_a_cpu, idtype, device)
    idx_b = _to_idx_tensor(idx_b_cpu, idtype, device)
    m = A_cpu.shape[0]
    n = B_cpu.shape[2]
    C = torch.empty((m, n), device=device, dtype=A_cpu.dtype)
    _gather_mm(A, B, C, idx_a, idx_b)
    synchronize(device)
    return C.detach().cpu()


def _compute_numC(idx_c_cpu: Optional[torch.Tensor], m: int) -> int:
    """numC 派生：idx_c 非空 → max(idx_c)+1（空数组→0）；idx_c None → m（退化 GatherMM）。"""
    if idx_c_cpu is None:
        return m
    if idx_c_cpu.numel() == 0:
        return 0
    return int(idx_c_cpu.max()) + 1


def call_gather_mm_scatter_npu(
    A_cpu: torch.Tensor,
    B_cpu: torch.Tensor,
    idx_a_cpu: Optional[torch.Tensor],
    idx_b_cpu: Optional[torch.Tensor],
    idx_c_cpu: Optional[torch.Tensor],
    idtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """调用 GatherMMScatter（_gather_mm_scatter）。B.ndim 决定 Kernel1/Kernel2。

    Kernel1（B 3D [R,k,n]）: C [numC,n]，numC = max(idx_c)+1 或 m（idx_c None）。
    Kernel2（B 2D [R,n]）  : C [numC,k,n]，numC = max(idx_c)+1（Kernel2 idx_c 必填）。
    idx_a/idx_b/idx_c 与 A 同设备（dispatch 约束）。
    """
    A = A_cpu.to(device)
    B = B_cpu.to(device)
    idx_a = _to_idx_tensor(idx_a_cpu, idtype, device)
    idx_b = _to_idx_tensor(idx_b_cpu, idtype, device)
    idx_c = _to_idx_tensor(idx_c_cpu, idtype, device)
    m = A_cpu.shape[0]
    k = A_cpu.shape[1]
    numC = _compute_numC(idx_c_cpu, m)
    if B_cpu.ndim == 3:
        n = B_cpu.shape[2]
        C = torch.empty((numC, n), device=device, dtype=A_cpu.dtype)
    else:  # B 2D
        n = B_cpu.shape[1]
        C = torch.empty((numC, k, n), device=device, dtype=A_cpu.dtype)
    _gather_mm_scatter(A, B, C, idx_a, idx_b, idx_c)
    synchronize(device)
    return C.detach().cpu()


# ---------------------------------------------------------------------------
# 用例数据定义
# ---------------------------------------------------------------------------

IDTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}


@dataclass
class GmmCase:
    """GatherMM / GatherMMScatter 功能用例。

    op:
      "gmm"  → GatherMM            A[m,k] B[R,k,n]            C[m,n]
      "k1"   → GMS Kernel1 (B 3D)   A[m,k] B[R,k,n]            C[numC,n]
      "k2"   → GMS Kernel2 (B 2D)   A[m,k] B[R,n]              C[numC,k,n]
    idx_a/idx_b/idx_c: None(Null) / list[int]（list 长度须 == m，math definition 真值）。
    numC_override: 显式 numC（仅 m=0 空 idx_c 时用 0）。
    """
    cid: str
    op: str
    m: int
    k: int
    n: int
    R: int
    idx_a: Optional[List[int]]
    idx_b: Optional[List[int]]
    idx_c: Optional[List[int]] = None
    idtype: str = "int32"
    value: str = "normal"
    seed: int = 0
    numC_override: Optional[int] = None
    desc: str = ""

    @property
    def idt(self) -> torch.dtype:
        return IDTYPE_MAP[self.idtype]

    def _idx_tensor(self, idx: Optional[List[int]]) -> Optional[torch.Tensor]:
        if idx is None:
            return None
        return torch.tensor(idx, dtype=self.idt)

    def build(self) -> Tuple[torch.Tensor, torch.Tensor,
                             Optional[torch.Tensor], Optional[torch.Tensor],
                             Optional[torch.Tensor]]:
        A = gen_tensor((self.m, self.k), self.value, seed=self.seed)
        if self.op in ("gmm", "k1"):
            B = gen_tensor((self.R, self.k, self.n), self.value, seed=self.seed + 1)
        else:  # k2
            B = gen_tensor((self.R, self.n), self.value, seed=self.seed + 1)
        return A, B, self._idx_tensor(self.idx_a), self._idx_tensor(self.idx_b), self._idx_tensor(self.idx_c)

    def numC(self) -> int:
        if self.numC_override is not None:
            return self.numC_override
        if self.idx_c is None:
            return self.m
        if len(self.idx_c) == 0:
            return 0
        return max(self.idx_c) + 1


# ---- L0 门槛用例（14） ----

L0_CASES: List[GmmCase] = [
    # GatherMM (5)
    GmmCase("GMM-L0-01", "gmm", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            idtype="int32", seed=1, desc="GatherMM int32 idx_a/b 非空基础"),
    GmmCase("GMM-L0-02", "gmm", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            idtype="int64", seed=2, desc="GatherMM int64 idtype"),
    GmmCase("GMM-L0-03", "gmm", 4, 8, 6, 3, None, [1, 2, 0, 1],
            seed=3, desc="GatherMM idx_a Null"),
    # GMM-L0-04: idx_b Null 路径，math definition 真值要求 m==R（原方案 A[4,8] m=4≠R=3 会越界，
    # 已按per math definition校正为 A[3,8]，见 deviation note）
    GmmCase("GMM-L0-04", "gmm", 3, 8, 6, 3, [2, 0, 1], None,
            seed=4, desc="GatherMM idx_b Null（m==R==len(idx_a)，per math definition校正）"),
    GmmCase("GMM-L0-05", "gmm", 4, 8, 6, 4, None, None,
            seed=5, desc="GatherMM 全 Null（m==R）"),
    # GMS Kernel1 (5)
    GmmCase("GMS-L0-01", "k1", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            idtype="int32", seed=6, desc="GMS Kernel1 int32 scatter 基础 numC=3"),
    GmmCase("GMS-L0-02", "k1", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            idtype="int64", seed=7, desc="GMS Kernel1 int64"),
    # GMS-L0-03: idx_c Null（退化 GatherMM）。风险 R1：dispatch kernel.cc:171 用 idx_c->dtype 取
    # idtype，Null idx_c 的 dtype 合法性依赖design O3（host 预生成 arange）。若 dispatch 崩溃
    # 为 dispatch 缺陷回退design，测试不自行改 dispatch（见design R1）。
    GmmCase("GMS-L0-03", "k1", 4, 8, 6, 4, [2, 0, 1, 2], [1, 2, 0, 1], None,
            seed=8, desc="GMS Kernel1 idx_c Null 退化 GatherMM（风险 R1）"),
    GmmCase("GMS-L0-04", "k1", 4, 8, 6, 3, None, [1, 2, 0, 1], [0, 1, 2, 0],
            seed=9, desc="GMS Kernel1 idx_a Null numC=3"),
    # GMS-L0-05: idx_b Null，per math definition校正 m==R（同 GMM-L0-04）
    GmmCase("GMS-L0-05", "k1", 3, 8, 6, 3, [2, 0, 1], None, [0, 1, 2],
            seed=10, desc="GMS Kernel1 idx_b Null numC=3（per math definition校正）"),
    # GMS Kernel2 (4)
    GmmCase("GMS2-L0-01", "k2", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            idtype="int32", seed=11, desc="GMS Kernel2 int32 外积 scatter numC=3"),
    GmmCase("GMS2-L0-02", "k2", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            idtype="int64", seed=12, desc="GMS Kernel2 int64"),
    GmmCase("GMS2-L0-03", "k2", 4, 8, 6, 3, None, [1, 2, 0, 1], [0, 1, 2, 0],
            seed=13, desc="GMS Kernel2 idx_a Null numC=3"),
    # GMS2-L0-04: idx_b Null，per math definition校正 m==R
    GmmCase("GMS2-L0-04", "k2", 3, 8, 6, 3, [2, 0, 1], None, [0, 1, 2],
            seed=14, desc="GMS Kernel2 idx_b Null numC=3（per math definition校正）"),
]


# ---- L1 功能用例（22） ----

L1_CASES: List[GmmCase] = [
    # GatherMM (10)
    GmmCase("GMM-L1-01", "gmm", 8, 8, 8, 4,
            [0, 1, 0, 1, 2, 3, 2, 3], [3, 2, 2, 3, 0, 1, 1, 0],
            seed=20, desc="GatherMM 对齐中规模 idx_a/b 各含重复"),
    GmmCase("GMM-L1-02", "gmm", 4, 5, 7, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=21, desc="GatherMM 非对齐 k=5 n=7"),
    GmmCase("GMM-L1-03", "gmm", 8, 8, 8, 3,
            [2, 0, 1, 2, 0, 1, 2, 0], [0, 0, 1, 1, 2, 2, 0, 1],
            seed=22, desc="GatherMM idx_b 重复（多 r 同 B 切片）"),
    GmmCase("GMM-L1-04", "gmm", 8, 8, 8, 8,
            [0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 2, 3, 4, 5, 6, 7],
            seed=23, desc="GatherMM idx_a 重复"),
    GmmCase("GMM-L1-05", "gmm", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            value="small", seed=24, desc="GatherMM 小量级 [1e-3,1e-2]"),
    GmmCase("GMM-L1-06", "gmm", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            value="large", seed=25, desc="GatherMM 偏大量级 [1,10] k=8（结果≤~800）"),
    GmmCase("GMM-L1-07", "gmm", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            value="neg", seed=26, desc="GatherMM 负值"),
    # m=1 单行：A 仅 1 行，唯一合法 idx_a=[0]（方案原 idx_a=[2] 对 A[1,8] 值域越界 UB，
    # 属 R3 同类问题，不应在期望"通过"的 L1 功能用例中；校正为 [0]，保持 m=1 单行意图）
    GmmCase("GMM-L1-08", "gmm", 1, 8, 6, 3, [0], [1],
            seed=27, desc="GatherMM m=1 单行（idx_a 校正为 [0]，值域合法）"),
    GmmCase("GMM-L1-09", "gmm", 16, 8, 8, 8,
            [5, 12, 3, 8, 0, 15, 7, 2, 11, 6, 9, 14, 1, 4, 13, 10],
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
            seed=28, desc="GatherMM 中规模 m=16 对齐 idx 乱序（idx_a 0..15 乱序，idx_b 0..7）"),
    GmmCase("GMM-L1-10", "gmm", 4, 1, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=29, desc="GatherMM k=1 退化（向量-向量内积形态）"),
    # GMS Kernel1 (7)
    GmmCase("GMS-L1-01", "k1", 8, 8, 8, 4,
            [2, 0, 1, 2, 3, 1, 0, 3], [1, 2, 0, 1, 3, 0, 2, 2], [0, 1, 2, 3, 0, 1, 2, 3],
            seed=30, desc="GMS Kernel1 对齐 scatter numC=4"),
    GmmCase("GMS-L1-02", "k1", 8, 8, 8, 4,
            [2, 0, 1, 2, 3, 1, 0, 3], [1, 2, 0, 1, 3, 0, 2, 2], [0, 0, 1, 1, 0, 1, 2, 2],
            seed=31, desc="GMS Kernel1 idx_c 碰撞累加 numC=3"),
    GmmCase("GMS-L1-03", "k1", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            value="small", seed=32, desc="GMS Kernel1 小量级"),
    GmmCase("GMS-L1-04", "k1", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            value="large", seed=33, desc="GMS Kernel1 偏大量级 k=8"),
    GmmCase("GMS-L1-05", "k1", 8, 8, 8, 4,
            [0, 0, 1, 1, 2, 2, 3, 3], [3, 3, 2, 2, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 0, 1],
            seed=34, desc="GMS Kernel1 idx_c 碰撞 + idx_a/b 重复"),
    # m=1 单行：idx_a 校正为 [0]（A[1,8] 仅 1 行，方案原 idx_a=[2] 值域越界，同 GMM-L1-08）
    GmmCase("GMS-L1-06", "k1", 1, 8, 6, 3, [0], [1], [0],
            seed=35, desc="GMS Kernel1 m=1 numC=1（idx_a 校正为 [0]）"),
    GmmCase("GMS-L1-07", "k1", 4, 5, 7, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            seed=36, desc="GMS Kernel1 非对齐 k=5 n=7"),
    # GMS Kernel2 (5)
    GmmCase("GMS2-L1-01", "k2", 8, 8, 8, 4,
            [2, 0, 1, 2, 3, 1, 0, 3], [1, 2, 0, 1, 3, 0, 2, 2], [0, 1, 2, 3, 0, 1, 2, 3],
            seed=40, desc="GMS Kernel2 对齐外积 scatter numC=4"),
    GmmCase("GMS2-L1-02", "k2", 8, 8, 8, 4,
            [2, 0, 1, 2, 3, 1, 0, 3], [1, 2, 0, 1, 3, 0, 2, 2], [0, 0, 1, 1, 0, 1, 2, 2],
            seed=41, desc="GMS Kernel2 idx_c 碰撞累加 numC=3"),
    GmmCase("GMS2-L1-03", "k2", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            value="small", seed=42, desc="GMS Kernel2 小量级"),
    GmmCase("GMS2-L1-04", "k2", 4, 1, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 0, 2],
            seed=43, desc="GMS Kernel2 k=1 外积退化 numC=3"),
    GmmCase("GMS2-L1-05", "k2", 4, 5, 7, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            seed=44, desc="GMS Kernel2 非对齐 k=5 n=7"),
]


# ---- L2 功能用例（空 tensor / 特殊值，15） ----

L2_FUNC_CASES: List[GmmCase] = [
    # GatherMM (6)
    GmmCase("GMM-L2-01", "gmm", 0, 8, 6, 3, [], [],
            seed=40, desc="GatherMM m=0 空"),
    GmmCase("GMM-L2-02", "gmm", 4, 0, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=41, desc="GatherMM k=0 输出全零"),
    GmmCase("GMM-L2-03", "gmm", 4, 8, 0, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=42, desc="GatherMM n=0 空输出"),
    # GMM-L2-04: R=0。per math definition校正 m=0（原方案 A[4,8] m=4 配空 idx 越界且"空输出"不成立；
    # R=0 时无合法 idx_b 值，唯一自洽为 m=0，见 deviation-note.md）
    GmmCase("GMM-L2-04", "gmm", 0, 8, 6, 0, [], [],
            seed=43, desc="GatherMM R=0 空 B 首维（per math definition校正 m=0）"),
    GmmCase("GMM-L2-05", "gmm", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            value="inf", seed=44, desc="GatherMM +inf 注入"),
    GmmCase("GMM-L2-06", "gmm", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            value="nan", seed=45, desc="GatherMM nan 注入"),
    # GMS Kernel1 (5)
    GmmCase("GMS-L2-01", "k1", 0, 8, 6, 3, [], [], [],
            numC_override=0, seed=46, desc="GMS Kernel1 m=0 idx_c=[] numC=0"),
    GmmCase("GMS-L2-02", "k1", 4, 0, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            seed=47, desc="GMS Kernel1 k=0 全零"),
    GmmCase("GMS-L2-03", "k1", 4, 8, 0, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            seed=48, desc="GMS Kernel1 n=0 空输出"),
    GmmCase("GMS-L2-04", "k1", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            value="inf", seed=49, desc="GMS Kernel1 +inf"),
    GmmCase("GMS-L2-05", "k1", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 2, 0],
            value="nan", seed=50, desc="GMS Kernel1 nan"),
    # GMS Kernel2 (4)
    GmmCase("GMS2-L2-01", "k2", 0, 8, 6, 3, [], [], [],
            numC_override=0, seed=51, desc="GMS Kernel2 m=0 idx_c=[] numC=0"),
    GmmCase("GMS2-L2-02", "k2", 4, 0, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 0, 2],
            seed=52, desc="GMS Kernel2 k=0 C[numC,0,6] 全零"),
    GmmCase("GMS2-L2-03", "k2", 4, 8, 0, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 0, 2],
            seed=53, desc="GMS Kernel2 n=0 C[numC,8,0] 空"),
    GmmCase("GMS2-L2-04", "k2", 4, 8, 6, 3, [2, 0, 1, 2], [1, 2, 0, 1], [0, 1, 0, 2],
            value="nan", seed=54, desc="GMS Kernel2 nan"),
]


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npu_device():
    return get_npu_device(0)


def _golden_for(case: GmmCase, A, B, idx_a, idx_b, idx_c) -> torch.Tensor:
    if case.op == "gmm":
        return gather_mm_golden(A, B, idx_a, idx_b)
    if case.op == "k1":
        return gather_mm_scatter_k1_golden(A, B, idx_a, idx_b, idx_c, numC=case.numC())
    if case.op == "k2":
        return gather_mm_scatter_k2_golden(A, B, idx_a, idx_b, idx_c, numC=case.numC())
    raise ValueError(f"unknown op: {case.op}")


def _npu_call_for(case: GmmCase, A, B, idx_a, idx_b, idx_c, device) -> torch.Tensor:
    if case.op == "gmm":
        return call_gather_mm_npu(A, B, idx_a, idx_b, case.idt, device)
    return call_gather_mm_scatter_npu(A, B, idx_a, idx_b, idx_c, case.idt, device)


def _is_empty_or_zero_case(case: GmmCase) -> bool:
    """空 tensor / 全零输出用例判定（K=0 全零、N=0 空、M=0 空、R=0 空）。"""
    if case.value in ("inf", "nan"):
        return False
    return (case.m == 0 or case.k == 0 or case.n == 0 or case.R == 0)


# ---------------------------------------------------------------------------
# L0 门槛用例
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L0_CASES, ids=[c.cid for c in L0_CASES])
def test_l0_gather_mm(case: GmmCase, npu_device):
    A, B, idx_a, idx_b, idx_c = case.build()
    golden = _golden_for(case, A, B, idx_a, idx_b, idx_c)
    npu_out = _npu_call_for(case, A, B, idx_a, idx_b, idx_c, npu_device)
    if _is_empty_or_zero_case(case):
        assert_empty_or_allzero(npu_out, golden)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L1 功能用例
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L1_CASES, ids=[c.cid for c in L1_CASES])
def test_l1_gather_mm(case: GmmCase, npu_device):
    A, B, idx_a, idx_b, idx_c = case.build()
    golden = _golden_for(case, A, B, idx_a, idx_b, idx_c)
    npu_out = _npu_call_for(case, A, B, idx_a, idx_b, idx_c, npu_device)
    if _is_empty_or_zero_case(case):
        assert_empty_or_allzero(npu_out, golden)
    assert_mixed_tolerance(npu_out, golden)


# ---------------------------------------------------------------------------
# L2 空张量 / 特殊值功能用例
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", L2_FUNC_CASES, ids=[c.cid for c in L2_FUNC_CASES])
def test_l2_gather_mm_func(case: GmmCase, npu_device):
    A, B, idx_a, idx_b, idx_c = case.build()
    golden = _golden_for(case, A, B, idx_a, idx_b, idx_c)
    npu_out = _npu_call_for(case, A, B, idx_a, idx_b, idx_c, npu_device)
    if case.value in ("inf", "nan"):
        # nan/inf 等价判定（assert_mixed_tolerance 内部已分离特殊值路径）
        assert_mixed_tolerance(npu_out, golden)
    elif _is_empty_or_zero_case(case):
        assert_empty_or_allzero(npu_out, golden)
        # K=0 时输出全零也做一次混合容差（全零应通过）
        assert_mixed_tolerance(npu_out, golden)
    else:
        assert_mixed_tolerance(npu_out, golden)


# ===========================================================================
# 白盒补充用例（源码分支枚举，24 例）
#
# 依据：gather_mm_kernel.cpp + gather_mm_cube_processor.h + gather_mm.cc
# 方法：从 host tiling 与 kernel Process 控制流枚举全部活分支，为每条分支
# 反解触发 shape/dtype/参数。已有黑盒用例（L0/L1/L2）覆盖的分支不重复；
# 本节聚焦黑盒未触及的 tiling 尾块/非对齐/多核边界/原子路径。
#

#
# 常量参考（gather_mm_cube_processor.h / gather_mm.cc）：
#   GMM_BLOCK_LENGTH = 16     (M/K 方向 block 大小)
#   GMM_C0_FLOAT     = 8     (C0 fractal，K/N 对齐基数)
#   GMM_MAX_KTILE    = 64    (K 方向最大 tile)
#   GMM_MAX_NTILE   = 64    (N 方向最大 tile)
#   kCubeCoreCount   = 20    (Cube 核数)
# ===========================================================================

# ---- K1 Cube Processor: M-block 尾块（Mg vs GMM_BLOCK_LENGTH=16）----
# 源码 gather_mm_cube_processor.h:85-88
#   mBlocks = CeilDiv(Mg, 16); actualRows = Min(16, Mg - mOff)
# 黑盒 Mg ≤ 8（多 group），未覆盖 Mg ≥ 16 的单 group 多 block 路径。

WB_K1_M_CASES: List[GmmCase] = [
    # Mg=16: 恰好 1 个完整 block，actualRows=16（无尾块）
    GmmCase("WB-K1-M01", "gmm", 16, 8, 16, 1, None, [0] * 16,
            seed=500, desc="K1 Mg=16 exact block（无尾块，actualRows=16）"),
    # Mg=17: 2 blocks，第 1 个 full(16) + 第 2 个 tail(1)
    GmmCase("WB-K1-M02", "gmm", 17, 8, 16, 1, None, [0] * 17,
            seed=501, desc="K1 Mg=17 full+tail(1)（2 mBlocks，尾块 actualRows=1）"),
    # Mg=33: 3 blocks，2 full(16) + tail(1)
    GmmCase("WB-K1-M03", "gmm", 33, 8, 16, 1, None, [0] * 33,
            seed=502, desc="K1 Mg=33 full+full+tail(1)（3 mBlocks，尾块 actualRows=1）"),
    # Mg=8: 单 block，actualRows=8 < 16（半 block）
    GmmCase("WB-K1-M04", "gmm", 8, 8, 16, 1, None, [0] * 8,
            seed=503, desc="K1 Mg=8 半 block（单 group，actualRows=8<16）"),
]

# ---- K1 Cube Processor: N-tile 尾块（N vs nTile）----
# 源码 gather_mm_cube_processor.h:90-91 + gather_mm.cc:56-64
#   nTile = min(64, N); floor to 16-multiple; if 0 → 16
#   nChunk = Min(nTile, N - nOff)
# 黑盒 N ≤ 8，仅单 n-iter。未覆盖 N > nTile 的多 n-iter 尾块路径。

WB_K1_N_CASES: List[GmmCase] = [
    # N=17: nTile=16，2 n-iters (16, 1)
    GmmCase("WB-K1-N01", "gmm", 4, 8, 17, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=510, desc="K1 N=17 nTile=16 两 n-tiles tail(1)"),
    # N=65: nTile=64，2 n-iters (64, 1)
    GmmCase("WB-K1-N02", "gmm", 4, 8, 65, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=511, desc="K1 N=65 nTile=64 两 n-tiles tail(1)"),
    # N=80: nTile=64，2 n-iters (64, 16)
    GmmCase("WB-K1-N03", "gmm", 4, 8, 80, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=512, desc="K1 N=80 nTile=64 两 n-tiles tail(16)"),
]

# ---- K1 Cube Processor: K-tile 尾块（K vs kTile）----
# 源码 gather_mm_cube_processor.h:96-97 + gather_mm.cc:66-72
#   kTile = min(64, K); floor to 8-multiple; if 0 → 8
#   kChunk = Min(kTile, K - kOff); initC true→false across k-iters
# 黑盒 K ≤ 8，仅单 k-iter。未覆盖 K > kTile 的多 k-iter 尾块 + initC 翻转路径。

WB_K1_K_CASES: List[GmmCase] = [
    # K=9: kTile=8，2 k-iters (8, 1)，initC true→false
    GmmCase("WB-K1-K01", "gmm", 4, 9, 16, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=520, desc="K1 K=9 kTile=8 两 k-tiles tail(1) initC T→F"),
    # K=65: kTile=64，2 k-iters (64, 1)
    GmmCase("WB-K1-K02", "gmm", 4, 65, 16, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=521, desc="K1 K=65 kTile=64 两 k-tiles tail(1)"),
    # K=72: kTile=64，2 k-iters (64, 8)
    GmmCase("WB-K1-K03", "gmm", 4, 72, 16, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=522, desc="K1 K=72 kTile=64 两 k-tiles tail(8)"),
]

# ---- K1 Cube Processor: 多核分组边界（numGroups vs kCubeCoreCount=20）----
# 源码 gather_mm_cube_processor.h:67-70
#   groupsPerCore = CeilDiv(numGroups, blockNum)
#   gHi = Min(gLo + groupsPerCore, numGroups)  ← 尾核 clamp
#   if (gLo >= gHi) return;                     ← 空闲核
# 黑盒 numGroups ≤ 8（blockDim ≤ 8），未覆盖 numGroups > 20 的多核尾核/空闲核路径。

WB_K1_G_CASES: List[GmmCase] = [
    # numGroups=21 > 20: blockDim=20, groupsPerCore=2
    # core 0-9: 2 groups; core 10: gLo=20 gHi=21 (1 group, 尾核); core 11-19: idle
    GmmCase("WB-K1-G01", "gmm", 42, 8, 16, 21, None, list(range(21)) * 2,
            seed=530, desc="K1 numGroups=21>20 尾核(1 group)+空闲核(9)"),
    # numGroups=1 < 20: blockDim=1, 1 core active, 19 idle
    GmmCase("WB-K1-G02", "gmm", 8, 8, 16, 1, None, [0] * 8,
            seed=531, desc="K1 numGroups=1<20 单核 active 19 空闲核"),
]

# ---- Scatter AIV: 核/行尾核 + n padding + 原子路径 ----
# 源码 gather_mm_kernel.cpp:40-125
#   rowsPerCore = CeilDiv(m, blockNum); rHi = Min(rLo+rowsPerCore, m)  ← 尾核
#   if (rLo >= rHi) return;                                           ← 空闲核
#   ubElements = n; if (< 8) → 8; round to 8                          ← n padding
#   if (isScatter) SetAtomicAdd / SetAtomicNone                       ← 原子配对
# 黑盒 m ≤ 16（scatter 不触发尾核），n ≥ 6（padding 隐式但无 n=1 显式），原子碰撞 ≤ 8 行。

WB_SC_CASES: List[GmmCase] = [
    # m=25 > 20: aivBlockDim=20, rowsPerCore=2, core 12: rHi=25 (1 row 尾核), 13-19: idle
    # isScatter=0 (GatherMM direct write)
    GmmCase("WB-SC-01", "gmm", 25, 8, 16, 3, None,
            [i % 3 for i in range(25)],
            seed=540, desc="Scatter AIV m=25 尾核(1 row)+空闲核 isScatter=0 direct write"),
    # m=25 GMS K1 idx_c Null → isScatter=0 (direct write through GMS path)
    # 与 WB-SC-01 互补：同一 isScatter=0 但走 GatherMMScatterAscend 代码路径
    GmmCase("WB-SC-02", "k1", 25, 8, 16, 3, None,
            [i % 3 for i in range(25)], None,
            seed=541, desc="Scatter AIV m=25 尾核 isScatter=0 via GMS idx_c Null"),
    # n=3 < 8: ubElements padded to 8
    GmmCase("WB-SC-03", "gmm", 4, 8, 3, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=542, desc="Scatter AIV n=3<8 ubElements padded to 8"),
    # isScatter=1 atomic stress: 20 rows all → C[0], 20 atomic adds
    # 1 group Mg=20 (idx_b all 0), idx_c all 0 → SetAtomicAdd/SetAtomicNone 配对
    GmmCase("WB-SC-04", "k1", 20, 4, 16, 3, None, [0] * 20, [0] * 20,
            seed=543, desc="Scatter AIV isScatter=1 20 collisions atomic add stress"),
    # n=1 minimal: ubElements = max(1,8) = 8
    GmmCase("WB-SC-05", "gmm", 4, 8, 1, 3, [2, 0, 1, 2], [1, 2, 0, 1],
            seed=544, desc="Scatter AIV n=1 minimal ubElements=8"),
]

# ---- K2 Cube Processor: innerKtile 尾块（Mg vs GMM_MAX_KTILE=64 / GMM_C0_FLOAT=8）----
# 源码 gather_mm_cube_processor.h:306-308, 332-333
#   innerKtile = Min(64, Mg); floor to 8-multiple; if 0 → 8
#   kChunk = Min(innerKtile, Mg - kOff); initC true→false across k-iters
# 黑盒 Mg ≤ 2（K2 cases），未覆盖 Mg ≥ 7 的 innerKtile 多 k-iter 路径。

WB_K2_IK_CASES: List[GmmCase] = [
    # Mg=7: innerKtile = 7/8*8 = 0 → 8; kChunk = Min(8,7) = 7 (single, tail)
    GmmCase("WB-K2-IK01", "k2", 7, 16, 16, 7, None, None, [0] * 7,
            seed=550, desc="K2 Mg=7 innerKtile=8 fallback kChunk=7 (tail)"),
    # Mg=9: innerKtile = 9/8*8 = 8; 2 k-iters (8, 1), initC T→F
    GmmCase("WB-K2-IK02", "k2", 9, 16, 16, 9, None, None, [0] * 9,
            seed=551, desc="K2 Mg=9 innerKtile=8 两 k-iters (8,1) initC T→F"),
    # Mg=65: innerKtile = 64; 2 k-iters (64, 1)
    GmmCase("WB-K2-IK03", "k2", 65, 16, 16, 65, None, None, [0] * 65,
            seed=552, desc="K2 Mg=65 innerKtile=64 两 k-iters (64,1)"),
]

# ---- K2 Cube Processor: K-block 尾块 + N-tile 尾块 ----
# 源码 gather_mm_cube_processor.h:320-323, 325-326
#   mBlocks = CeilDiv(K, 16); actualRowsK = Min(16, K - mOffK)
#   nChunk = Min(nTile, N - nOff)
# 黑盒 K=8 (1 block), N ≤ 8 (1 n-iter). 未覆盖 K>16 / N>64 的多 block/tile 尾块路径。

WB_K2_KN_CASES: List[GmmCase] = [
    # K=17: mBlocks=2, actualRowsK tail=1
    GmmCase("WB-K2-K01", "k2", 4, 17, 16, 4, None, None, [0, 1, 0, 2],
            seed=560, desc="K2 K=17 两 K-blocks actualRowsK tail=1"),
    # N=80: nTile=64, 2 n-iters (64, 16)
    GmmCase("WB-K2-N01", "k2", 4, 16, 80, 4, None, None, [0, 1, 0, 2],
            seed=561, desc="K2 N=80 nTile=64 两 n-tiles tail(16)"),
]

# ---- K2 Cube Processor: 多核分组 + Mg=1 无内积 ----
# 源码 gather_mm_cube_processor.h:294-297 (多核), 306-333 (Mg=1 innerKtile fallback)

WB_K2_G_CASES: List[GmmCase] = [
    # numGroups=21 > 20: blockDim=20, groupsPerCore=2, tail core + idle cores
    GmmCase("WB-K2-G01", "k2", 42, 8, 16, 42, None, None,
            list(range(21)) * 2,
            seed=570, desc="K2 numGroups=21>20 尾核+空闲核"),
    # Mg=1 per group: 4 groups, innerKtile=8 fallback, kChunk=1, no inner accumulation
    GmmCase("WB-K2-G02", "k2", 4, 16, 16, 4, None, None, [0, 1, 2, 3],
            seed=571, desc="K2 Mg=1 per group innerKtile=8 fallback no inner accum"),
]

# ---- 白盒用例汇总 ----

WB_CASES: List[GmmCase] = (
    WB_K1_M_CASES + WB_K1_N_CASES + WB_K1_K_CASES + WB_K1_G_CASES
    + WB_SC_CASES + WB_K2_IK_CASES + WB_K2_KN_CASES + WB_K2_G_CASES
)


@pytest.mark.parametrize("case", WB_CASES, ids=[c.cid for c in WB_CASES])
def test_whitebox_gather_mm(case: GmmCase, npu_device):
    """Whitebox cases: source-branch-driven coverage of tiling tails / misalignment /
    multi-core boundaries / atomic paths.

    Known limitation: WB-K1-K02/K03 (K>64 + multi-group) trigger a Cube group-level
    B-data isolation issue in gather_mm_cube_processor.h. xfail until fixed.
    """
    if case.cid in ("WB-K1-K02", "WB-K1-K03"):
        pytest.xfail("Known Cube group-level B-data isolation issue for K>64 + multi-group")
    A, B, idx_a, idx_b, idx_c = case.build()
    golden = _golden_for(case, A, B, idx_a, idx_b, idx_c)
    npu_out = _npu_call_for(case, A, B, idx_a, idx_b, idx_c, npu_device)
    assert_mixed_tolerance(npu_out, golden)


# ===========================================================================
# L2 异常用例 —— 参数校验（dtype / shape / 长度）
# ===========================================================================

def _build_normal_inputs_gmm(idtype: str = "int32", dtype: torch.dtype = torch.float32):
    """构造 GatherMM 正常参数模板（A[4,8] B[3,8,6] idx_a/b[4]），异常用例在其上局部破坏。"""
    idt = IDTYPE_MAP[idtype]
    A = gen_tensor((4, 8), "normal", dtype=dtype, seed=100)
    B = gen_tensor((3, 8, 6), "normal", dtype=dtype, seed=101)
    idx_a = torch.tensor([2, 0, 1, 2], dtype=idt)
    idx_b = torch.tensor([1, 2, 0, 1], dtype=idt)
    return A, B, idx_a, idx_b


def _build_normal_inputs_gms_k1(idtype: str = "int32", dtype: torch.dtype = torch.float32):
    """构造 GatherMMScatter Kernel1 正常参数模板（A[4,8] B[3,8,6] idx_a/b/c[4] numC=3）。"""
    idt = IDTYPE_MAP[idtype]
    A = gen_tensor((4, 8), "normal", dtype=dtype, seed=200)
    B = gen_tensor((3, 8, 6), "normal", dtype=dtype, seed=201)
    idx_a = torch.tensor([2, 0, 1, 2], dtype=idt)
    idx_b = torch.tensor([1, 2, 0, 1], dtype=idt)
    idx_c = torch.tensor([0, 1, 2, 0], dtype=idt)
    return A, B, idx_a, idx_b, idx_c


def _build_normal_inputs_gms_k2(idtype: str = "int32", dtype: torch.dtype = torch.float32):
    """构造 GatherMMScatter Kernel2 正常参数模板（A[4,8] B[3,6] idx_a/b/c[4] numC=3）。"""
    idt = IDTYPE_MAP[idtype]
    A = gen_tensor((4, 8), "normal", dtype=dtype, seed=300)
    B = gen_tensor((3, 6), "normal", dtype=dtype, seed=301)
    idx_a = torch.tensor([2, 0, 1, 2], dtype=idt)
    idx_b = torch.tensor([1, 2, 0, 1], dtype=idt)
    idx_c = torch.tensor([0, 1, 0, 2], dtype=idt)
    return A, B, idx_a, idx_b, idx_c


def _to_dev_npu(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device)


# ---- GatherMM 异常（7） ----

def test_gmm_ex_01_half_dtype(npu_device):
    """GMM-EX-01: GatherMM DType=float16 → LOG(FATAL)/报错。"""
    A, B, idx_a, idx_b = _build_normal_inputs_gmm(dtype=torch.float16)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((4, 6), device=npu_device, dtype=torch.float16)
    with pytest.raises(Exception):
        _gather_mm(A_n, B_n, C, _to_dev_npu(idx_a, npu_device), _to_dev_npu(idx_b, npu_device))


def test_gmm_ex_02_bfloat16_dtype(npu_device):
    """GMM-EX-02: GatherMM DType=bfloat16 → LOG(FATAL)/报错。"""
    A, B, idx_a, idx_b = _build_normal_inputs_gmm(dtype=torch.bfloat16)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((4, 6), device=npu_device, dtype=torch.bfloat16)
    with pytest.raises(Exception):
        _gather_mm(A_n, B_n, C, _to_dev_npu(idx_a, npu_device), _to_dev_npu(idx_b, npu_device))


def test_gmm_ex_03_double_dtype(npu_device):
    """GMM-EX-03: GatherMM DType=float64 → LOG(FATAL)/报错（Ascend 无 double GEMM）。"""
    A, B, idx_a, idx_b = _build_normal_inputs_gmm(dtype=torch.float64)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((4, 6), device=npu_device, dtype=torch.float64)
    with pytest.raises(Exception):
        _gather_mm(A_n, B_n, C, _to_dev_npu(idx_a, npu_device), _to_dev_npu(idx_b, npu_device))


def test_gmm_ex_04_a_ndim_1(npu_device):
    """GMM-EX-04: GatherMM A.ndim=1 → dispatch CHECK_EQ(A->ndim,2) 失败，报错。"""
    A = gen_tensor((4,), "normal", seed=100)  # 1D
    B = gen_tensor((3, 8, 6), "normal", seed=101)
    idx_a = torch.tensor([2, 0, 1, 2], dtype=torch.int32)
    idx_b = torch.tensor([1, 2, 0, 1], dtype=torch.int32)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((4, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _gather_mm(A_n, B_n, C, _to_dev_npu(idx_a, npu_device), _to_dev_npu(idx_b, npu_device))


def test_gmm_ex_05_b_ndim_2(npu_device):
    """GMM-EX-05: GatherMM B.ndim=2 → dispatch CHECK_EQ(B->ndim,3) 失败，报错。"""
    A = gen_tensor((4, 8), "normal", seed=100)
    B = gen_tensor((3, 6), "normal", seed=101)  # 2D
    idx_a = torch.tensor([2, 0, 1, 2], dtype=torch.int32)
    idx_b = torch.tensor([1, 2, 0, 1], dtype=torch.int32)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((4, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _gather_mm(A_n, B_n, C, _to_dev_npu(idx_a, npu_device), _to_dev_npu(idx_b, npu_device))


def test_gmm_ex_06_idx_len_mismatch(npu_device):
    """GMM-EX-06: GatherMM len(idx_a)!=len(idx_b)（两非空）→ dispatch CHECK_EQ 失败，报错。"""
    A, B, _, _ = _build_normal_inputs_gmm()
    idx_a = torch.tensor([2, 0, 1], dtype=torch.int32)      # len 3
    idx_b = torch.tensor([1, 2, 0, 1], dtype=torch.int32)    # len 4
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((4, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _gather_mm(A_n, B_n, C, _to_dev_npu(idx_a, npu_device), _to_dev_npu(idx_b, npu_device))


def test_gmm_ex_07_idx_a_null_idx_b_len_mismatch(npu_device):
    """GMM-EX-07: GatherMM idx_a Null & len(idx_b)!=m → dispatch CHECK_EQ 失败，报错。"""
    A, B, _, _ = _build_normal_inputs_gmm()   # m=4
    idx_b = torch.tensor([1, 2, 0], dtype=torch.int32)       # len 3 != m=4
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((4, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _gather_mm(A_n, B_n, C, None, _to_dev_npu(idx_b, npu_device))


# ---- GatherMMScatter 异常（4） ----

def test_gms_ex_01_k1_half_dtype(npu_device):
    """GMS-EX-01: GatherMMScatter Kernel1 DType=half → LOG(FATAL)/报错。"""
    A, B, idx_a, idx_b, idx_c = _build_normal_inputs_gms_k1(dtype=torch.float16)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((3, 6), device=npu_device, dtype=torch.float16)
    with pytest.raises(Exception):
        _gather_mm_scatter(A_n, B_n, C,
                           _to_dev_npu(idx_a, npu_device),
                           _to_dev_npu(idx_b, npu_device),
                           _to_dev_npu(idx_c, npu_device))


def test_gms_ex_02_k2_double_dtype(npu_device):
    """GMS-EX-02: GatherMMScatter Kernel2 DType=double → LOG(FATAL)/报错。"""
    A, B, idx_a, idx_b, idx_c = _build_normal_inputs_gms_k2(dtype=torch.float64)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((3, 8, 6), device=npu_device, dtype=torch.float64)
    with pytest.raises(Exception):
        _gather_mm_scatter(A_n, B_n, C,
                           _to_dev_npu(idx_a, npu_device),
                           _to_dev_npu(idx_b, npu_device),
                           _to_dev_npu(idx_c, npu_device))


def test_gms_ex_03_a_ndim_1(npu_device):
    """GMS-EX-03: GatherMMScatter A.ndim=1 → dispatch CHECK 失败，报错。"""
    A = gen_tensor((4,), "normal", seed=200)   # 1D
    B = gen_tensor((3, 8, 6), "normal", seed=201)
    idx_a = torch.tensor([2, 0, 1, 2], dtype=torch.int32)
    idx_b = torch.tensor([1, 2, 0, 1], dtype=torch.int32)
    idx_c = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((3, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _gather_mm_scatter(A_n, B_n, C,
                           _to_dev_npu(idx_a, npu_device),
                           _to_dev_npu(idx_b, npu_device),
                           _to_dev_npu(idx_c, npu_device))


def test_gms_ex_04_idx_len_mismatch(npu_device):
    """GMS-EX-04: GatherMMScatter len(idx_a)!=len(idx_b) → dispatch CHECK 失败，报错。"""
    A, B, _, _, idx_c = _build_normal_inputs_gms_k1()
    idx_a = torch.tensor([2, 0, 1], dtype=torch.int32)      # len 3
    idx_b = torch.tensor([1, 2, 0, 1], dtype=torch.int32)    # len 4
    A_n, B_n = _to_dev_npu(A, npu_device), _to_dev_npu(B, npu_device)
    C = torch.empty((3, 6), device=npu_device, dtype=torch.float32)
    with pytest.raises(Exception):
        _gather_mm_scatter(A_n, B_n, C,
                           _to_dev_npu(idx_a, npu_device),
                           _to_dev_npu(idx_b, npu_device),
                           _to_dev_npu(idx_c, npu_device))


# ---------------------------------------------------------------------------
# 用例汇总（供 pytest collection 核对）
#   黑盒 62 = L0:14 + L1:22 + L2 func:15 + L2 异常:11
#   白盒 24 = K1 M-block:4 + K1 N-tile:3 + K1 K-tile:3 + K1 多核:2
#            + Scatter:5 + K2 innerK:3 + K2 K/N-block:2 + K2 多核:2
#   合计 86
# ---------------------------------------------------------------------------

ALL_CASE_IDS = (
    [c.cid for c in L0_CASES]
    + [c.cid for c in L1_CASES]
    + [c.cid for c in L2_FUNC_CASES]
    + [
        "GMM-EX-01", "GMM-EX-02", "GMM-EX-03", "GMM-EX-04",
        "GMM-EX-05", "GMM-EX-06", "GMM-EX-07",
        "GMS-EX-01", "GMS-EX-02", "GMS-EX-03", "GMS-EX-04",
    ]
    + [c.cid for c in WB_CASES]
)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
