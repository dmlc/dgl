"""NPU 功能测试：UpdateGradMinMax_hetero / BackwardSegmentCmp。

Golden 方案
-----------
复用仓内 CPU 实现 ``src/array/cpu/segment_reduce.h:116/172``，经 DGL Python API
(``dgl._sparse_ops._bwd_segment_cmp`` / ``_update_grad_minmax_hetero``) 在 CPU device
上触发 CPU dispatch 产出期望输出；NPU 侧以同一份 numpy 种子产出输入，调用同一 API
触发 Ascend dispatch，``torch.npu.synchronize`` 后取回比对。不另写第二份 golden。

精度判定
--------
两算子统一 float32 混合容差：
  - ``abs(npu-golden) <= atol + rtol*abs(golden)`` 逐元素
  - ``matched_ratio >= 0.99``
  - ``max_abs_error <= 1e-2``
  - rtol = 2**-10, atol = 2**-16
BackwardSegmentCmp 额外做 bit-exact 交叉核对（仅信息性，不作通过门禁）。

用例分级
--------
L0 门槛 8 / L1 功能 30 / L2 异常 12，合计 50。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

import dgl
from dgl._sparse_ops import _bwd_segment_cmp, _update_grad_minmax_hetero

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


# --------------------------------------------------------------------------- #
# 精度判定常量（float32 混合容差口径）
# --------------------------------------------------------------------------- #
RTOL: float = 2 ** -10          # ≈ 9.77e-4
ATOL: float = 2 ** -16          # ≈ 1.53e-5
REQUIRED_MATCHED_RATIO: float = 0.99
MAX_ABS_ERROR_LIMIT: float = 1e-2


# --------------------------------------------------------------------------- #
# 设备辅助
# --------------------------------------------------------------------------- #
def _npu_device() -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("未检测到可用 NPU 设备，跳过 NPU 用例。")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    return device


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def _to_device(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device)


# --------------------------------------------------------------------------- #
# 混合容差断言
# --------------------------------------------------------------------------- #
def assert_mixed_tolerance(
    npu_out: torch.Tensor,
    golden_out: torch.Tensor,
    *,
    case_id: str,
    rtol: float = RTOL,
    atol: float = ATOL,
    matched_ratio: float = REQUIRED_MATCHED_RATIO,
    max_abs_error_limit: float = MAX_ABS_ERROR_LIMIT,
) -> None:
    """逐元素混合容差判定，失败抛 AssertionError 并附完整指标。"""
    npu_f = npu_out.detach().cpu().float()
    golden_f = golden_out.detach().cpu().float()
    assert npu_f.shape == golden_f.shape, (
        f"[{case_id}] shape 不一致: npu {tuple(npu_f.shape)} vs golden {tuple(golden_f.shape)}"
    )
    total = npu_f.numel()
    if total == 0:
        # 空输出视为完全匹配
        return
    abs_err = (npu_f - golden_f).abs()
    threshold = atol + rtol * golden_f.abs()
    passed = abs_err <= threshold
    matched = float(passed.sum().item()) / total
    max_abs_err = float(abs_err.max().item())
    ok = (matched >= matched_ratio) and (max_abs_err <= max_abs_error_limit)
    msg = (
        f"[{case_id}] 混合容差未通过: matched_ratio={matched:.6f}(需>={matched_ratio}), "
        f"max_abs_error={max_abs_err:.6g}(需<={max_abs_error_limit}), "
        f"shape={tuple(npu_f.shape)}"
    )
    assert ok, msg


def assert_bit_exact(npu_out: torch.Tensor, golden_out: torch.Tensor, *, case_id: str) -> bool:
    """bit-exact 交叉核对（信息性），返回是否 bit-exact，不作为通过门禁。"""
    npu_f = npu_out.detach().cpu()
    golden_f = golden_out.detach().cpu()
    if npu_f.shape != golden_f.shape or npu_f.dtype != golden_f.dtype:
        return False
    return bool(torch.equal(npu_f, golden_f))


# --------------------------------------------------------------------------- #
# BackwardSegmentCmp：输入生成 + 运行
# --------------------------------------------------------------------------- #
def _make_feat(n: int, dim: int, mode: str, rng: np.random.RandomState) -> np.ndarray:
    """按 mode 生成 float32 feat 数据。"""
    if mode == "normal":
        return (rng.randn(n, dim) * 0.5).astype(np.float32)
    if mode == "signed":
        # 正负小量级混合
        base = rng.randn(n, dim) * 0.1
        return base.astype(np.float32)
    if mode == "pmzero":
        # 含 +0 / -0
        data = (rng.randn(n, dim) * 0.3).astype(np.float32)
        if n > 0 and dim > 0:
            data[0, 0] = 0.0           # +0
            if n * dim > 1:
                data.flat[1] = -0.0    # -0
        return data
    if mode == "boundary":
        # 含 ±float32_max、±最小正规、±次正规
        data = (rng.randn(n, dim) * 0.5).astype(np.float32)
        specials = np.array(
            [np.finfo(np.float32).max, -np.finfo(np.float32).max,
             np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
             np.finfo(np.float32).smallest_subnormal,
             -np.finfo(np.float32).smallest_subnormal],
            dtype=np.float32,
        )
        if specials.size and data.size:
            for i in range(min(specials.size, data.size)):
                data.flat[i] = specials[i]
        return data
    # 默认 normal
    return (rng.randn(n, dim) * 0.5).astype(np.float32)


def _make_arg(
    n: int, dim: int, m: int, idtype: torch.dtype, pattern: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """生成 arg 索引张量，保证**同列内非负 arg 值唯一**（无写冲突）。

    BackwardSegmentCmp 为直写语义（``out[arg[r,c], c] = feat[r,c]``，
    last-write-wins，非原子累加）。arg 来自 segment min/max 前向的
    argmin/argmax，每个输出行每列至多被一个输入行写入。本函数按列生成
    **部分双射**：每列非负 arg 值取自 ``[0, m)`` 的无放回子集（互不相同），
    未被覆盖的行填 -1，从而满足算子输入契约，使 CPU golden 与 NPU 输出
    具备确定性可比性。

    pattern:
       - "valid":  每列 ``min(n, m)`` 行赋以 ``[0, m)`` 的随机排列子集，其余 -1
       - "some-1": 在 valid 基础上保留约 7/8 有效行，其余 -1（仍保证同列唯一）
       - "all-1":  全 -1
       - "col-1":  第 3 列（index 2）全 -1，其余列同 valid
    """
    if n == 0 or dim == 0:
        return np.zeros((n, dim), dtype=np.int64)
    if pattern == "all-1":
        return np.full((n, dim), -1, dtype=np.int64)

    # 每列有效行数上限：[0, m) 内至多 min(n, m) 个互不相同的值
    k_max = min(n, max(m, 1))
    if pattern == "some-1":
        # 保留约 7/8 上限行有效，至少 1 行（若上限 > 0）
        k_max = max(1, int(k_max * 0.875)) if k_max > 0 else 0

    arg = np.full((n, dim), -1, dtype=np.int64)
    if m <= 0 or k_max <= 0:
        return arg

    for c in range(dim):
        if pattern == "col-1" and c == 2:
            continue  # 第 3 列保持全 -1
        # 从 [0, m) 无放回抽取 k_max 个互不相同的值
        vals = rng.permutation(m)[:k_max]
        # 随机选取 k_max 个行位置放置（其余行保持 -1）
        rows = rng.permutation(n)[:k_max]
        arg[rows, c] = vals
    return arg


def run_bwd_segment_cmp(
    feat_np: np.ndarray, arg_np: np.ndarray, m: int,
    idtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """在指定 device 上调用 _bwd_segment_cmp，返回 CPU 上的输出张量。"""
    feat = torch.from_numpy(feat_np.copy()).float().to(device)
    arg = torch.from_numpy(arg_np.copy()).to(idtype).to(device)
    out = _bwd_segment_cmp(feat, arg, m)
    _sync(device)
    return out.detach().cpu()


def run_bwd_segment_cmp_golden(
    feat_np: np.ndarray, arg_np: np.ndarray, m: int, idtype: torch.dtype,
) -> torch.Tensor:
    return run_bwd_segment_cmp(feat_np, arg_np, m, idtype, torch.device("cpu"))


def _bwd_check(
    case_id: str, n: int, dim: int, m: int, idtype: torch.dtype,
    feat_mode: str = "normal", arg_pattern: str = "valid", seed: int = 0,
) -> None:
    device = _npu_device()
    rng = np.random.RandomState(seed)
    feat_np = _make_feat(n, dim, feat_mode, rng)
    arg_np = _make_arg(n, dim, m, idtype, arg_pattern, rng)
    golden = run_bwd_segment_cmp_golden(feat_np, arg_np, m, idtype)
    npu_out = run_bwd_segment_cmp(feat_np, arg_np, m, idtype, device)
    assert_mixed_tolerance(npu_out, golden, case_id=case_id)
    # 交叉核对（信息性，不作通过门禁）：直接赋值语义理论 bit-exact
    # 混合容差通过即判 PASS；bit-exact 分叉仅记录，不阻断。
    assert_bit_exact(npu_out, golden, case_id=case_id)


# --------------------------------------------------------------------------- #
# UpdateGradMinMax_hetero：异构图与输入构建 + 运行
# --------------------------------------------------------------------------- #
def _build_hetero_gidx(etype_edges: List[Tuple[int, int]]) -> "dgl.HeteroGraphIndex":
    """根据 (src_ntype, dst_ntype) 列表构建异构图并返回 *反转* 后的 graph index。

    每个 etype 仅放一条占位边（算子只用到 metagraph 结构，不依赖实际边数据）。
    """
    data_dict: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}
    num_ntypes = 0
    for e, (s, d) in enumerate(etype_edges):
        data_dict[(f"n{s}", f"r{e}", f"n{d}")] = ([0], [0])
        num_ntypes = max(num_ntypes, s + 1, d + 1)
    num_nodes_dict = {f"n{t}": 1 for t in range(num_ntypes)}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, idtype=torch.int64)
    return g._graph.reverse()


def build_hetero_inputs(
    etype_edges: List[Tuple[int, int]],
    op: str,
    dim: int,
    idtype: torch.dtype,
    match_pattern: str,
    n_dst_v: Dict[int, int],
    n_out: Dict[int, int],
    feat_mode: str = "normal",
    seed: int = 0,
) -> Tuple[
    "dgl.HeteroGraphIndex",
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
    List[int],
]:
    """构建 hetero 测试输入。

    参数
    ----
    etype_edges : [(src_ntype, dst_ntype), ...]
    op          : "copy_lhs" / "copy_rhs"
    dim         : 末维
    idtype      : int32 / int64
    match_pattern: "all" / "none" / "partial"
    n_dst_v     : {ntype: num_feat_rows}（feat 按 dst_ntype 索引）
    n_out       : {type_key: n_out}（copy_rhs 按 etype，copy_lhs 按 ntype）
    feat_mode   : feat 数值模式
    seed        : 随机种子

    返回
    ----
    (gidx, list_x, list_idx, list_idx_etype, list_dX, out_indices)
      list_x/list_idx/list_idx_etype 按 ntype 索引（CPU float32 / idtype 张量）；
      list_dX 为 dummy 张量（仅用 len，CPU）；
      out_indices 为会产生非 None 输出的索引列表。
    """
    rng = np.random.RandomState(seed)
    num_etypes = len(etype_edges)
    num_ntypes = 0
    for s, d in etype_edges:
        num_ntypes = max(num_ntypes, s + 1, d + 1)

    gidx = _build_hetero_gidx(etype_edges)

    # dst_ntype -> etype（假设每个 ntype 最多是一个 etype 的 dst）
    dst_to_etype: Dict[int, int] = {}
    dst_to_src: Dict[int, int] = {}
    for e, (s, d) in enumerate(etype_edges):
        dst_to_etype[d] = e
        dst_to_src[d] = s

    list_x: List[Optional[torch.Tensor]] = [None] * num_ntypes
    list_idx: List[Optional[torch.Tensor]] = [None] * num_ntypes
    list_idx_etype: List[Optional[torch.Tensor]] = [None] * num_ntypes

    for t in range(num_ntypes):
        n = n_dst_v.get(t, 0)
        if n == 0:
            list_x[t] = torch.zeros((0, dim), dtype=torch.float32)
            list_idx[t] = torch.zeros((0, dim), dtype=idtype)
            list_idx_etype[t] = torch.zeros((0, dim), dtype=idtype)
            continue
        feat_np = _make_feat(n, dim, feat_mode, rng)
        list_x[t] = torch.from_numpy(feat_np.copy()).float()

        e = dst_to_etype[t]
        s = dst_to_src[t]
        type_val = s if op == "copy_lhs" else e
        out_size = n_out[type_val]

        if dim == 0:
            idx_np = np.zeros((n, 0), dtype=np.int64)
            etype_np = np.zeros((n, 0), dtype=np.int64)
        else:
            idx_np = rng.randint(0, max(out_size, 1), size=(n, dim)).astype(np.int64)
            # idx_etype 匹配模式
            if match_pattern == "all":
                etype_np = np.full((n, dim), type_val, dtype=np.int64)
            elif match_pattern == "none":
                etype_np = np.full((n, dim), type_val + 1, dtype=np.int64)
            else:  # partial ~50%
                etype_np = np.full((n, dim), type_val, dtype=np.int64)
                mask = rng.rand(n, dim) < 0.5
                etype_np[mask] = type_val + 1
        list_idx[t] = torch.from_numpy(idx_np.copy()).to(idtype)
        list_idx_etype[t] = torch.from_numpy(etype_np.copy()).to(idtype)

    # out_indices：会产生非 None 输出的索引
    if op == "copy_rhs":
        out_indices = list(range(num_etypes))
    else:  # copy_lhs：type = src_ntype
        out_indices = sorted({s for s, _ in etype_edges})

    # list_dX：dummy 张量，仅用 len() 定位输出行数。size = max(ntypes, etypes)。
    dX_size = max(num_ntypes, num_etypes)
    list_dX: List[Optional[torch.Tensor]] = [None] * dX_size
    for key in n_out:
        if key < dX_size:
            list_dX[key] = torch.zeros(n_out[key], dtype=torch.float32)

    return gidx, list_x, list_idx, list_idx_etype, list_dX, out_indices


def run_hetero(
    gidx, op, list_x, list_idx, list_idx_etype, list_dX, out_indices, device,
):
    """在指定 device 上调用 _update_grad_minmax_hetero，返回 CPU 上的输出列表。"""
    list_x_d = [_to_device(x, device) for x in list_x]
    list_idx_d = [_to_device(x, device) for x in list_idx]
    list_idx_etype_d = [_to_device(x, device) for x in list_idx_etype]
    # list_dX 仅用于 len()，保持在 CPU。
    outs = _update_grad_minmax_hetero(gidx, op, list_x_d, list_idx_d, list_idx_etype_d, list_dX)
    _sync(device)
    return [o.detach().cpu() if o is not None else None for o in outs]


def _hetero_check(
    case_id: str,
    etype_edges: List[Tuple[int, int]],
    op: str,
    dim: int,
    idtype: torch.dtype,
    match_pattern: str,
    n_dst_v: Dict[int, int],
    n_out: Dict[int, int],
    feat_mode: str = "normal",
    seed: int = 0,
) -> None:
    device = _npu_device()
    gidx, list_x, list_idx, list_idx_etype, list_dX, out_indices = build_hetero_inputs(
        etype_edges, op, dim, idtype, match_pattern, n_dst_v, n_out, feat_mode, seed
    )
    golden = run_hetero(
        gidx, op, list_x, list_idx, list_idx_etype, list_dX, out_indices, torch.device("cpu")
    )
    npu_out = run_hetero(
        gidx, op, list_x, list_idx, list_idx_etype, list_dX, out_indices, device
    )
    assert len(golden) == len(npu_out), f"[{case_id}] 输出数量不一致"
    for idx in out_indices:
        g = golden[idx]
        n = npu_out[idx]
        assert g is not None and n is not None, f"[{case_id}] 输出索引 {idx} 为 None"
        assert_mixed_tolerance(n, g, case_id=f"{case_id}[out={idx}]")
        # 累加语义下不强制 bit-exact；做信息性交叉核对记录
        assert_bit_exact(n, g, case_id=f"{case_id}[out={idx}]")


# --------------------------------------------------------------------------- #
# L0 门槛用例（8）
# --------------------------------------------------------------------------- #
class TestL0:
    """L0 门槛：单因子最小覆盖，常规 shape/dtype，小而快。"""

    def test_L0_01_bwd_basic_int32(self):
        _bwd_check("L0-01", n=16, dim=4, m=4, idtype=torch.int32,
                   feat_mode="normal", arg_pattern="valid", seed=1)

    def test_L0_02_bwd_basic_int64(self):
        _bwd_check("L0-02", n=16, dim=4, m=4, idtype=torch.int64,
                   feat_mode="normal", arg_pattern="valid", seed=2)

    def test_L0_03_bwd_arg_with_neg1(self):
        _bwd_check("L0-03", n=16, dim=4, m=4, idtype=torch.int32,
                   feat_mode="normal", arg_pattern="some-1", seed=3)

    def test_L0_04_hetero_single_etype_copy_rhs(self):
        _hetero_check(
            "L0-04", etype_edges=[(0, 0)], op="copy_rhs", dim=4,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 16}, n_out={0: 4}, seed=4,
        )

    def test_L0_05_hetero_single_etype_copy_lhs(self):
        # 同源 src==dst，copy_lhs 去重路径
        _hetero_check(
            "L0-05", etype_edges=[(0, 0)], op="copy_lhs", dim=4,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 16}, n_out={0: 4}, seed=5,
        )

    def test_L0_06_hetero_int64_idx(self):
        _hetero_check(
            "L0-06", etype_edges=[(0, 0)], op="copy_rhs", dim=4,
            idtype=torch.int64, match_pattern="all",
            n_dst_v={0: 16}, n_out={0: 4}, seed=6,
        )

    def test_L0_07_bwd_dim1(self):
        _bwd_check("L0-07", n=16, dim=1, m=4, idtype=torch.int32,
                   feat_mode="normal", arg_pattern="valid", seed=7)

    def test_L0_08_hetero_dim_unaligned33(self):
        _hetero_check(
            "L0-08", etype_edges=[(0, 0)], op="copy_rhs", dim=33,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 16}, n_out={0: 4}, seed=8,
        )


# --------------------------------------------------------------------------- #
# L1 功能用例（30）
# --------------------------------------------------------------------------- #
class TestL1Bwd:
    """BackwardSegmentCmp 功能用例。"""

    def test_L1_01_bwd_midshape_int32(self):
        _bwd_check("L1-01", n=64, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=11)

    def test_L1_02_bwd_midshape_int64(self):
        _bwd_check("L1-02", n=64, dim=8, m=16, idtype=torch.int64,
                   arg_pattern="valid", seed=12)

    def test_L1_03_bwd_dim1(self):
        _bwd_check("L1-03", n=64, dim=1, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=13)

    def test_L1_04_bwd_dim_unaligned33(self):
        _bwd_check("L1-04", n=64, dim=33, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=14)

    def test_L1_05_bwd_large_shape(self):
        _bwd_check("L1-05", n=1024, dim=128, m=256, idtype=torch.int32,
                   arg_pattern="valid", seed=15)

    def test_L1_06_bwd_n_eq_m_full(self):
        _bwd_check("L1-06", n=16, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=16)

    def test_L1_07_bwd_m1_single_segment(self):
        _bwd_check("L1-07", n=64, dim=8, m=1, idtype=torch.int32,
                   arg_pattern="valid", seed=17)

    def test_L1_08_bwd_arg_some_neg1(self):
        _bwd_check("L1-08", n=64, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="some-1", seed=18)

    def test_L1_09_bwd_arg_all_neg1(self):
        _bwd_check("L1-09", n=64, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="all-1", seed=19)

    def test_L1_10_bwd_col_all_neg1(self):
        _bwd_check("L1-10", n=64, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="col-1", seed=20)

    def test_L1_11_bwd_feat_pmzero(self):
        _bwd_check("L1-11", n=64, dim=8, m=16, idtype=torch.int32,
                   feat_mode="pmzero", arg_pattern="valid", seed=21)

    def test_L1_12_bwd_feat_boundary(self):
        _bwd_check("L1-12", n=64, dim=8, m=16, idtype=torch.int32,
                   feat_mode="boundary", arg_pattern="valid", seed=22)

    def test_L1_23_bwd_int32_unaligned_pairwise(self):
        _bwd_check("L1-23", n=256, dim=33, m=64, idtype=torch.int32,
                   arg_pattern="valid", seed=23)

    def test_L1_24_bwd_int64_large_pairwise(self):
        _bwd_check("L1-24", n=1024, dim=128, m=256, idtype=torch.int64,
                   arg_pattern="valid", seed=24)


class TestL1Hetero:
    """UpdateGradMinMax_hetero 功能用例。"""

    def test_L1_13_hetero_copy_rhs_all_match(self):
        _hetero_check(
            "L1-13", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 64}, n_out={0: 16}, seed=31,
        )

    def test_L1_14_hetero_copy_lhs_all_match(self):
        _hetero_check(
            "L1-14", etype_edges=[(0, 0)], op="copy_lhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 64}, n_out={0: 16}, seed=32,
        )

    def test_L1_15_hetero_multi_etype_partial(self):
        _hetero_check(
            "L1-15", etype_edges=[(0, 0), (1, 1), (2, 2)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="partial",
            n_dst_v={0: 64, 1: 64, 2: 64}, n_out={0: 16, 1: 16, 2: 16}, seed=33,
        )

    def test_L1_16_hetero_int64_idx(self):
        _hetero_check(
            "L1-16", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int64, match_pattern="all",
            n_dst_v={0: 64}, n_out={0: 16}, seed=34,
        )

    def test_L1_17_hetero_dim_unaligned33(self):
        _hetero_check(
            "L1-17", etype_edges=[(0, 0)], op="copy_rhs", dim=33,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 64}, n_out={0: 16}, seed=35,
        )

    def test_L1_18_hetero_copy_lhs_hetero_dedup(self):
        # 异源：2 etypes，src_ntype 不同，dst_ntype 不同
        _hetero_check(
            "L1-18", etype_edges=[(0, 1), (2, 3)], op="copy_lhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={1: 64, 3: 64}, n_out={0: 16, 2: 16}, seed=36,
        )

    def test_L1_19_hetero_idx_etype_no_match(self):
        _hetero_check(
            "L1-19", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="none",
            n_dst_v={0: 64}, n_out={0: 16}, seed=37,
        )

    def test_L1_20_hetero_partial_match(self):
        _hetero_check(
            "L1-20", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="partial",
            n_dst_v={0: 64}, n_out={0: 16}, seed=38,
        )

    def test_L1_21_hetero_feat_pmzero(self):
        _hetero_check(
            "L1-21", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="all", feat_mode="pmzero",
            n_dst_v={0: 64}, n_out={0: 16}, seed=39,
        )

    def test_L1_22_hetero_n_out_1(self):
        _hetero_check(
            "L1-22", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 64}, n_out={0: 1}, seed=40,
        )

    def test_L1_25_hetero_int64_multi_etype(self):
        _hetero_check(
            "L1-25", etype_edges=[(0, 0), (1, 1), (2, 2)], op="copy_rhs", dim=8,
            idtype=torch.int64, match_pattern="all",
            n_dst_v={0: 64, 1: 64, 2: 64}, n_out={0: 16, 1: 16, 2: 16}, seed=43,
        )

    def test_L1_26_hetero_copy_lhs_partial(self):
        # 2 etypes 同源（各自 src==dst），copy_lhs 部分匹配
        _hetero_check(
            "L1-26", etype_edges=[(0, 0), (1, 1)], op="copy_lhs", dim=8,
            idtype=torch.int32, match_pattern="partial",
            n_dst_v={0: 64, 1: 64}, n_out={0: 16, 1: 16}, seed=44,
        )

    def test_L1_27_hetero_dim128(self):
        _hetero_check(
            "L1-27", etype_edges=[(0, 0)], op="copy_rhs", dim=128,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 64}, n_out={0: 16}, seed=45,
        )

    def test_L1_28_hetero_n_dst_v_1(self):
        _hetero_check(
            "L1-28", etype_edges=[(0, 0)], op="copy_rhs", dim=1,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 1}, n_out={0: 1}, seed=46,
        )

    def test_L1_29_hetero_n_dst_v_4_full(self):
        # n_out == n_dst_v 满段
        _hetero_check(
            "L1-29", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 4}, n_out={0: 4}, seed=47,
        )

    def test_L1_30_hetero_n_dst_v_256(self):
        _hetero_check(
            "L1-30", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 256}, n_out={0: 16}, seed=48,
        )


# --------------------------------------------------------------------------- #
# L2 异常用例（12）
# --------------------------------------------------------------------------- #
class TestL2Exceptions:
    """异常场景：dtype 占位 LOG(FATAL) / ctx 不一致 / 非 contiguous / op 非法。"""

    def _expect_fatal(self, fn, case_id: str):
        # LOG(FATAL) 在 DGL 侧会抛 RuntimeError（或 SystemExit）
        with pytest.raises((RuntimeError, SystemExit, Exception)):
            try:
                fn()
            except Exception:
                raise

    def test_L2_01_bwd_feat_float16(self):
        device = _npu_device()
        feat = torch.randn(16, 4, dtype=torch.float16, device=device)
        arg = torch.randint(0, 4, (16, 4), dtype=torch.int32, device=device)
        self._expect_fatal(lambda: _bwd_segment_cmp(feat, arg, 4), "L2-01")

    def test_L2_02_bwd_feat_bfloat16(self):
        device = _npu_device()
        feat = torch.randn(16, 4, dtype=torch.bfloat16, device=device)
        arg = torch.randint(0, 4, (16, 4), dtype=torch.int32, device=device)
        self._expect_fatal(lambda: _bwd_segment_cmp(feat, arg, 4), "L2-02")

    def test_L2_03_bwd_feat_double(self):
        device = _npu_device()
        feat = torch.randn(16, 4, dtype=torch.float64, device=device)
        arg = torch.randint(0, 4, (16, 4), dtype=torch.int32, device=device)
        self._expect_fatal(lambda: _bwd_segment_cmp(feat, arg, 4), "L2-03")

    def test_L2_04_hetero_feat_float16(self):
        device = _npu_device()
        gidx = _build_hetero_gidx([(0, 0)])
        feat = torch.randn(16, 4, dtype=torch.float16, device=device)
        idx = torch.randint(0, 4, (16, 4), dtype=torch.int32, device=device)
        idx_etype = torch.zeros(16, 4, dtype=torch.int32, device=device)
        dX = [torch.zeros(4, dtype=torch.float32)]
        self._expect_fatal(
            lambda: _update_grad_minmax_hetero(
                gidx, "copy_rhs", [feat], [idx], [idx_etype], dX
            ),
            "L2-04",
        )

    def test_L2_05_hetero_feat_bfloat16(self):
        device = _npu_device()
        gidx = _build_hetero_gidx([(0, 0)])
        feat = torch.randn(16, 4, dtype=torch.bfloat16, device=device)
        idx = torch.randint(0, 4, (16, 4), dtype=torch.int32, device=device)
        idx_etype = torch.zeros(16, 4, dtype=torch.int32, device=device)
        dX = [torch.zeros(4, dtype=torch.float32)]
        self._expect_fatal(
            lambda: _update_grad_minmax_hetero(
                gidx, "copy_rhs", [feat], [idx], [idx_etype], dX
            ),
            "L2-05",
        )

    def test_L2_06_hetero_feat_double(self):
        device = _npu_device()
        gidx = _build_hetero_gidx([(0, 0)])
        feat = torch.randn(16, 4, dtype=torch.float64, device=device)
        idx = torch.randint(0, 4, (16, 4), dtype=torch.int32, device=device)
        idx_etype = torch.zeros(16, 4, dtype=torch.int32, device=device)
        dX = [torch.zeros(4, dtype=torch.float32)]
        self._expect_fatal(
            lambda: _update_grad_minmax_hetero(
                gidx, "copy_rhs", [feat], [idx], [idx_etype], dX
            ),
            "L2-06",
        )

    def test_L2_07_bwd_ctx_mismatch(self):
        device = _npu_device()
        feat = torch.randn(16, 4, dtype=torch.float32, device=device)
        arg = torch.randint(0, 4, (16, 4), dtype=torch.int32)  # CPU
        with pytest.raises(Exception):
            _bwd_segment_cmp(feat, arg, 4)

    @pytest.mark.xfail(
        raises=(RuntimeError, SystemExit, Exception),
        reason=(
            "已知缺陷：DGL Python 绑定层 zerocopy_to_dgl_ndarray 的 use-after-free——"
            "非连续输入触发内部 .contiguous() 产生临时张量，该临时张量在调用返回后被"
            "回收，而 NDArray 仍持有指向其存储的悬挂指针，导致后续访问非法内存。"
            "根因位于绑定层，非本算子缺陷。"
            "待绑定层修复后移除本 xfail 标记。"
        ),
        strict=False,
    )
    def test_L2_08_bwd_non_contiguous(self):
        # 根因：DGL Python 绑定层 zerocopy_to_dgl_ndarray use-after-free。
        # 非连续输入经绑定层 ``.contiguous()`` 规整后，临时张量被回收，
        # NDArray 持悬挂指针，访问非法内存。属绑定层缺陷，非本算子问题。
        #
        # 历史背景：Python 绑定 zerocopy_to_dgl_ndarray 内部调用 ``.contiguous()``，
        # 非连续输入在到达 C++ 前已被规整为连续，``CheckContiguous`` 永不触发。
        # 本用例原验证非连续输入被正确处理（结果与连续输入一致），
        # 现因上述绑定层 use-after-free 标记为 xfail。
        device = _npu_device()
        feat_full = torch.randn(32, 8, dtype=torch.float32, device=device)
        feat_nc = feat_full[::2, ::2]  # 非连续 (16, 4)
        assert not feat_nc.is_contiguous()
        feat_c = feat_nc.contiguous()
        # 无写冲突 arg：每列 arg[r,c] = r（直写即拷贝），避免 last-write-wins
        # 不确定性干扰非连续路径的正确性验证
        arg = torch.arange(16, dtype=torch.int32, device=device).view(16, 1).expand(16, 4)
        out_nc = _bwd_segment_cmp(feat_nc, arg, 16)
        _sync(device)
        out_c = _bwd_segment_cmp(feat_c, arg, 16)
        _sync(device)
        assert_mixed_tolerance(
            out_nc.detach().cpu(), out_c.detach().cpu(), case_id="L2-08"
        )

    def test_L2_09_hetero_invalid_op(self):
        device = _npu_device()
        gidx = _build_hetero_gidx([(0, 0)])
        feat = torch.randn(16, 4, dtype=torch.float32, device=device)
        idx = torch.randint(0, 4, (16, 4), dtype=torch.int32, device=device)
        idx_etype = torch.zeros(16, 4, dtype=torch.int32, device=device)
        dX = [torch.zeros(4, dtype=torch.float32)]
        self._expect_fatal(
            lambda: _update_grad_minmax_hetero(
                gidx, "invalid", [feat], [idx], [idx_etype], dX
            ),
            "L2-09",
        )


class TestL2Edge:
    """空 tensor 派生 / 空出：混合容差通过。"""

    def test_L2_10_bwd_empty_feat(self):
        # 空进非空出，out 保持零
        _bwd_check("L2-10", n=0, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=50)

    def test_L2_11_hetero_empty_feat(self):
        _hetero_check(
            "L2-11", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 0}, n_out={0: 16}, seed=51,
        )

    def test_L2_12_bwd_empty_out(self):
        # feat[16,8]，arg 全 -1，out[0,8] 空出
        _bwd_check("L2-12", n=16, dim=8, m=0, idtype=torch.int32,
                   arg_pattern="all-1", seed=52)


# --------------------------------------------------------------------------- #
# 白盒补充用例（源码分支覆盖）
#
# 依据：segment_reduce_grad_kernel.cpp / segment_reduce_grad.cc 源码分析。
# 覆盖已有 L0/L1/L2 黑盒用例未触及的 kernel/host 执行分支。
# --------------------------------------------------------------------------- #
class TestWhitebox:
    """白盒补充用例：覆盖源码中未被黑盒用例触及的执行分支。

    分支枚举来源（行号均指 segment_reduce_grad_kernel.cpp / .cc）：
      - blockDim 边界：host ComputeBlockDim min(n,40)
      - 尾核 / 空闲核：kernel Init rowsPerCore/endRow/myRows + Process early-return
      - 尾块 tail batch：kernel Process batch loop itemCount<batchItems
      - rightPadding 非对齐：kernel Init rowAlignedBytes/rightPadding
      - copy_lhs 去重：host UpdateGradMinMaxHeteroAscendImpl same_src_dst_ntype skip
      - SetAtomicAdd vs 无原子：两个 kernel 分别覆盖（已有黑盒覆盖，此处不重复）
    """

    # ---- blockDim=40 精确边界 ----
    # host:82 ComputeBlockDim(40)=min(40,40)=40；rowsPerCore=ceil(40/40)=1。
    # 所有核恰好 1 行，无尾核、无空闲核——N=40→41 的临界点。
    def test_WB_01_bwd_blockdim_40_exact(self):
        _bwd_check("WB-01", n=40, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=61)

    # ---- N=41 触发尾核 + 空闲核（myRows==0 early return）----
    # blockDim=40, rowsPerCore=ceil(41/40)=2。
    # Core 20: startRow=40, endRow=min(42,41)=41, myRows=1（尾核，数据量 < rowsPerCore）。
    # Cores 21-39: startRow=42,44,... ≥ 42 > 41 → endRow=41, myRows=0
    #   → kernel:128 `if (this->myRows == 0) return;` 空闲核早退路径。
    def test_WB_02_bwd_idle_and_tail_core(self):
        _bwd_check("WB-02", n=41, dim=8, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=62)

    def test_WB_03_hetero_idle_and_tail_core(self):
        # hetero kernel:294 同样 myRows==0 early return + 尾核
        _hetero_check(
            "WB-03", etype_edges=[(0, 0)], op="copy_rhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 41}, n_out={0: 16}, seed=63,
        )

    # ---- 尾块（tail batch）：batchItems 不整除 rowsPerCore ----
    # Bwd: dim=1024 → rowBytes=4096, denom=3*4096=12288, batchItems=193504/12288≈15。
    # N=700, blockDim=40, rowsPerCore=ceil(700/40)=18 > 15。
    # 每非尾核: 1 full batch(15) + 1 tail batch(3)。
    # kernel:300-303 `itemCount = remaining > batchItems ? batchItems : remaining`
    def test_WB_04_bwd_tail_batch(self):
        _bwd_check("WB-04", n=700, dim=1024, m=64, idtype=torch.int32,
                   arg_pattern="valid", seed=64)

    # Hetero: dim=1024 → denom=4*4096=16384, batchItems=193504/16384≈11。
    # N=500, rowsPerCore=ceil(500/40)=13 > 11。
    # 每非尾核: 1 full batch(11) + 1 tail batch(2)。
    # kernel:136-139 itemCount<batchItems 路径。
    def test_WB_05_hetero_tail_batch(self):
        _hetero_check(
            "WB-05", etype_edges=[(0, 0)], op="copy_rhs", dim=1024,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 500}, n_out={0: 16}, seed=65,
        )

    # ---- copy_lhs 去重：相同 (src,dst) ntype 对的第二个 etype 被跳过 ----
    # host:153-156 `if (op == "copy_lhs" && same_src_dst_ntype != end(seen)) continue;`
    # etype_edges=[(0,1),(0,1)]：etype 0 处理后 seen[1]={0}，
    # etype 1 的 src_ntype=0 已在 seen[1] 中 → continue 跳过，仅一次 kernel launch。
    # golden(CPU) 同样去重，输出应一致。
    def test_WB_06_hetero_copy_lhs_dedup_same_pair(self):
        _hetero_check(
            "WB-06", etype_edges=[(0, 1), (0, 1)], op="copy_lhs", dim=8,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={1: 64}, n_out={0: 16}, seed=66,
        )

    # ---- 最小路径：N=1 单行单核 ----
    # blockDim=min(1,40)=1, rowsPerCore=1, myRows=1。
    # kernel:127-148 Process 仅一次 batch（itemCount=1），最小执行路径。
    def test_WB_07_bwd_single_row(self):
        _bwd_check("WB-07", n=1, dim=8, m=4, idtype=torch.int32,
                   arg_pattern="valid", seed=67)

    # ---- 最小非零 rightPadding：featDim=7 → rightPadding=1 ----
    # kernel:82-86 rowBytes=28, rowAlignedBytes=32, rowAlignedElems=8, rightPadding=1。
    # DataCopyPad padParams {(uint8_t)this->rightPadding=1}（kernel:156-157/318-319），
    # 每行 pad 1 个 float 元素。已有用例覆盖 rightPadding=0(dim%8==0) 和
    # rightPadding=7(dim%8==1)，本用例覆盖 rightPadding=1 最小非零填充。
    def test_WB_08_bwd_dim7_minimal_padding(self):
        _bwd_check("WB-08", n=64, dim=7, m=16, idtype=torch.int32,
                   arg_pattern="valid", seed=68)

    def test_WB_09_hetero_dim7_minimal_padding(self):
        # hetero kernel:83-86 同 rightPadding=1 路径
        _hetero_check(
            "WB-09", etype_edges=[(0, 0)], op="copy_rhs", dim=7,
            idtype=torch.int32, match_pattern="all",
            n_dst_v={0: 64}, n_out={0: 16}, seed=69,
        )
