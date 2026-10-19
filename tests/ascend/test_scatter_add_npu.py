"""
Test scatter_add (ScatterAdd) on Ascend NPU.

Tests the Ascend ScatterAdd kernel: out[idx[i], *] += feat[i, *] on the first
dimension. Mirrors the pytest style of test_segment_reduce_npu.py /
test_index_select_npu.py.

Coverage: 46 cases — L0(6) + L1(24) + L2(9) + 补-1(1) blackbox + WB(6) whitebox.
  - L0/L1/补-1/WB: precision cases, asserted via mixed tolerance gate.
  - L2-1..L2-5: boundary cases.
  - L2-6..L2-9: dtype LOG(FATAL) exception cases.

Golden: `dgl.backend.scatter_add` CPU dispatch → cpu::ScatterAdd (semantic
authority). torch.Tensor.scatter_add_ is an auxiliary cross-check only.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dgl.backend as F  # noqa: E402


# ---------------------------------------------------------------------------
# Precision constants (float32 mixed tolerance).
# ---------------------------------------------------------------------------

RTOL = 2 ** (-10)
ATOL = 2 ** (-16)
REQUIRED_MATCHED_RATIO = 0.99
MAX_ABS_ERROR_LIMIT = 1e-2
_ULP32 = 2 ** (-23)
_MAX_ABS_ERROR_LIMIT_EFF = max(MAX_ABS_ERROR_LIMIT, 32 * _ULP32)
AUX_RTOL = 1e-4
AUX_ATOL = 1e-4

RESULTS_DIR = os.environ.get(
    "SCATTER_ADD_PREC_RESULTS_DIR",
    os.path.join(os.path.dirname(__file__), "results"),
)


# ---------------------------------------------------------------------------
# Precision gate
# ---------------------------------------------------------------------------

def check_precision(out_npu: torch.Tensor, out_golden: torch.Tensor) -> Dict:
    npu = out_npu.detach().cpu().to(torch.float32).numpy()
    golden = out_golden.detach().cpu().to(torch.float32).numpy()

    if npu.shape != golden.shape:
        raise ValueError(
            f"Shape mismatch: npu {npu.shape} vs golden {golden.shape}"
        )

    g64 = golden.astype(np.float64)
    a64 = npu.astype(np.float64)
    finite_mask = np.isfinite(g64)
    g64_safe = np.where(finite_mask, g64, np.float64(0.0))
    a64_safe = np.where(finite_mask, a64, np.float64(0.0))
    abs_err_finite = np.abs(a64_safe - g64_safe)
    elem_threshold = ATOL + RTOL * np.abs(g64_safe)
    finite_passed = (abs_err_finite <= elem_threshold) & finite_mask

    both_nonfinite = (~finite_mask) & (~np.isfinite(a64))
    inf_match = (
        np.isinf(a64) & np.isinf(g64)
        & (np.signbit(a64) == np.signbit(g64))
    )
    nan_match = np.isnan(a64) & np.isnan(g64)
    nonfinite_passed = (inf_match | nan_match) & both_nonfinite

    elem_passed = finite_passed | nonfinite_passed

    finite_count = int(np.sum(finite_mask))
    max_abs_error = float(np.max(abs_err_finite)) if finite_count > 0 else 0.0

    total = npu.size
    if total == 0:
        matched_ratio = 1.0
        max_abs_error = 0.0
    else:
        matched_ratio = float(np.sum(elem_passed)) / total

    ratio_pass = matched_ratio >= REQUIRED_MATCHED_RATIO
    max_err_pass = max_abs_error <= _MAX_ABS_ERROR_LIMIT_EFF
    is_pass = bool(ratio_pass and max_err_pass)

    aux_pass = bool(torch.allclose(
        out_npu.detach().cpu().to(torch.float32),
        out_golden.detach().cpu().to(torch.float32),
        rtol=AUX_RTOL, atol=AUX_ATOL, equal_nan=True,
    ))

    return {
        "output_tensor": "out",
        "dtype": "float32",
        "metric": "mixed_tolerance",
        "rtol": RTOL,
        "atol": ATOL,
        "matched_ratio": matched_ratio,
        "required_matched_ratio": REQUIRED_MATCHED_RATIO,
        "max_abs_error": max_abs_error,
        "max_abs_error_limit": _MAX_ABS_ERROR_LIMIT_EFF,
        "verdict": "PASS" if is_pass else "FAIL",
        "ratio_pass": bool(ratio_pass),
        "max_err_pass": bool(max_err_pass),
        "aux_allclose": aux_pass,
        "aux_divergent_from_authoritative": bool(is_pass != aux_pass),
        "shape": tuple(npu.shape),
        "total_elements": int(total),
    }


def assert_precision(
    case_id: str,
    out_npu: torch.Tensor,
    out_golden: torch.Tensor,
    *,
    persist: bool = True,
) -> Dict:
    result = check_precision(out_npu, out_golden)
    result["case_id"] = case_id
    result["authoritative_source"] = "float32 mixed tolerance"

    if persist:
        try:
            import json
            os.makedirs(RESULTS_DIR, exist_ok=True)
            path = os.path.join(RESULTS_DIR, f"{case_id}.json")
            with open(path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            print(f"[warn] failed to persist precision record for {case_id}: {e}")

    assert result["verdict"] == "PASS", (
        f"[{case_id}] precision FAIL: "
        f"matched_ratio={result['matched_ratio']:.6f} "
        f"(req {REQUIRED_MATCHED_RATIO}), "
        f"max_abs_error={result['max_abs_error']:.6g} "
        f"(limit {_MAX_ABS_ERROR_LIMIT_EFF})."
    )
    return result


def summarize(records: List[Dict]) -> Dict:
    total = len(records)
    passed = sum(1 for r in records if r["verdict"] == "PASS")
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "matched_ratio_min": (
            min(r["matched_ratio"] for r in records) if records else 0.0
        ),
        "max_abs_error_max": (
            max(r["max_abs_error"] for r in records) if records else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Golden — authoritative path via cpu::ScatterAdd (dgl.backend.scatter_add CPU)
# ---------------------------------------------------------------------------

def golden_scatter_add_cpu(
    feat_cpu: torch.Tensor, idx_cpu: torch.Tensor, m: int
) -> torch.Tensor:
    out = F.scatter_add(feat_cpu, idx_cpu, m)
    return out.detach().clone()


def aux_torch_scatter_add(
    feat_cpu: torch.Tensor, idx_cpu: torch.Tensor, m: int
) -> torch.Tensor:
    """Auxiliary cross-check only (NOT a pass gate)."""
    out = torch.zeros((m,) + tuple(feat_cpu.shape[1:]), dtype=feat_cpu.dtype)
    idx_long = idx_cpu.to(torch.int64)
    view_shape = [idx_long.shape[0]] + [1] * (feat_cpu.ndim - 1)
    idx_expanded = idx_long.view(view_shape).expand_as(feat_cpu)
    out.scatter_add_(0, idx_expanded, feat_cpu)
    return out


# ---------------------------------------------------------------------------
# Case table
# ---------------------------------------------------------------------------

@dataclass
class Case:
    case_id: str
    level: str
    desc: str
    build: Callable[[], tuple]  # () -> (feat_cpu, idx_cpu, M, idtype_str)


FLT_MAX = float(np.finfo(np.float32).max)
FLT_MIN_NORMAL = float(np.finfo(np.float32).tiny)
FLT_SUBNORM_MIN = float(np.finfo(np.float32).smallest_subnormal)


def _randn(*shape, **kw):
    return torch.randn(*shape, **kw)


def _make_feat_with_row_value(n, dim, row_idx, value):
    f = torch.randn(n, dim, dtype=torch.float32)
    f[row_idx] = float(value)
    return f


# ---- L0 (6) ----

def _l0_1():
    feat = _randn(4, 16, dtype=torch.float32)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l0_2():
    feat = _randn(2, 4, dtype=torch.float32)
    idx = torch.tensor([0, 1], dtype=torch.int64)
    return feat, idx, 2, "int64"

def _l0_3():
    feat = _randn(4, 8, dtype=torch.float32)
    idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    return feat, idx, 2, "int32"

def _l0_4():
    feat = _randn(2, 8, dtype=torch.float32)
    idx = torch.tensor([1, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l0_5():
    feat = _randn(4, 1, dtype=torch.float32)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l0_6():
    feat = _randn(4, 17, dtype=torch.float32)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"


# ---- L1 (24) ----

def _l1_1():
    feat = _randn(4, 16, dtype=torch.float32)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_2():
    feat = _randn(8, 16, dtype=torch.float32)
    idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_3():
    feat = _randn(4, 16, dtype=torch.float32)
    idx = torch.tensor([1, 3, 5, 7], dtype=torch.int32)
    return feat, idx, 8, "int32"

def _l1_4():
    feat = _randn(4, 16, dtype=torch.float32)
    idx = torch.tensor([3, 2, 1, 0], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_5():
    n = 1024
    feat = _randn(n, 1, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int32)
    return feat, idx, n, "int32"

def _l1_6():
    n = 64
    feat = _randn(n, 17, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int32)
    return feat, idx, n, "int32"

def _l1_7():
    n = 64
    feat = _randn(n, 33, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int32)
    return feat, idx, n, "int32"

def _l1_8():
    n = 1024
    feat = _randn(n, 256, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int32)
    return feat, idx, n, "int32"

def _l1_9():
    n = 64
    feat = _randn(n, 16, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int32)
    return feat, idx, n, "int32"

def _l1_10():
    n = 64
    feat = _randn(n, 16, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int64)
    return feat, idx, n, "int64"

def _l1_11():
    n = 1024
    feat = _randn(n, 16, dtype=torch.float32)
    idx = torch.zeros(n, dtype=torch.int32)
    return feat, idx, 1, "int32"

def _l1_12():
    feat = _randn(4, 16, dtype=torch.float32)
    idx = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
    return feat, idx, 8, "int32"

def _l1_13():
    n = 4096
    feat = _randn(n, 128, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int32)
    return feat, idx, n, "int32"

def _l1_14():
    feat = torch.tensor(
        [[0.0, -0.0, 0.0, -0.0],
         [-0.0, 0.0, -0.0, 0.0],
         [0.0, -0.0, 0.0, -0.0],
         [-0.0, 0.0, -0.0, 0.0]],
        dtype=torch.float32,
    )
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_15():
    feat = _randn(4, 4, dtype=torch.float32)
    feat[0] = float("inf")
    feat[1] = float("inf")
    idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    return feat, idx, 2, "int32"

def _l1_16():
    feat = _randn(4, 4, dtype=torch.float32)
    feat[0] = float("-inf")
    feat[1] = float("-inf")
    idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    return feat, idx, 2, "int32"

def _l1_17():
    feat = _randn(2, 4, dtype=torch.float32)
    feat[0] = float("inf")
    feat[1] = float("-inf")
    idx = torch.tensor([0, 0], dtype=torch.int32)
    return feat, idx, 1, "int32"

def _l1_18():
    feat = _randn(4, 4, dtype=torch.float32)
    feat[0, 0] = float("nan")
    feat[1, 1] = float("nan")
    idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    return feat, idx, 2, "int32"

def _l1_19():
    n = 1024
    feat = torch.zeros(n, 16, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int32)
    return feat, idx, n, "int32"

def _l1_20():
    base = torch.tensor(
        [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3, -1e-3, -1e-2, -1e-1,
         -1.0, -10.0, -1e3, 5e-2, -5e-2, 0.0, 1.0],
        dtype=torch.float32,
    )
    feat = base.repeat(4, 1)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_21():
    feat = torch.tensor(
        [[FLT_MAX, -FLT_MAX, 1.0, 2.0],
         [-FLT_MAX, FLT_MAX, 3.0, 4.0],
         [FLT_MAX, FLT_MAX, 5.0, 6.0],
         [-FLT_MAX, -FLT_MAX, 7.0, 8.0]],
        dtype=torch.float32,
    )
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_22():
    feat = torch.tensor(
        [[FLT_MIN_NORMAL, -FLT_MIN_NORMAL, 1.0, 2.0],
         [-FLT_MIN_NORMAL, FLT_MIN_NORMAL, 3.0, 4.0],
         [FLT_MIN_NORMAL, FLT_MIN_NORMAL, 5.0, 6.0],
         [-FLT_MIN_NORMAL, -FLT_MIN_NORMAL, 7.0, 8.0]],
        dtype=torch.float32,
    )
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_23():
    feat = torch.tensor(
        [[FLT_SUBNORM_MIN, -FLT_SUBNORM_MIN, 1.0, 2.0],
         [-FLT_SUBNORM_MIN, FLT_SUBNORM_MIN, 3.0, 4.0],
         [FLT_SUBNORM_MIN, FLT_SUBNORM_MIN, 5.0, 6.0],
         [-FLT_SUBNORM_MIN, -FLT_SUBNORM_MIN, 7.0, 8.0]],
        dtype=torch.float32,
    )
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l1_24():
    feat = _randn(8, 16, dtype=torch.float32)
    idx = torch.tensor([0, 7, 0, 7, 0, 7, 0, 7], dtype=torch.int32)
    return feat, idx, 8, "int32"


# ---- L2 boundary (5) ----

def _l2_1():
    feat = torch.tensor([[0.5]], dtype=torch.float32)
    idx = torch.tensor([0], dtype=torch.int32)
    return feat, idx, 1, "int32"

def _l2_2():
    feat = torch.zeros(0, 16, dtype=torch.float32)
    idx = torch.zeros(0, dtype=torch.int32)
    return feat, idx, 4, "int32"

def _l2_3():
    n = 1024
    feat = _randn(n, 16, dtype=torch.float32)
    idx = torch.zeros(n, dtype=torch.int32)
    return feat, idx, 1, "int32"

def _l2_4():
    feat = _randn(1, 16, dtype=torch.float32)
    idx = torch.tensor([torch.randint(0, 1024, (1,)).item()],
                       dtype=torch.int32)
    return feat, idx, 1024, "int32"

def _l2_5():
    feat = torch.tensor([[1.0]], dtype=torch.float32)
    idx = torch.tensor([0], dtype=torch.int32)
    return feat, idx, 1, "int32"


# ---- 补充 (1) ----

def _supp_1():
    n = 1024
    feat = _randn(n, 16, dtype=torch.float32)
    idx = torch.full((n,), 1023, dtype=torch.int32)
    return feat, idx, 1024, "int32"


# ---- Whitebox (6) ----

def _wb_1():
    n = 40
    feat = torch.randn(n, 16, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int32)
    return feat, idx, n, "int32"

def _wb_2():
    n = 41
    feat = torch.randn(n, 16, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int32)
    return feat, idx, n, "int32"

def _wb_3():
    n = 8
    feat_dim = 20001
    feat = torch.randn(n, feat_dim, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int32)
    return feat, idx, n, "int32"

def _wb_4():
    feat = torch.randn(4, 2, 8, dtype=torch.float32)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    return feat, idx, 4, "int32"

def _wb_5():
    n = 80
    feat = torch.randn(n, 16, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int32)
    return feat, idx, n, "int32"

def _wb_6():
    n = 240
    feat_dim = 4097
    feat = torch.randn(n, feat_dim, dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.int32)
    return feat, idx, n, "int32"


# ---- Case registries ----

CASES: List[Case] = [
    Case("L0-1", "L0", "基础顺序 N=M=4 dim=16 int32", _l0_1),
    Case("L0-2", "L0", "IdType=int64 最小", _l0_2),
    Case("L0-3", "L0", "N>M 重复命中", _l0_3),
    Case("L0-4", "L0", "N<M 间隙", _l0_4),
    Case("L0-5", "L0", "dim=1", _l0_5),
    Case("L0-6", "L0", "dim 非对齐 17", _l0_6),
    Case("L1-1", "L1", "N4 正常组合", _l1_1),
    Case("L1-2", "L1", "N>M 重复命中", _l1_2),
    Case("L1-3", "L1", "N<M 间隙", _l1_3),
    Case("L1-4", "L1", "idx 逆序", _l1_4),
    Case("L1-5", "L1", "dim=1 大规模", _l1_5),
    Case("L1-6", "L1", "dim=17 非对齐", _l1_6),
    Case("L1-7", "L1", "dim=33 非对齐", _l1_7),
    Case("L1-8", "L1", "dim=256 对齐大规模", _l1_8),
    Case("L1-9", "L1", "IdType=int32 中规模", _l1_9),
    Case("L1-10", "L1", "IdType=int64 中规模", _l1_10),
    Case("L1-11", "L1", "全部命中同一行", _l1_11),
    Case("L1-12", "L1", "部分行未命中", _l1_12),
    Case("L1-13", "L1", "GNN 典型规模", _l1_13),
    Case("L1-14", "L1", "feat 含 ±0", _l1_14),
    Case("L1-15", "L1", "feat 含 +inf", _l1_15),
    Case("L1-16", "L1", "feat 含 -inf", _l1_16),
    Case("L1-17", "L1", "feat 含 ±inf 同行", _l1_17),
    Case("L1-18", "L1", "feat 含 nan", _l1_18),
    Case("L1-19", "L1", "feat 全 0", _l1_19),
    Case("L1-20", "L1", "feat 量级带覆盖", _l1_20),
    Case("L1-21", "L1", "feat 含 ±FLT_MAX", _l1_21),
    Case("L1-22", "L1", "feat 含最小正规", _l1_22),
    Case("L1-23", "L1", "feat 含次正规", _l1_23),
    Case("L1-24", "L1", "idx 含 0 与 M-1", _l1_24),
    Case("L2-1", "L2", "最小 shape 单元素", _l2_1),
    Case("L2-2", "L2", "空 idx/空 feat", _l2_2),
    Case("L2-3", "L2", "M=1 全命中唯一行", _l2_3),
    Case("L2-4", "L2", "N=1 单行散射", _l2_4),
    Case("L2-5", "L2", "dim=1 且 N=M=1", _l2_5),
    Case("补-1", "补充", "idx 全 M-1", _supp_1),
    Case("WB-1", "WB", "N=40 blockDim 边界", _wb_1),
    Case("WB-2", "WB", "N=41 尾核+空闲核", _wb_2),
    Case("WB-3", "WB", "大 featDim batchItems=1", _wb_3),
    Case("WB-4", "WB", "3D feat 多维", _wb_4),
    Case("WB-5", "WB", "N=80 整除 blockDim", _wb_5),
    Case("WB-6", "WB", "大 featDim 满批+尾批", _wb_6),
]

EXCEPTION_CASES: List[tuple] = [
    ("L2-6", torch.float16, torch.int32, "DType=half LOG(FATAL)"),
    ("L2-7", torch.bfloat16, torch.int64, "DType=bfloat16 LOG(FATAL)"),
    ("L2-8", torch.float64, torch.int32, "DType=double LOG(FATAL)"),
    ("L2-9", torch.int8, torch.int32, "DType=非浮点 LOG(FATAL)"),
]


# ---------------------------------------------------------------------------
# NPU device helpers
# ---------------------------------------------------------------------------

def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()

def _get_npu_device() -> torch.device:
    if not _npu_available():
        pytest.skip("NPU device not available; skipping NPU tests.")
    dev = torch.device("npu:0")
    torch.npu.set_device(dev)
    return dev

def _sync(dev: torch.device) -> None:
    if dev.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(dev)


# ---------------------------------------------------------------------------
# Test runner — precision cases
# ---------------------------------------------------------------------------

_SUITE_RECORDS: list = []

def _run_precision_case(case):
    dev = _get_npu_device()
    feat_cpu, idx_cpu, m, _idtype = case.build()
    assert feat_cpu.dtype == torch.float32
    assert idx_cpu.dtype in (torch.int32, torch.int64)

    out_golden = golden_scatter_add_cpu(feat_cpu, idx_cpu, m)

    feat_npu = feat_cpu.to(dev)
    idx_npu = idx_cpu.to(dev)
    out_npu = F.scatter_add(feat_npu, idx_npu, m)
    _sync(dev)
    out_npu_cpu = out_npu.detach().cpu()

    rec = assert_precision(case.case_id, out_npu_cpu, out_golden)
    _SUITE_RECORDS.append(rec)

    aux = aux_torch_scatter_add(feat_cpu, idx_cpu, m)
    aux_match = torch.equal(
        out_golden.to(torch.float32), aux.to(torch.float32)
    ) or torch.allclose(
        out_golden.to(torch.float32), aux.to(torch.float32),
        rtol=1e-4, atol=1e-4, equal_nan=True,
    )
    if not aux_match:
        print(
            f"[{case.case_id}] aux torch.scatter_add_ diverges from "
            f"cpu::ScatterAdd golden — CPU golden is authoritative."
        )

    if case.case_id == "L2-2":
        assert out_npu_cpu.numel() > 0
        assert torch.all(out_npu_cpu == 0.0)
    if case.case_id in ("L1-12", "L0-4"):
        assert (out_npu_cpu == 0.0).any()
    if case.case_id == "WB-3":
        assert torch.allclose(
            out_npu_cpu.to(torch.float32), feat_cpu.to(torch.float32),
            rtol=1e-4, atol=1e-4,
        )
    if case.case_id == "WB-5":
        assert out_npu_cpu.shape[0] == 80
        assert not torch.all(out_npu_cpu == 0.0)
    if case.case_id == "WB-4":
        assert out_npu_cpu.ndim == 3
        assert tuple(out_npu_cpu.shape) == (4, 2, 8)


def _make_precision_test(case):
    def test_fn(self=None):
        _run_precision_case(case)
    test_fn.__name__ = f"test_{case.case_id.replace('-', '_')}"
    test_fn.__doc__ = f"[{case.level}] {case.case_id}: {case.desc}"
    return test_fn

for _c in CASES:
    fn = _make_precision_test(_c)
    globals()[fn.__name__] = fn


# ---------------------------------------------------------------------------
# Test runner — exception cases
# ---------------------------------------------------------------------------

def _run_exception_case(case_id, feat_dtype, idx_dtype, desc):
    dev = _get_npu_device()
    if feat_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        feat = torch.randn(4, 4).to(feat_dtype)
    else:
        feat = torch.tensor([[1, 2, 3, 4]] * 4, dtype=feat_dtype)
    idx = torch.tensor([0, 0, 1, 1], dtype=idx_dtype)
    m = 2

    feat_npu = feat.to(dev)
    idx_npu = idx.to(dev)
    with pytest.raises(Exception):
        F.scatter_add(feat_npu, idx_npu, m)
        _sync(dev)

def _make_exception_test(case_id, feat_dtype, idx_dtype, desc):
    def test_fn(self=None):
        _run_exception_case(case_id, feat_dtype, idx_dtype, desc)
    test_fn.__name__ = f"test_{case_id.replace('-', '_')}"
    test_fn.__doc__ = f"[L2] {case_id}: {desc}"
    return test_fn

for _cid, _fd, _id, _desc in EXCEPTION_CASES:
    fn = _make_exception_test(_cid, _fd, _id, _desc)
    globals()[fn.__name__] = fn


# ---------------------------------------------------------------------------
# Suite-level summary
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _persist_suite_summary():
    yield
    try:
        import json
        os.makedirs(RESULTS_DIR, exist_ok=True)
        summary = summarize(_SUITE_RECORDS)
        with open(os.path.join(RESULTS_DIR, "suite_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(
            f"\n[scatter_add suite] {summary['passed']}/{summary['total']} "
            f"PASS, {summary['failed']} FAIL"
        )
    except Exception as e:
        print(f"[warn] suite summary persistence failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _npu_available():
        print("NPU not available; exiting.")
        sys.exit(0)
    fn_names = sorted(globals().keys())
    failures = 0
    total = 0
    for name in fn_names:
        if not name.startswith("test_"):
            continue
        fn = globals()[name]
        if not callable(fn):
            continue
        total += 1
        try:
            fn()
            print(f"  PASS [{name}]")
        except Exception as e:
            print(f"  FAIL [{name}] {e}")
            failures += 1
    print(f"\nResults: {total - failures}/{total} passed, {failures} failed")
    sys.exit(1 if failures else 0)
