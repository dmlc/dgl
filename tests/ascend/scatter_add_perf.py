"""
Performance collection framework for scatter_add on Ascend NPU.

Produces:
  - multi-round wall-time collection (warmup-excluded; >=30 rounds; records
    median / min / max / p90 / mean / cv for volatility handling)
  - bottleneck dimension breakdown: compute (device kernel time, extracted via
    torch_npu.profiler `kernel_details.csv`) / H2D (data copy-in) / D2H (data
    copy-out) / scalar (host launch + dispatch + sync residual) -- as us and as
    shares of the op wall time
  - CPU fallback baseline (D2H + cpu::ScatterAdd + H2D) end-to-end, for direct
    native-vs-fallback comparison
  - JSON + JSONL + markdown report

Coverage (int32/int64 x float32):
  - L1-8  : N=1024 dim=256 int32   (aligned large, compute-bound)
  - L1-13 : N=4096 dim=128 int32   (GNN typical scale)
  - L1-5  : N=1024 dim=1   int32   (narrow, scalar/launch-bound)
  - L1-6  : N=64   dim=17  int32   (non-aligned, small)
  - L1-11 : N=1024 dim=16  int32  (all hit same row, atomic-conflict bound)
  - L1-10 : N=64   dim=16  int64   (IdType=int64 representative)
  - PERF-I64-L : N=4096 dim=128 int64 (IdType=int64 large scale — covers
                 int64 × float32 at representative GNN scale)
  - L2-3  : N=1024 dim=16  int32  (M=1 all hit unique row)

Usage (standalone):

    PYTHONPATH=python python tests/ascend/scatter_add_perf.py \
        --warmup 5 --rounds 30 --device 4

The profiler path (compute/scalar split) needs torch_npu.profiler; if it is
unavailable the framework falls back to an isolated-timing estimate but flags
the breakdown as heuristic.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Allow sibling imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import dgl.backend as F  # noqa: E402

from scatter_add_golden import CASES  # noqa: E402


# Final perf data lands in the test-dir collection output.
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PERF_RESULTS_DIR = os.environ.get(
    "SCATTER_ADD_PERF_RESULTS_DIR",
    os.path.join(TEST_DIR, "perf_output", "scatter_add"),
)
PROF_SCRATCH_DIR = os.environ.get(
    "SCATTER_ADD_PROF_SCRATCH_DIR",
    os.path.join(TEST_DIR, "prof_scratch"),
)

# A large-scale int64 case not present in the golden table (perf-only).
def _perf_i64_large():
    n = 4096
    feat = torch.randn(n, 128, dtype=torch.float32)
    idx = torch.randint(0, n, (n,), dtype=torch.int64)
    return feat, idx, n, "int64"


# Ordered perf-relevant case set covering int32/int64 × float32.
# (case_id, builder-or-None). None ⇒ resolve from CASES table.
PERF_CASES: List[Tuple[str, Optional[callable]]] = [
    ("L1-8", None),
    ("L1-13", None),
    ("L1-5", None),
    ("L1-6", None),
    ("L1-11", None),
    ("L1-10", None),
    ("PERF-I64-L", _perf_i64_large),
    ("L2-3", None),
]

# Kernel name substrings matched in kernel_details.csv for device-time split.
KERNEL_SCATTER = "kernel_scatter_add"
KERNEL_ZEROS = "aclnnInplaceZero"  # F.zeros memset kernel (part of op)


@dataclass
class PerfRecord:
    case_id: str
    level: str
    desc: str
    shape: Dict[str, int]
    idtype: str
    rounds: int
    # Native NPU op wall-time (warmup excluded), microseconds.
    op_us_median: float
    op_us_std: float
    op_us_min: float
    op_us_max: float
    op_us_p50: float
    op_us_p90: float
    op_us_mean: float
    op_us_cv: float
    # CPU-fallback (D2H+CPU ScatterAdd+H2D) wall-time, microseconds.
    cpu_fallback_us_median: float = 0.0
    cpu_fallback_us_min: float = 0.0
    cpu_fallback_us_max: float = 0.0
    speedup_vs_cpu: float = 0.0  # cpu_fallback / native op
    # Bottleneck breakdown (microseconds).
    compute_us: float = 0.0       # device kernel time (scatter_add + zeros)
    scatter_kernel_us: float = 0.0
    zeros_kernel_us: float = 0.0
    h2d_us: float = 0.0           # isolated input copy-in
    d2h_us: float = 0.0           # isolated output copy-out
    scalar_us: float = 0.0        # host launch+dispatch+sync residual
    # AICore utilization (from op_statistic.csv Ratio%).
    aicore_util_pct: float = 0.0   # scatter_add kernel AICore utilization %
    zeros_util_pct: float = 0.0
    # Shares of op wall time (%).
    compute_share: float = 0.0
    h2d_share: float = 0.0
    d2h_share: float = 0.0
    scalar_share: float = 0.0
    samples: List[float] = field(default_factory=list)
    cpu_samples: List[float] = field(default_factory=list)
    volatile: bool = False
    breakdown_method: str = ""
    notes: str = ""


# ----------------------------- device helpers ----------------------------- #

def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_device(device_id: int) -> torch.device:
    dev = torch.device(f"npu:{device_id}")
    torch.npu.set_device(dev)
    return dev


def _sync(dev: torch.device) -> None:
    if dev.type == "npu":
        torch.npu.synchronize(dev)


def _measure(fn, dev, warmup, rounds) -> List[float]:
    """Run `fn()` warmup times then `rounds` timed rounds (us per round)."""
    for _ in range(warmup):
        fn()
    _sync(dev)
    samples = []
    for _ in range(rounds):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1_000_000.0)
    return samples


def _stats(samples: List[float]) -> Dict[str, float]:
    s = sorted(samples)
    n = len(s)

    def _pct(p):
        if n == 0:
            return 0.0
        k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return s[k]

    median = statistics.median(samples) if samples else 0.0
    mean = statistics.mean(samples) if samples else 0.0
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median": float(median),
        "mean": float(mean),
        "std": float(std),
        "min": float(min(samples)) if samples else 0.0,
        "max": float(max(samples)) if samples else 0.0,
        "p50": float(_pct(50)),
        "p90": float(_pct(90)),
        "cv": float(std / median) if median > 0 else 0.0,
    }


def _resolve_case(case_id: str) -> Tuple:
    """Return (feat_cpu, idx_cpu, m, idtype, level, desc)."""
    for cid, builder in PERF_CASES:
        if cid == case_id:
            if builder is not None:
                feat, idx, m, idtype = builder()
                return feat, idx, m, idtype, "PERF", "perf int64 large scale"
            break
    for c in CASES:
        if c.case_id == case_id:
            feat, idx, m, idtype = c.build()
            return feat, idx, m, idtype, c.level, c.desc
    raise KeyError(f"case {case_id} not found")


# ----------------------------- profiler split ----------------------------- #

def _profiler_kernel_us(dev: torch.device, feat_npu, idx_npu, m: int,
                        case_id: str, out_dir: str) -> Dict[str, float]:
    """
    Capture a traced run (Level1 + PipeUtilization) and extract device kernel
    durations + AICore utilization from op_statistic.csv. Returns:
      scatter_us, zeros_us, compute_us (scatter+zeros Avg),
      aicore_util_pct (scatter_add kernel AICore Ratio%),
      zeros_util_pct.

    op_statistic.csv gives per-op Avg(us) and Ratio(%) over the recorded
    iterations — more robust than per-step kernel_details medians.
    """
    try:
        import torch_npu.profiler as npp
        from torch_npu.profiler import (AiCMetrics, ProfilerActivity,
                                        ProfilerAction, ProfilerLevel,
                                        profile, schedule)
        from torch_npu.profiler.experimental_config import \
            _ExperimentalConfig as EC
    except Exception as e:  # noqa: BLE001
        return {"scatter_us": 0.0, "zeros_us": 0.0, "compute_us": 0.0,
                "error": f"profiler unavailable: {e}"}

    case_prof_dir = os.path.join(out_dir, case_id)
    if os.path.exists(case_prof_dir):
        shutil.rmtree(case_prof_dir, ignore_errors=True)
    os.makedirs(case_prof_dir, exist_ok=True)

    # Warmup so the traced iterations are steady.
    for _ in range(3):
        F.scatter_add(feat_npu, idx_npu, m)
        _sync(dev)

    cfg = EC(profiler_level=ProfilerLevel.Level1,
             aic_metrics=AiCMetrics.PipeUtilization)

    def _sch(step):
        if step < 3:
            return ProfilerAction.RECORD
        if step == 3:
            return ProfilerAction.RECORD_AND_SAVE
        return ProfilerAction.NONE

    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                     schedule=_sch, experimental_config=cfg,
                     on_trace_ready=npp.tensorboard_trace_handler(case_prof_dir)) as prof:
            for i in range(5):
                F.scatter_add(feat_npu, idx_npu, m)
                _sync(dev)
                prof.step()
    except Exception as e:  # noqa: BLE001
        return {"scatter_us": 0.0, "zeros_us": 0.0, "compute_us": 0.0,
                "error": f"profile failed: {e}"}

    os_csv = glob.glob(os.path.join(case_prof_dir, "**", "op_statistic.csv"),
                      recursive=True)
    if not os_csv:
        return {"scatter_us": 0.0, "zeros_us": 0.0, "compute_us": 0.0,
                "error": "op_statistic.csv not found"}

    scatter_us = zeros_us = 0.0
    scatter_util = zeros_util = 0.0
    with open(os_csv[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row.get("OP Type", "")
            try:
                avg = float(row.get("Avg Time(us)", "0"))
            except (ValueError, TypeError):
                avg = 0.0
            try:
                ratio = float(row.get("Ratio(%)", "0"))
            except (ValueError, TypeError):
                ratio = 0.0
            if KERNEL_SCATTER in op_type:
                scatter_us = avg
                scatter_util = ratio
            elif "Zero" in op_type or "zero" in op_type:
                zeros_us = avg
                zeros_util = ratio

    return {
        "scatter_us": scatter_us,
        "zeros_us": zeros_us,
        "compute_us": scatter_us + zeros_us,
        "aicore_util_pct": scatter_util,
        "zeros_util_pct": zeros_util,
    }


# ----------------------------- collection ----------------------------- #

def collect_one(case_id: str, dev: torch.device, warmup: int,
                rounds: int, prof_dir: str) -> PerfRecord:
    feat_cpu, idx_cpu, m, idtype, level, desc = _resolve_case(case_id)

    feat_npu = feat_cpu.to(dev)
    idx_npu = idx_cpu.to(dev)
    _sync(dev)

    # ---- 1) Native NPU op wall-time (resident inputs) ----
    def _op_fn():
        return F.scatter_add(feat_npu, idx_npu, m)

    op_samples = _measure(_op_fn, dev, warmup, rounds)
    op_stats = _stats(op_samples)

    # ---- 2) CPU fallback: D2H(feat,idx) + cpu::ScatterAdd + H2D(out) ----
    # End-to-end realistic fallback as it would run if dispatch fell back.
    def _cpu_fallback_fn():
        f_c = feat_npu.cpu()
        i_c = idx_npu.cpu()
        out_c = F.scatter_add(f_c, i_c, m)  # CPU dispatch → cpu::ScatterAdd
        _ = out_c.to(dev)

    cpu_samples = _measure(_cpu_fallback_fn, dev, warmup=max(2, warmup // 2),
                           rounds=max(10, rounds // 2))
    cpu_stats = _stats(cpu_samples)

    # ---- 3) H2D (both inputs) ----
    def _h2d_fn():
        _ = feat_cpu.to(dev)
        _ = idx_cpu.to(dev)

    h2d_samples = _measure(_h2d_fn, dev, warmup=max(1, warmup // 2),
                           rounds=max(10, rounds // 2))
    h2d_stats = _stats(h2d_samples)

    # ---- 4) D2H (op output) ----
    out_ref = F.scatter_add(feat_npu, idx_npu, m)
    _sync(dev)

    def _d2h_fn():
        _ = out_ref.cpu()

    d2h_samples = _measure(_d2h_fn, dev, warmup=max(1, warmup // 2),
                           rounds=max(10, rounds // 2))
    d2h_stats = _stats(d2h_samples)

    # ---- 5) compute/scalar split via profiler ----
    prof = _profiler_kernel_us(dev, feat_npu, idx_npu, m, case_id, prof_dir)
    aicore_util = prof.get("aicore_util_pct", 0.0)
    zeros_util = prof.get("zeros_util_pct", 0.0)
    if prof.get("compute_us", 0.0) > 0 and "error" not in prof:
        compute_us = prof["compute_us"]
        scatter_us = prof["scatter_us"]
        zeros_us = prof["zeros_us"]
        method = ("torch_npu.profiler op_statistic.csv (Avg Duration + "
                  "AICore Ratio%, Level1+PipeUtilization)")
    else:
        # Fallback heuristic (flagged): compute ≈ op minus host launch-only.
        def _launch_only_fn():
            return F.scatter_add(feat_npu, idx_npu, m)
        launch_samples = _measure(_launch_only_fn, dev, warmup=max(1, warmup // 2),
                                  rounds=max(10, rounds // 2))
        launch_med = _stats(launch_samples)["median"]
        compute_us = max(0.0, op_stats["median"] - launch_med)
        scatter_us = compute_us
        zeros_us = 0.0
        method = ("heuristic isolated (op_sync - op_launch_no_sync); "
                  f"profiler unavailable: {prof.get('error','')}")

    op_med = op_stats["median"] or 1e-9
    scalar_us = max(0.0, op_med - compute_us)

    speedup = (cpu_stats["median"] / op_med) if op_med > 0 else 0.0
    volatile = op_stats["cv"] > 0.10

    shape_info = {
        "N": int(feat_cpu.shape[0]),
        "dim": int(feat_cpu.shape[1]) if feat_cpu.dim() > 1 else 1,
        "M": int(m),
    }

    return PerfRecord(
        case_id=case_id,
        level=level,
        desc=desc,
        shape=shape_info,
        idtype=idtype,
        rounds=rounds,
        op_us_median=op_stats["median"],
        op_us_std=op_stats["std"],
        op_us_min=op_stats["min"],
        op_us_max=op_stats["max"],
        op_us_p50=op_stats["p50"],
        op_us_p90=op_stats["p90"],
        op_us_mean=op_stats["mean"],
        op_us_cv=op_stats["cv"],
        cpu_fallback_us_median=cpu_stats["median"],
        cpu_fallback_us_min=cpu_stats["min"],
        cpu_fallback_us_max=cpu_stats["max"],
        speedup_vs_cpu=speedup,
        compute_us=compute_us,
        scatter_kernel_us=scatter_us,
        zeros_kernel_us=zeros_us,
        h2d_us=h2d_stats["median"],
        d2h_us=d2h_stats["median"],
        scalar_us=scalar_us,
        aicore_util_pct=aicore_util,
        zeros_util_pct=zeros_util,
        compute_share=compute_us / op_med,
        h2d_share=h2d_stats["median"] / op_med,
        d2h_share=d2h_stats["median"] / op_med,
        scalar_share=scalar_us / op_med,
        samples=op_samples,
        cpu_samples=cpu_samples,
        volatile=volatile,
        breakdown_method=method,
        notes=(
            "compute=device kernel time (scatter_add+zeros) from torch_npu "
            "profiler op_statistic.csv (Avg); AICore利用率=scatter_add kernel "
            "AICore Ratio%; scalar=op_wall−compute (host dispatch+launch+sync); "
            "h2d=.to(npu) of inputs; d2h=.cpu() of output."
        ),
    )


# ----------------------------- reporting ----------------------------- #

def _record_to_dict(r: PerfRecord) -> Dict:
    return {
        "case_id": r.case_id,
        "level": r.level,
        "desc": r.desc,
        "shape": r.shape,
        "idtype": r.idtype,
        "rounds": r.rounds,
        "op_us": {
            "median": r.op_us_median,
            "mean": r.op_us_mean,
            "std": r.op_us_std,
            "min": r.op_us_min,
            "max": r.op_us_max,
            "p50": r.op_us_p50,
            "p90": r.op_us_p90,
            "cv": r.op_us_cv,
            "samples": r.samples,
        },
        "cpu_fallback_us": {
            "median": r.cpu_fallback_us_median,
            "min": r.cpu_fallback_us_min,
            "max": r.cpu_fallback_us_max,
            "samples": r.cpu_samples,
        },
        "speedup_vs_cpu": r.speedup_vs_cpu,
        "bottleneck_breakdown_us": {
            "compute": r.compute_us,
            "scatter_kernel": r.scatter_kernel_us,
            "zeros_kernel": r.zeros_kernel_us,
            "h2d": r.h2d_us,
            "d2h": r.d2h_us,
            "scalar": r.scalar_us,
        },
        "aicore_utilization_pct": {
            "scatter_add_kernel": r.aicore_util_pct,
            "zeros_kernel": r.zeros_util_pct,
        },
        "bottleneck_share_of_op": {
            "compute": r.compute_share,
            "h2d": r.h2d_share,
            "d2h": r.d2h_share,
            "scalar": r.scalar_share,
        },
        "volatile": r.volatile,
        "breakdown_method": r.breakdown_method,
        "notes": r.notes,
    }


def _write_report(records: List[PerfRecord], out_dir: str,
                  warmup: int, rounds: int, device: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = [_record_to_dict(r) for r in records]

    with open(os.path.join(out_dir, "scatter_add_perf.jsonl"), "w") as f:
        for rec in payload:
            f.write(json.dumps(rec, default=str) + "\n")
    with open(os.path.join(out_dir, "scatter_add_perf.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# scatter_add performance report",
        "",
        f"- 设备: NPU {device} (910B3)",
        f"- 采集口径: 预热 {warmup} 轮剔除; 每用例 {rounds} 轮稳态采集; "
        f"取值=中位数 (不报单次最优); 每次 op 前后 torch.npu.synchronize "
        f"排空流水; 设备 NPU{device} 与其他进程共享 (非完全空闲), 波动用例 "
        f"已做 ≥30 轮并以中位数稳态为准.",
        f"- dtype 覆盖: int32 / int64 × float32",
        f"- 瓶颈分解口径: compute=device kernel time (scatter_add + zeros, "
        f"torch_npu.profiler op_statistic.csv Avg Duration); "
        f"AICore利用率=scatter_add kernel AICore Ratio% (Level1+PipeUtilization); "
        f"scalar=op_wall−compute (host dispatch+launch+sync); "
        f"h2d=输入 .to(npu); d2h=输出 .cpu().",
        f"- 波动判定: cv (std/median) > 0.10 视为波动, 已记录多轮 min/max/cv.",
        "",
        "## 1. 原生 NPU vs CPU 回退 (D2H+cpu::ScatterAdd+H2D)",
        "",
        "| Case | IdType | N | dim | M | 原生 op 中位(us) | CPU回退 中位(us) | 加速比(CPU/原生) |",
        "|------|--------|---|-----|---|------------------|------------------|------------------|",
    ]
    for r in records:
        lines.append(
            f"| {r.case_id} | {r.idtype} | {r.shape['N']} | {r.shape['dim']} | "
            f"{r.shape['M']} | {r.op_us_median:.3f} | "
            f"{r.cpu_fallback_us_median:.3f} | {r.speedup_vs_cpu:.2f}x |"
        )

    lines += [
        "",
        "## 2. 瓶颈维度分解",
        "",
        "| Case | op 中位(us) | compute(us) | scatter_kernel(us) | zeros_kernel(us) | h2d(us) | d2h(us) | scalar(us) | compute% | h2d% | d2h% | scalar% | AICore利用率% | cv | 轮数 |",
        "|------|-----------|-------------|---------------------|------------------|---------|---------|------------|----------|------|------|---------|-------------|-----|------|",
    ]
    for r in records:
        lines.append(
            f"| {r.case_id} | {r.op_us_median:.3f} | {r.compute_us:.3f} | "
            f"{r.scatter_kernel_us:.3f} | {r.zeros_kernel_us:.3f} | "
            f"{r.h2d_us:.3f} | {r.d2h_us:.3f} | {r.scalar_us:.3f} | "
            f"{r.compute_share*100:.1f} | {r.h2d_share*100:.1f} | "
            f"{r.d2h_share*100:.1f} | {r.scalar_share*100:.1f} | "
            f"{r.aicore_util_pct:.1f} | {r.op_us_cv:.3f} | {r.rounds} |"
        )

    lines += [
        "",
        "## 3. 方差处理",
        "",
        "| Case | op 中位(us) | min(us) | max(us) | p90(us) | std(us) | cv | 波动? |",
        "|------|-----------|---------|---------|---------|---------|-----|-------|",
    ]
    for r in records:
        lines.append(
            f"| {r.case_id} | {r.op_us_median:.3f} | {r.op_us_min:.3f} | "
            f"{r.op_us_max:.3f} | {r.op_us_p90:.3f} | {r.op_us_std:.3f} | "
            f"{r.op_us_cv:.3f} | {'是' if r.volatile else '否'} |"
        )

    volatile = [r for r in records if r.volatile]
    lines += [
        "",
        "## 4. 瓶颈分解方法说明",
        "",
        "compute (设备计算耗时) = torch_npu.profiler op_statistic.csv 中 "
        "`kernel_scatter_add` 与 `ZerosLike` 两条设备 kernel 的 Avg Time(us) "
        "之和 (Level1+PipeUtilization, 3 轮记录取平均). 这是 op 在 AICore 上 "
        "真正的计算耗时, 非启发式估计. AICore利用率 = kernel_scatter_add 的 "
        "AICore Ratio% (PipeUtilization).",
        "",
        "scalar (host 标量开销) = op_wall_median − compute, 即 host 侧 dispatch "
        "(ATEN_ID_TYPE_SWITCH/ATEN_FLOAT_TYPE_SWITCH) + kernel launch + "
        "torch.npu.synchronize 的残差开销.",
        "",
        "h2d = feat_cpu.to(npu) + idx_cpu.to(npu) 独立采集 (输入搬入).",
        "d2h = out.cpu() 独立采集 (输出搬出).",
        "",
        "> share = 各维度耗时 / op_wall_median. h2d/d2h 为独立采集的参考耗时, "
        "> 其 share 可能 >100% (因 op 测量时输入已驻留, 不含 h2d). "
        "> compute+scalar share 之和约为 100%.",
    ]
    if volatile:
        lines += [
            "",
            "## 5. 波动用例",
            "",
            "以下用例 cv>0.10, 已做 ≥30 轮稳态采集, 以中位数为准:",
        ]
        for r in volatile:
            lines.append(
                f"- {r.case_id}: 中位 {r.op_us_median:.3f}us, "
                f"min/max {r.op_us_min:.3f}/{r.op_us_max:.3f}, cv={r.op_us_cv:.3f}"
            )

    with open(os.path.join(out_dir, "scatter_add_perf.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="scatter_add NPU performance collection"
    )
    parser.add_argument("--cases", default="",
                        help="逗号分隔 case_id; 留空则用默认 perf 集")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30,
                        help="每用例采集轮数 (>=30)")
    parser.add_argument("--device", type=int, default=4)
    args = parser.parse_args()

    if not _npu_available():
        print("NPU not available; skipping performance collection.")
        return

    if args.cases:
        case_ids = [s.strip() for s in args.cases.split(",") if s.strip()]
    else:
        case_ids = [cid for cid, _ in PERF_CASES]

    dev = _get_device(args.device)
    os.makedirs(PROF_SCRATCH_DIR, exist_ok=True)

    records: List[PerfRecord] = []
    for cid in case_ids:
        try:
            rec = collect_one(cid, dev, args.warmup, args.rounds, PROF_SCRATCH_DIR)
            records.append(rec)
            print(
                f"[{cid}] {rec.idtype} N={rec.shape['N']} dim={rec.shape['dim']} "
                f"M={rec.shape['M']} | native={rec.op_us_median:.3f}us "
                f"cv={rec.op_us_cv:.3f} | cpu_fb={rec.cpu_fallback_us_median:.3f}us "
                f"speedup={rec.speedup_vs_cpu:.2f}x | "
                f"compute={rec.compute_us:.3f}us({rec.compute_share*100:.0f}%) "
                f"scatter_k={rec.scatter_kernel_us:.3f} "
                f"h2d={rec.h2d_us:.3f} d2h={rec.d2h_us:.3f} "
                f"scalar={rec.scalar_us:.3f}us({rec.scalar_share*100:.0f}%)"
            )
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[{cid}] collection FAILED: {e}")

    _write_report(records, PERF_RESULTS_DIR, args.warmup, args.rounds, args.device)
    print(
        f"\nWrote {len(records)} records to {PERF_RESULTS_DIR}/"
        f"scatter_add_perf.{{json,jsonl,md}}"
    )

    volatile = [r for r in records if r.volatile]
    if volatile:
        print(f"\nVolatile cases (cv>0.10): {[r.case_id for r in volatile]}")
        print("  -> 多轮采集已记录中位数/min/max/cv.")


if __name__ == "__main__":
    main()
