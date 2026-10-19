from __future__ import annotations

import argparse
import time
from typing import Any, List, Tuple

import dgl
import numpy as np
import torch
from dgl.data import CoauthorCSDataset, CoraGraphDataset, PubmedGraphDataset

try:
    import torch_npu
except ImportError:
    torch_npu = None


Reducer = str

SUPPORTED_REDUCERS: Tuple[Reducer, ...] = (
    "sum",
    "mean",
    "max",
    "min",
)


def get_npu_device(device_id: int) -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("未检测到可用的 NPU 设备。")
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)


def build_benchmark_inputs(graph: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    graph = dgl.add_self_loop(graph)
    if "feat" not in graph.ndata:
        raise KeyError(f"图缺少节点特征 'feat'，当前可用特征键: {list(graph.ndata.keys())}")

    seglen = graph.in_degrees().to(torch.int64).cpu()
    num_edges = graph.num_edges()
    feat = graph.ndata["feat"]
    feat_dim = feat.shape[1] if feat.ndim > 1 else 1
    value = torch.randn((num_edges, feat_dim), dtype=torch.float32)
    return seglen, value


def run_segment_reduce(
    seglen: torch.Tensor,
    value: torch.Tensor,
    reducer: Reducer,
    device: torch.device,
) -> torch.Tensor:
    seglen = seglen.to(device)
    value = value.to(device)
    output = dgl.ops.segment_reduce(seglen, value, reducer=reducer)
    synchronize(device)
    return output.detach().cpu()


def benchmark_segment_reduce(
    seglen: torch.Tensor,
    value: torch.Tensor,
    reducer: Reducer,
    device: torch.device,
    runs_per_measure: int,
    measurements: int,
    warmup: int,
) -> Tuple[torch.Tensor, float, float]:
    seglen = seglen.to(device)
    value = value.to(device)

    for _ in range(warmup):
        _ = dgl.ops.segment_reduce(seglen, value, reducer=reducer)
    synchronize(device)

    times_us: List[float] = []
    for _ in range(measurements):
        start = time.perf_counter()
        for _ in range(runs_per_measure):
            output = dgl.ops.segment_reduce(seglen, value, reducer=reducer)
        synchronize(device)
        end = time.perf_counter()
        times_us.append((end - start) * 1_000_000 / runs_per_measure)

    return output.detach().cpu(), float(np.median(times_us)), float(np.std(times_us))


def compare_outputs(cpu_output: torch.Tensor, npu_output: torch.Tensor) -> Tuple[float, float, int]:
    diff = torch.abs(cpu_output - npu_output)
    max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
    mean_abs_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0
    mismatch_count = int(torch.count_nonzero(diff > 1e-4).item())
    return max_abs_diff, mean_abs_diff, mismatch_count


def run_segment_reduce_benchmark(
    runs_per_measure: int,
    measurements: int,
    warmup: int,
    device_id: int,
) -> None:
    npu_device = get_npu_device(device_id)
    datasets = [
        ("cora", CoraGraphDataset(verbose=False)),
        # ("pubmed", PubmedGraphDataset(verbose=False)),
        # ("coauthor_cs", CoauthorCSDataset(verbose=False)),
    ]

    if torch_npu is None:
        print("Warning: torch_npu 未成功导入，将依赖当前环境中的 torch.npu 接口。")

    print(f"Using NPU device: {npu_device}")
    print(f"Reducers under test: {', '.join(SUPPORTED_REDUCERS)}")
    for ds_name, ds in datasets:
        sample = ds[0]
        graph = sample[0] if isinstance(sample, tuple) else sample
        seglen_cpu, value_cpu = build_benchmark_inputs(graph)

        print(f"\n=== Dataset: {ds_name} ===")
        print(
            f"num_segments: {seglen_cpu.shape[0]}, "
            f"num_items: {value_cpu.shape[0]}, feat_dim: {value_cpu.shape[1]}, "
            f"feature_bytes: {value_cpu.shape[1] * value_cpu.element_size()}"
        )

        for reducer in SUPPORTED_REDUCERS:
            cpu_output = run_segment_reduce(
                seglen=seglen_cpu,
                value=value_cpu,
                reducer=reducer,
                device=torch.device("cpu"),
            )
            npu_output, npu_median_us, npu_std_us = benchmark_segment_reduce(
                seglen=seglen_cpu,
                value=value_cpu,
                reducer=reducer,
                device=npu_device,
                runs_per_measure=runs_per_measure,
                measurements=measurements,
                warmup=warmup,
            )
            max_abs_diff, mean_abs_diff, mismatch_count = compare_outputs(cpu_output, npu_output)

            print(f"\n--- Reducer: {reducer} ---")
            print(f"NPU segment_reduce_{reducer}: {npu_median_us:.3f} us (StdDev: {npu_std_us:.3f})")
            print(
                f"CPU vs NPU: max_abs_diff={max_abs_diff:.8e}, "
                f"mean_abs_diff={mean_abs_diff:.8e}, "
                f"mismatch_count(diff>1e-4)={mismatch_count}/{cpu_output.numel()}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark segment_reduce on NPU")
    parser.add_argument("--runs", type=int, default=100, help="单次测量内部的连续执行次数")
    parser.add_argument("--measurements", type=int, default=50, help="总共进行多少次独立测量")
    parser.add_argument("--warmup", type=int, default=100, help="预热执行次数")
    parser.add_argument("--device-id", type=int, default=7, help="NPU 设备编号")
    args = parser.parse_args()

    run_segment_reduce_benchmark(
        runs_per_measure=args.runs,
        measurements=args.measurements,
        warmup=args.warmup,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
