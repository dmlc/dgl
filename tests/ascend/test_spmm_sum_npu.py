from __future__ import annotations

import argparse
import time
from typing import Any, List, Tuple

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import CoraGraphDataset

try:
    import torch_npu
except ImportError:
    torch_npu = None


DatasetSpec = Tuple[str, Any]


def get_npu_device(device_id: int) -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("未检测到可用的 NPU 设备。")
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)


def pad_feature_dim_to_32bytes(features: torch.Tensor) -> torch.Tensor:
    """Pad feature dim (last dim) to 32-byte alignment for Ascend kernels."""
    if features.ndim == 1:
        features = features.unsqueeze(1)
    feat_dim = features.shape[1]
    dtype_bytes = features.element_size()
    aligned_feat_dim = ((feat_dim * dtype_bytes + 31) // 32) * 32 // dtype_bytes
    if aligned_feat_dim == feat_dim:
        return features
    return F.pad(features, (0, aligned_feat_dim - feat_dim), mode="constant", value=0)


def get_graph_from_dataset_sample(sample: Any) -> dgl.DGLGraph:
    return sample[0] if isinstance(sample, tuple) else sample


def build_benchmark_inputs(graph: dgl.DGLGraph) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    graph = dgl.add_self_loop(graph)
    graph = graph.int()
    graph = graph.formats(["csc", "csr"])
    graph.create_formats_()

    if "feat" not in graph.ndata:
        raise KeyError(f"图缺少节点特征 'feat'，当前可用特征键: {list(graph.ndata.keys())}")

    features = graph.ndata["feat"].detach().cpu().to(torch.float16)
    features = pad_feature_dim_to_32bytes(features).contiguous()
    return graph, features


def get_reduce_func(reduce: str) -> Any:
    if reduce == "sum":
        return fn.sum("m", "h")
    if reduce == "max":
        return fn.max("m", "h")
    if reduce == "min":
        return fn.min("m", "h")
    raise ValueError(f"Unsupported reduce: {reduce}")


def run_spmm(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    device: torch.device,
    reduce: str,
) -> torch.Tensor:
    graph = graph.to(device)
    features = features.to(device)
    reduce_func = get_reduce_func(reduce)
    with graph.local_scope():
        graph.ndata["h"] = features
        graph.update_all(fn.copy_u("h", "m"), reduce_func)
        synchronize(device)
        return graph.ndata["h"].detach().cpu()


def benchmark_spmm(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    device: torch.device,
    reduce: str,
    runs_per_measure: int,
    measurements: int,
    warmup: int,
) -> Tuple[torch.Tensor, float, float]:
    graph = graph.to(device)
    features = features.to(device)
    reduce_func = get_reduce_func(reduce)

    for _ in range(warmup):
        with graph.local_scope():
            graph.ndata["h"] = features
            graph.update_all(fn.copy_u("h", "m"), reduce_func)
    synchronize(device)

    times_us: List[float] = []
    output = None
    for _ in range(measurements):
        start = time.perf_counter()
        for _ in range(runs_per_measure):
            with graph.local_scope():
                graph.ndata["h"] = features
                graph.update_all(fn.copy_u("h", "m"), reduce_func)
                output = graph.ndata["h"]
        synchronize(device)
        end = time.perf_counter()
        times_us.append((end - start) * 1_000_000 / runs_per_measure)

    if output is None:
        raise RuntimeError("SpMM benchmark did not produce output.")
    return output.detach().cpu(), float(np.median(times_us)), float(np.std(times_us))


def compare_outputs(cpu_output: torch.Tensor, npu_output: torch.Tensor) -> Tuple[float, float, int]:
    cpu_output = cpu_output.float()
    npu_output = npu_output.float()
    diff = torch.abs(cpu_output - npu_output)
    max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
    mean_abs_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0
    mismatch_count = int(torch.count_nonzero(diff > 1e-2).item())
    return max_abs_diff, mean_abs_diff, mismatch_count


def run_spmm_benchmark(
    reduces: List[str],
    runs_per_measure: int,
    measurements: int,
    warmup: int,
    device_id: int,
) -> None:
    npu_device = get_npu_device(device_id)
    datasets: List[DatasetSpec] = [
        ("cora", CoraGraphDataset(verbose=False)),
        # ("pubmed", PubmedGraphDataset(verbose=False)),
        # ("coauthor_cs", CoauthorCSDataset(verbose=False)),
    ]

    if torch_npu is None:
        print("Warning: torch_npu 未成功导入，将依赖当前环境中的 torch.npu 接口。")

    print(f"Using NPU device: {npu_device}")
    operator_names = [f"update_all(copy_u, {reduce}) / SpMM {reduce}" for reduce in reduces]
    print(f"Operators under test: {', '.join(operator_names)}")

    for ds_name, ds in datasets:
        sample = ds[0]
        graph = get_graph_from_dataset_sample(sample)
        graph_cpu, features_cpu = build_benchmark_inputs(graph)

        print(f"\n=== Dataset: {ds_name} ===")
        print(
            f"num_nodes: {graph_cpu.num_nodes()}, "
            f"num_edges: {graph_cpu.num_edges()}, feat_dim: {features_cpu.shape[1]}, "
            f"feature_bytes: {features_cpu.shape[1] * features_cpu.element_size()}"
        )

        for reduce in reduces:
            cpu_output = run_spmm(
                graph=graph_cpu,
                features=features_cpu.float(),
                device=torch.device("cpu"),
                reduce=reduce,
            )
            npu_output, npu_median_us, npu_std_us = benchmark_spmm(
                graph=graph_cpu,
                features=features_cpu,
                device=npu_device,
                reduce=reduce,
                runs_per_measure=runs_per_measure,
                measurements=measurements,
                warmup=warmup,
            )
            max_abs_diff, mean_abs_diff, mismatch_count = compare_outputs(cpu_output, npu_output)

            print(f"\nNPU SpMM {reduce}: {npu_median_us:.3f} us (StdDev: {npu_std_us:.3f})")
            print(
                f"CPU vs NPU: max_abs_diff={max_abs_diff:.8e}, "
                f"mean_abs_diff={mean_abs_diff:.8e}, "
                f"mismatch_count(diff>1e-2)={mismatch_count}/{cpu_output.numel()}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SpMM sum/max/min on NPU")
    parser.add_argument("--runs", type=int, default=100, help="单次测量内部的连续执行次数")
    parser.add_argument("--measurements", type=int, default=50, help="总共进行多少次独立测量")
    parser.add_argument("--warmup", type=int, default=100, help="预热执行次数")
    parser.add_argument("--device-id", type=int, default=6, help="NPU 设备编号")
    parser.add_argument(
        "--reduces",
        nargs="+",
        default=["sum", "max", "min"],
        choices=["sum", "max", "min"],
        help="需要测试的 reduce 类型",
    )
    args = parser.parse_args()

    run_spmm_benchmark(
        reduces=args.reduces,
        runs_per_measure=args.runs,
        measurements=args.measurements,
        warmup=args.warmup,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
