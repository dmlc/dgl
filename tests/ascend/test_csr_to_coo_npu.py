"""CSRToCOO correctness and msprof performance tests on Ascend NPU.

Correctness:
    python tests/ascend/test_csr_to_coo_npu.py

The unittest class can also be collected by pytest when pytest is installed.

Profile every performance case in a separate msprof collection:
    python tests/ascend/test_csr_to_coo_npu.py --run-msprof \
        --msprof-output ./msprof_csr_to_coo

Profile one case directly:
    msprof --output=./msprof_csr_to_coo/typical_false \
        --ascendcl=on --runtime-api=on --task-time=l2 --ai-core=on \
        --aic-metrics=PipeUtilization \
        python tests/ascend/test_csr_to_coo_npu.py \
        --performance --perf-case typical_false

The public DGL calls intentionally exercise both C++ branches:

* graph.edges(form="all", order=None) on a CSR graph calls
  CSRToCOO(csr, false) and returns COO row/col/data.
* graph.adj_tensors("coo") on a fresh CSR graph calls
  CSRToCOO(csr, true) and returns COO row/col. COO data is empty because the
  input CSR data is consumed as the output order.

A fresh graph is used for every data_as_order=True conversion so that DGL's
cached COO representation cannot bypass CSRToCOO.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import dgl
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


CANONICAL_ETYPE = ("_U", "_E", "_V")
ID_DTYPES = (torch.int32, torch.int64)


def _npu_is_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except RuntimeError:
        return False


NPU_AVAILABLE = _npu_is_available()


@dataclass(frozen=True)
class CSRCase:
    """A deterministic CSR test specification stored directly in this file."""

    name: str
    num_rows: int
    num_cols: int
    nnz: int
    data_as_orders: Tuple[bool, ...]
    with_data: bool = True
    row_pattern: str = "balanced"
    col_pattern: str = "mixed"
    data_pattern: str = "identity"


# Functional matrix requested for CSRToCOO. Shapes are explicit and no external
# graph files or case configurations are used.
FUNCTIONAL_CASES = (
    CSRCase(
        "long_tail_row",
        1024,
        1024,
        8192,
        (False, True),
        row_pattern="long_tail",
    ),
    CSRCase("typical", 10000, 10000, 500000, (False, True)),
    CSRCase("stress", 100000, 100000, 2000000, (False, True)),
    CSRCase(
        "single_sparse_row",
        1,
        4096,
        1024,
        (False, True),
        col_pattern="single_row_sparse",
    ),
    CSRCase(
        "repeated_single_column",
        4096,
        1,
        8192,
        (False, True),
        col_pattern="single_column",
    ),
    CSRCase(
        "unsorted_within_row",
        4096,
        4096,
        65536,
        (False, True),
        col_pattern="descending",
    ),
    CSRCase(
        "reverse_edge_ids",
        4096,
        4096,
        65536,
        (False, True),
        data_pattern="reverse",
    ),
    CSRCase(
        "sampled_subgraph",
        12000,
        2000,
        60000,
        (False, True),
        row_pattern="sampled",
    ),
    CSRCase(
        "without_data",
        4096,
        4096,
        65536,
        (False,),
        with_data=False,
    ),
    CSRCase(
        "valid_reverse_permutation",
        4096,
        4096,
        65536,
        (True,),
        data_pattern="reverse",
    ),
    CSRCase(
        "valid_cyclic_permutation",
        4096,
        4096,
        65536,
        (True,),
        data_pattern="cyclic",
    ),
    CSRCase(
        "self_loops",
        4096,
        4096,
        4096,
        (False, True),
        col_pattern="self_loop",
    ),
)


# Performance matrix requested for msprof. All performance cases use int32 IDs.
PERFORMANCE_CASES = {
    "valid_reverse_true": ("valid_reverse_permutation", True),
    "typical_false": ("typical", False),
    "typical_true": ("typical", True),
    "sampled_subgraph_false": ("sampled_subgraph", False),
    "sampled_subgraph_true": ("sampled_subgraph", True),
}
CASES_BY_NAME = {case.name: case for case in FUNCTIONAL_CASES}


def _balanced_counts(num_rows: int, nnz: int) -> torch.Tensor:
    quotient, remainder = divmod(nnz, num_rows)
    counts = torch.full((num_rows,), quotient, dtype=torch.int64)
    counts[:remainder] += 1
    return counts


def _row_counts(case: CSRCase) -> torch.Tensor:
    if case.row_pattern == "balanced":
        return _balanced_counts(case.num_rows, case.nnz)

    if case.row_pattern == "long_tail":
        # Every row is non-empty and one hot row owns 75% of the remaining
        # entries, producing a deliberately skewed degree distribution.
        assert case.nnz >= case.num_rows
        counts = torch.ones(case.num_rows, dtype=torch.int64)
        remaining = case.nnz - case.num_rows
        hot = remaining * 3 // 4
        counts[case.num_rows // 2] += hot
        tail = remaining - hot
        quotient, remainder = divmod(tail, case.num_rows)
        counts += quotient
        counts[:remainder] += 1
        return counts

    if case.row_pattern == "sampled":
        # 80% of the edges are concentrated in the first 10% of source rows.
        hot_rows = max(1, case.num_rows // 10)
        hot_nnz = case.nnz * 4 // 5
        counts = torch.zeros(case.num_rows, dtype=torch.int64)
        counts[:hot_rows] = _balanced_counts(hot_rows, hot_nnz)
        cold_rows = case.num_rows - hot_rows
        if cold_rows:
            counts[hot_rows:] = _balanced_counts(
                cold_rows, case.nnz - hot_nnz
            )
        return counts

    raise ValueError(f"Unknown row pattern: {case.row_pattern}")


def _column_indices(case: CSRCase) -> torch.Tensor:
    positions = torch.arange(case.nnz, dtype=torch.int64)
    if case.col_pattern == "single_column":
        return torch.zeros(case.nnz, dtype=torch.int64)
    if case.col_pattern == "descending":
        return case.num_cols - 1 - positions.remainder(case.num_cols)
    if case.col_pattern == "single_row_sparse":
        return (positions * 3 + 1).remainder(case.num_cols)
    if case.col_pattern == "self_loop":
        assert case.num_rows == case.num_cols == case.nnz
        return positions
    if case.col_pattern == "mixed":
        # Deterministic non-monotonic columns without RNG state.
        return (positions * 104729 + 8191).remainder(case.num_cols)
    raise ValueError(f"Unknown column pattern: {case.col_pattern}")


def _edge_ids(case: CSRCase) -> Optional[torch.Tensor]:
    if not case.with_data:
        return None
    edge_ids = torch.arange(case.nnz, dtype=torch.int64)
    if case.data_pattern == "identity":
        return edge_ids
    if case.data_pattern == "reverse":
        return edge_ids.flip(0)
    if case.data_pattern == "cyclic":
        return edge_ids.roll(case.nnz // 3)
    raise ValueError(f"Unknown data pattern: {case.data_pattern}")


def _make_csr(
    case: CSRCase, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Materialize one deterministic, valid CSR input on CPU."""
    counts = _row_counts(case)
    assert counts.numel() == case.num_rows
    assert int(counts.sum()) == case.nnz

    indptr = torch.empty(case.num_rows + 1, dtype=torch.int64)
    indptr[0] = 0
    indptr[1:] = counts.cumsum(0)
    indices = _column_indices(case)
    data = _edge_ids(case)

    assert int(indptr[-1]) == case.nnz
    assert indices.shape == (case.nnz,)
    assert not indices.numel() or (
        int(indices.min()) >= 0 and int(indices.max()) < case.num_cols
    )

    return (
        indptr.to(dtype),
        indices.to(dtype),
        None if data is None else data.to(dtype),
    )


def _copy_csr_to(
    csr: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr, indices, data = csr
    empty_or_data = (
        torch.empty(0, dtype=dtype, device=device)
        if data is None
        else data.to(device)
    )
    return indptr.to(device), indices.to(device), empty_or_data


def _make_dgl_csr_graph(
    case: CSRCase,
    csr: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
) -> dgl.DGLGraph:
    indptr, indices, data = _copy_csr_to(csr, dtype, device)
    return dgl.heterograph(
        {CANONICAL_ETYPE: ("csr", (indptr, indices, data))},
        num_nodes_dict={"_U": case.num_rows, "_V": case.num_cols},
        idtype=dtype,
        device=device,
    )


def _dgl_csr_to_coo(
    graph: dgl.DGLGraph, data_as_order: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Invoke CSRToCOO through DGL's native graph API."""
    if data_as_order:
        row, col = graph.adj_tensors("coo", etype=CANONICAL_ETYPE)
        return row, col, None

    row, col, data = graph.edges(
        form="all", order=None, etype=CANONICAL_ETYPE
    )
    # DGL represents a missing data array as a zero-length ID array.
    return row, col, None if data.numel() == 0 else data


def _assert_same_tensor(
    label: str, npu_value: torch.Tensor, cpu_value: torch.Tensor
) -> None:
    actual = npu_value.cpu()
    assert actual.dtype == cpu_value.dtype, (
        f"{label}: dtype mismatch: NPU={actual.dtype}, CPU={cpu_value.dtype}"
    )
    assert actual.shape == cpu_value.shape, (
        f"{label}: shape mismatch: NPU={tuple(actual.shape)}, "
        f"CPU={tuple(cpu_value.shape)}"
    )
    if not torch.equal(actual, cpu_value):
        mismatch = torch.nonzero(actual != cpu_value, as_tuple=False)
        first = int(mismatch[0]) if mismatch.numel() else -1
        raise AssertionError(
            f"{label}: value mismatch at index {first}: "
            f"NPU={actual[first].item()}, CPU={cpu_value[first].item()}"
        )


def _assert_result(
    case: CSRCase,
    dtype: torch.dtype,
    data_as_order: bool,
    npu_result: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    cpu_result: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
) -> None:
    dtype_name = str(dtype).split(".")[-1]
    prefix = f"{case.name}/{dtype_name}/order={data_as_order}"
    npu_row, npu_col, npu_data = npu_result
    cpu_row, cpu_col, cpu_data = cpu_result

    assert npu_row.shape == (case.nnz,)
    assert npu_col.shape == (case.nnz,)
    assert npu_row.dtype == dtype
    assert npu_col.dtype == dtype
    _assert_same_tensor(f"{prefix}/row", npu_row, cpu_row)
    _assert_same_tensor(f"{prefix}/col", npu_col, cpu_col)

    expected_data = case.with_data and not data_as_order
    assert (npu_data is not None) == expected_data, (
        f"{prefix}/data: expected "
        f"{'an nnz-sized tensor' if expected_data else 'None'}"
    )
    assert (cpu_data is not None) == expected_data
    if expected_data:
        assert npu_data is not None and cpu_data is not None
        assert npu_data.shape == (case.nnz,)
        assert npu_data.dtype == dtype
        _assert_same_tensor(f"{prefix}/data", npu_data, cpu_data)


class TestCSRToCOONPU(unittest.TestCase):
    @unittest.skipUnless(
        NPU_AVAILABLE, "Ascend NPU is required for CSRToCOO tests"
    )
    def test_functional_matrix(self) -> None:
        """Compare native CPU and NPU CSRToCOO outputs exactly."""
        npu_device = torch.device("npu:0")
        torch.npu.set_device(npu_device)

        for case in FUNCTIONAL_CASES:
            for data_as_order in case.data_as_orders:
                for dtype in ID_DTYPES:
                    with self.subTest(
                        case=case.name,
                        data_as_order=data_as_order,
                        dtype=str(dtype).split(".")[-1],
                    ):
                        csr = _make_csr(case, dtype)
                        # Separate fresh graphs prevent a cached COO from
                        # bypassing conversion.
                        cpu_graph = _make_dgl_csr_graph(
                            case, csr, dtype, torch.device("cpu")
                        )
                        npu_graph = _make_dgl_csr_graph(
                            case, csr, dtype, npu_device
                        )
                        cpu_result = _dgl_csr_to_coo(
                            cpu_graph, data_as_order
                        )
                        npu_result = _dgl_csr_to_coo(
                            npu_graph, data_as_order
                        )
                        torch.npu.synchronize(npu_device)
                        _assert_result(
                            case,
                            dtype,
                            data_as_order,
                            npu_result=npu_result,
                            cpu_result=cpu_result,
                        )


def _run_performance_case(
    perf_name: str, warmup: int, repeat: int
) -> float:
    """Run one isolated int32 workload for collection by msprof."""
    if not NPU_AVAILABLE:
        raise RuntimeError("Ascend NPU is not available")
    if perf_name not in PERFORMANCE_CASES:
        raise ValueError(f"Unknown performance case: {perf_name}")

    case_name, data_as_order = PERFORMANCE_CASES[perf_name]
    case = CASES_BY_NAME[case_name]
    dtype = torch.int32
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    csr = _make_csr(case, dtype)

    # data_as_order=True caches COO, so independent graphs are required for
    # every profiled call. False can repeatedly use the same CSR graph.
    graph_count = warmup + repeat if data_as_order else 1
    graphs = [
        _make_dgl_csr_graph(case, csr, dtype, device)
        for _ in range(graph_count)
    ]

    print(
        f"PERF_CONFIG name={perf_name} shape=({case.num_rows},"
        f"{case.num_cols},{case.nnz}) dtype=int32 "
        f"data_as_order={data_as_order} warmup={warmup} repeat={repeat}",
        flush=True,
    )
    for iteration in range(warmup):
        graph = graphs[iteration] if data_as_order else graphs[0]
        _dgl_csr_to_coo(graph, data_as_order)
    torch.npu.synchronize(device)

    print(f"PERF_START {perf_name}", flush=True)
    start = time.perf_counter()
    for iteration in range(repeat):
        graph_index = warmup + iteration if data_as_order else 0
        _dgl_csr_to_coo(graphs[graph_index], data_as_order)
    torch.npu.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    average_ms = elapsed_ms / repeat
    print(
        f"PERF_END {perf_name} total_ms={elapsed_ms:.6f} "
        f"average_ms={average_ms:.6f}",
        flush=True,
    )
    return average_ms


def _run_msprof(
    output_dir: Path, perf_cases: Sequence[str], warmup: int, repeat: int
) -> None:
    msprof = shutil.which("msprof")
    if msprof is None:
        raise RuntimeError("msprof was not found in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for perf_name in perf_cases:
        case_output = output_dir / perf_name
        command = [
            msprof,
            f"--output={case_output}",
            "--ascendcl=on",
            "--runtime-api=on",
            "--task-time=l2",
            "--ai-core=on",
            "--aic-metrics=PipeUtilization",
            sys.executable,
            str(script),
            "--performance",
            "--perf-case",
            perf_name,
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--performance",
        action="store_true",
        help="Run NPU workloads, normally under msprof.",
    )
    mode.add_argument(
        "--run-msprof",
        action="store_true",
        help="Launch one msprof process per requested performance case.",
    )
    parser.add_argument(
        "--perf-case",
        choices=("all", *PERFORMANCE_CASES.keys()),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--msprof-output", type=Path, default=Path("./msprof_csr_to_coo")
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    return args


if __name__ == "__main__":
    cli_args = _parse_args()
    selected_cases = (
        tuple(PERFORMANCE_CASES)
        if cli_args.perf_case == "all"
        else (cli_args.perf_case,)
    )
    if not cli_args.performance and not cli_args.run_msprof:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            TestCSRToCOONPU
        )
        test_result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if test_result.wasSuccessful() else 1)
    if cli_args.run_msprof:
        _run_msprof(
            cli_args.msprof_output,
            selected_cases,
            warmup=cli_args.warmup,
            repeat=cli_args.repeat,
        )
    else:
        for selected_case in selected_cases:
            _run_performance_case(
                selected_case,
                warmup=cli_args.warmup,
                repeat=cli_args.repeat,
            )
