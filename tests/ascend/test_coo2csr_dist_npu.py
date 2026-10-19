"""
Distributed training tests for COOToCSR on Ascend NPU.

Covers:
  - 6 partition schemes (range, uneven, unsorted, empty, single-node, large-sparse)
    via per-worker random COO generation
  - 3 single-process shape edge cases (empty partition, single row, zero nnz)
  - 6 end-to-end tests starting from a global graph, partitioning, COO→CSR,
    then verifying indptr, indices, degrees, and edge-list reconstruction

Usage:
  pytest tests/ascend/test_coo2csr_dist_npu.py -v
  python tests/ascend/test_coo2csr_dist_npu.py
"""

import multiprocessing as mp
import os
import sys
import time

import torch
import dgl
from dgl.sparse import from_coo
import numpy as np


# ─── Shared helpers ────────────────────────────────────────────────


def _check(rank, name, indptr_npu, indices_npu, indptr_cpu, indices_cpu):
    ok = True
    if not torch.allclose(indptr_npu.cpu(), indptr_cpu):
        print(f"  [Worker {rank}] FAIL [{name}] indptr mismatch", flush=True)
        ok = False
    if not torch.allclose(indices_npu.cpu(), indices_cpu):
        print(f"  [Worker {rank}] FAIL [{name}] indices mismatch", flush=True)
        ok = False
    return ok


def _warmup_npu():
    if torch.npu.is_available():
        _ = torch.zeros(1, device=torch.device("npu:0"))


# ─── Worker functions: partition schemes ─────────────────────────


def worker_range_partition(rank, world_size, kwargs):
    """Each worker owns a contiguous row range (RangePartitionBook)."""
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    device = torch.device(f"npu:{device_id}")
    cpu = torch.device("cpu")

    global_num_rows = kwargs["global_num_rows"]
    num_cols = kwargs["num_cols"]
    nnz_per_part = kwargs["nnz_per_part"]
    dtype = kwargs["dtype"]
    seed = kwargs.get("seed", 42)
    torch.manual_seed(seed + rank)

    rows_per_part = max(1, (global_num_rows + world_size - 1) // world_size)
    row_start = rank * rows_per_part
    row_end = min((rank + 1) * rows_per_part, global_num_rows)
    local_num_rows = row_end - row_start

    if local_num_rows > 0 and nnz_per_part > 0:
        global_rows = torch.randint(
            row_start, max(row_start + 1, row_end), (nnz_per_part,), dtype=dtype
        )
        global_rows = global_rows.clamp(row_start, row_end - 1)
        cols = torch.randint(0, num_cols, (nnz_per_part,), dtype=dtype)
        vals = torch.randn(nnz_per_part)
    else:
        global_rows = torch.empty(0, dtype=dtype)
        cols = torch.empty(0, dtype=dtype)
        vals = torch.empty(0)

    local_rows = global_rows - row_start

    if local_num_rows > 0 and len(local_rows) > 0:
        first_row_edges = (local_rows == 0).sum()
        if first_row_edges == 0 and nnz_per_part >= local_num_rows:
            local_rows[:local_num_rows] = torch.arange(
                0, min(local_num_rows, nnz_per_part), dtype=dtype
            )
            local_rows = local_rows[torch.randperm(len(local_rows))]
            cols = torch.randint(0, num_cols, (nnz_per_part,), dtype=dtype)
            vals = torch.randn(nnz_per_part)

    mat_npu = from_coo(
        local_rows.to(device), cols.to(device), vals.to(device),
        (local_num_rows, num_cols)
    ) if len(local_rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=device),
        torch.empty(0, dtype=dtype, device=device),
        shape=(local_num_rows, num_cols)
    )

    mat_cpu = from_coo(
        local_rows.to(cpu), cols.to(cpu), vals.to(cpu),
        (local_num_rows, num_cols)
    ) if len(local_rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=cpu),
        torch.empty(0, dtype=dtype, device=cpu),
        shape=(local_num_rows, num_cols)
    )

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert _check(rank, "range_partition", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


def worker_uneven_partition(rank, world_size, kwargs):
    """Partitions have very different sizes (imbalanced)."""
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    device = torch.device(f"npu:{device_id}")
    cpu = torch.device("cpu")

    dtype = kwargs["dtype"]
    seed = kwargs.get("seed", 42)
    torch.manual_seed(seed + rank)

    local_num_rows = kwargs["local_num_rows"][rank]
    num_cols = kwargs["num_cols"]
    nnz = kwargs["nnz_per_part_actual"][rank]

    if local_num_rows > 0 and nnz > 0:
        rows = torch.randint(0, local_num_rows, (nnz,), dtype=dtype)
        cols = torch.randint(0, num_cols, (nnz,), dtype=dtype)
        vals = torch.randn(nnz)
        if nnz >= local_num_rows:
            rows[:local_num_rows] = torch.arange(0, local_num_rows, dtype=dtype)
            rows = rows[torch.randperm(nnz)]
    else:
        rows = torch.empty(0, dtype=dtype)
        cols = torch.empty(0, dtype=dtype)
        vals = torch.empty(0)

    mat_npu = from_coo(
        rows.to(device), cols.to(device), vals.to(device),
        (local_num_rows, num_cols)
    ) if len(rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=device),
        torch.empty(0, dtype=dtype, device=device),
        shape=(local_num_rows, num_cols)
    )

    mat_cpu = from_coo(
        rows.to(cpu), cols.to(cpu), vals.to(cpu),
        (local_num_rows, num_cols)
    ) if len(rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=cpu),
        torch.empty(0, dtype=dtype, device=cpu),
        shape=(local_num_rows, num_cols)
    )

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert _check(rank, "uneven", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


def worker_unsorted_partition(rank, world_size, kwargs):
    """Partition receives unsorted COO (simulating unsorted global graph)."""
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    device = torch.device(f"npu:{device_id}")
    cpu = torch.device("cpu")

    dtype = kwargs["dtype"]
    num_rows = kwargs["num_rows"]
    num_cols = kwargs["num_cols"]
    nnz = kwargs["nnz"]
    seed = kwargs.get("seed", 42)
    torch.manual_seed(seed + rank * 100)

    rows = torch.randint(0, num_rows, (nnz,), dtype=dtype)
    cols = torch.randint(0, num_cols, (nnz,), dtype=dtype)
    vals = torch.randn(nnz)

    perm = torch.randperm(nnz)
    rows = rows[perm]
    cols = cols[perm]
    vals = vals[perm]

    if nnz >= num_rows:
        rows[:num_rows] = torch.arange(0, num_rows, dtype=dtype)
        rows = rows[torch.randperm(nnz)]

    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (num_rows, num_cols))
    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), vals.to(cpu), (num_rows, num_cols))

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert _check(rank, "unsorted", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


def worker_empty_partition(rank, world_size, kwargs):
    """Some partitions have no edges, some have edges."""
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    device = torch.device(f"npu:{device_id}")
    cpu = torch.device("cpu")

    dtype = kwargs["dtype"]
    num_rows = kwargs["num_rows"]
    num_cols = kwargs["num_cols"]
    seed = kwargs.get("seed", 42)
    torch.manual_seed(seed + rank)

    if rank % 2 == 0:
        nnz = kwargs["nnz"]
        rows = torch.randint(0, num_rows, (nnz,), dtype=dtype)
        cols = torch.randint(0, num_cols, (nnz,), dtype=dtype)
        vals = torch.randn(nnz)
        if nnz >= num_rows:
            rows[:num_rows] = torch.arange(0, num_rows, dtype=dtype)
            rows = rows[torch.randperm(nnz)]
    else:
        rows = torch.empty(0, dtype=dtype)
        cols = torch.empty(0, dtype=dtype)
        vals = torch.empty(0)

    mat_npu = from_coo(
        rows.to(device), cols.to(device), vals.to(device), (num_rows, num_cols)
    ) if len(rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=device),
        torch.empty(0, dtype=dtype, device=device),
        shape=(num_rows, num_cols)
    )

    mat_cpu = from_coo(
        rows.to(cpu), cols.to(cpu), vals.to(cpu), (num_rows, num_cols)
    ) if len(rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=cpu),
        torch.empty(0, dtype=dtype, device=cpu),
        shape=(num_rows, num_cols)
    )

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert _check(rank, "empty", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


def worker_single_node_partitions(rank, world_size, kwargs):
    """Each partition = single node with its adjacency."""
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    device = torch.device(f"npu:{device_id}")
    cpu = torch.device("cpu")

    dtype = kwargs["dtype"]
    num_nodes = kwargs["num_nodes"]
    num_cols = kwargs["num_cols"]
    seed = kwargs.get("seed", 42)
    torch.manual_seed(seed + rank)

    nodes_this_part = num_nodes // world_size
    start = rank * nodes_this_part
    end = num_nodes if rank == world_size - 1 else (rank + 1) * nodes_this_part
    local_num_rows = end - start

    if local_num_rows == 0:
        rows = torch.empty(0, dtype=dtype)
        cols = torch.empty(0, dtype=dtype)
        vals = torch.empty(0)
    else:
        nnz = kwargs.get("nnz_per_row", 3) * local_num_rows
        rows = torch.randint(0, local_num_rows, (nnz,), dtype=dtype)
        cols = torch.randint(0, num_cols, (nnz,), dtype=dtype)
        vals = torch.randn(nnz)

    mat_npu = from_coo(
        rows.to(device), cols.to(device), vals.to(device), (local_num_rows, num_cols)
    ) if len(rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=device),
        torch.empty(0, dtype=dtype, device=device),
        shape=(local_num_rows, num_cols)
    )

    mat_cpu = from_coo(
        rows.to(cpu), cols.to(cpu), vals.to(cpu), (local_num_rows, num_cols)
    ) if len(rows) > 0 else from_coo(
        torch.empty(0, dtype=dtype, device=cpu),
        torch.empty(0, dtype=dtype, device=cpu),
        shape=(local_num_rows, num_cols)
    )

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert _check(rank, "single_node_part", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


def worker_large_sparse_partition(rank, world_size, kwargs):
    """Large matrix with very sparse rows (typical in real distributed training)."""
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    device = torch.device(f"npu:{device_id}")
    cpu = torch.device("cpu")

    dtype = kwargs["dtype"]
    num_rows = kwargs["num_rows"]
    num_cols = kwargs["num_cols"]
    nnz = kwargs["nnz"]
    seed = kwargs.get("seed", 42)
    torch.manual_seed(seed + rank * 1000)

    row_weights = torch.exp(-torch.linspace(0, 5, num_rows))
    row_weights = row_weights / row_weights.sum()
    rows = torch.multinomial(row_weights, nnz, replacement=True).to(dtype)
    cols = torch.randint(0, num_cols, (nnz,), dtype=dtype)
    vals = torch.randn(nnz)

    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device), (num_rows, num_cols))
    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), vals.to(cpu), (num_rows, num_cols))

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert _check(rank, "large_sparse", indptr_npu, indices_npu, indptr_cpu, indices_cpu)


# ─── Worker function: end-to-end (global graph → partition) ──────


def _range_partition_edges(
    global_rows, global_cols, global_num_rows, num_parts, part_id
):
    """Assign each edge to partition based on its destination row."""
    rows_per_part = (global_num_rows + num_parts - 1) // num_parts
    row_start = part_id * rows_per_part
    row_end = min((part_id + 1) * rows_per_part, global_num_rows)

    mask = (global_rows >= row_start) & (global_rows < row_end)
    local_rows = global_rows[mask] - row_start
    local_cols = global_cols[mask]
    return local_rows, local_cols, row_start, row_end


def worker_e2e(rank, world_size, kwargs):
    """End-to-end worker: partition global graph, convert COO→CSR, verify all properties."""
    device_id = rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    device = torch.device(f"npu:{device_id}")
    cpu = torch.device("cpu")

    global_rows = kwargs["global_rows"]
    global_cols = kwargs["global_cols"]
    global_num_rows = kwargs["global_num_rows"]
    global_num_cols = kwargs["global_num_cols"]
    dtype = kwargs["dtype"]
    name = kwargs.get("name", "")

    local_rows, local_cols, row_start, row_end = _range_partition_edges(
        global_rows, global_cols, global_num_rows, world_size, rank
    )
    local_num_rows = row_end - row_start

    total_edges = len(local_rows)
    vals = torch.ones(total_edges)

    if total_edges > 0:
        mat_npu = from_coo(
            local_rows.to(device), local_cols.to(device),
            vals.to(device), (local_num_rows, global_num_cols)
        )
    else:
        mat_npu = from_coo(
            torch.empty(0, dtype=dtype, device=device),
            torch.empty(0, dtype=dtype, device=device),
            shape=(local_num_rows, global_num_cols)
        )

    indptr_npu, indices_npu, vi_npu = mat_npu.csr()
    indptr_npu_cpu = indptr_npu.cpu()
    indices_npu_cpu = indices_npu.cpu()
    degrees_npu = indptr_npu_cpu[1:] - indptr_npu_cpu[:-1]

    if total_edges > 0:
        mat_cpu = from_coo(
            local_rows.to(cpu), local_cols.to(cpu),
            vals.to(cpu), (local_num_rows, global_num_cols)
        )
    else:
        mat_cpu = from_coo(
            torch.empty(0, dtype=dtype, device=cpu),
            torch.empty(0, dtype=dtype, device=cpu),
            shape=(local_num_rows, global_num_cols)
        )

    indptr_cpu, indices_cpu, vi_cpu = mat_cpu.csr()
    degrees_cpu = indptr_cpu[1:] - indptr_cpu[:-1]

    ok = True

    if not torch.allclose(indptr_npu_cpu, indptr_cpu):
        print(f"  [Worker {rank}] FAIL [{name}] indptr mismatch", flush=True)
        ok = False

    if not torch.allclose(indices_npu_cpu, indices_cpu):
        print(f"  [Worker {rank}] FAIL [{name}] indices mismatch", flush=True)
        ok = False

    expected_degrees = torch.zeros(local_num_rows, dtype=dtype)
    for r in local_rows:
        expected_degrees[r] += 1
    if not torch.allclose(degrees_npu, expected_degrees):
        print(f"  [Worker {rank}] FAIL [{name}] degree mismatch", flush=True)
        ok = False

    if total_edges > 0:
        edge_src = []
        edge_dst = []
        for r in range(local_num_rows):
            s = indptr_cpu[r].item()
            e = indptr_cpu[r + 1].item()
            for j in range(s, e):
                edge_src.append(r)
                edge_dst.append(indices_cpu[j].item())

        if len(edge_src) != total_edges:
            print(f"  [Worker {rank}] FAIL [{name}] edge count mismatch: "
                  f"{len(edge_src)} vs {total_edges}", flush=True)
            ok = False
        else:
            recon_src = torch.tensor(edge_src, dtype=dtype)
            recon_dst = torch.tensor(edge_dst, dtype=dtype)

            global_recon_src = recon_src + row_start
            orig_mask = (global_rows >= row_start) & (global_rows < row_end)
            orig_src = global_rows[orig_mask]
            orig_dst = global_cols[orig_mask]

            orig_combined = orig_src * (global_num_cols + 1) + orig_dst
            recon_combined = global_recon_src * (global_num_cols + 1) + recon_dst
            if not torch.allclose(orig_combined.sort().values, recon_combined.sort().values):
                print(f"  [Worker {rank}] FAIL [{name}] edge list mismatch", flush=True)
                ok = False

    assert ok, f"Worker {rank} [{name}] failed"


# ─── Test runners ─────────────────────────────────────────────────


def _run_worker_test(name, worker_fn, world_size, kwargs):
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=worker_fn, args=(rank, world_size, kwargs))
        p.start()
        processes.append(p)

    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"  [FAIL] {name}: worker exited with code {p.exitcode}", flush=True)
            failed = True

    if failed:
        assert False, f"Test '{name}' failed: one or more workers exited with non-zero code"
    print(f"  PASS [{name}]", flush=True)


def _run_e2e_test(name, worker_fn, world_size, kwargs):
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=worker_fn, args=(rank, world_size, kwargs))
        p.start()
        processes.append(p)

    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"  FAIL [{name}]: worker exited with code {p.exitcode}", flush=True)
            failed = True

    if not failed:
        print(f"  PASS [{name}]", flush=True)
    return not failed


def _make_graph(num_rows, num_cols, nnz, dtype, seed, col_max=None):
    torch.manual_seed(seed)
    rows = torch.randint(0, num_rows, (nnz,), dtype=dtype)
    col_max = col_max or num_cols
    cols = torch.randint(0, col_max, (nnz,), dtype=dtype)
    return rows, cols


# ─── Single-process shape tests ───────────────────────────────────


def _check_single(name, rows, cols, dtype, device, cpu):
    vals = torch.randn(len(rows))
    mat_npu = from_coo(rows.to(device), cols.to(device), vals.to(device),
                       (rows.max().item() + 1 if len(rows) > 0 else 1, cols.max().item() + 1 if len(cols) > 0 else 1))
    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), vals.to(cpu),
                       (rows.max().item() + 1 if len(rows) > 0 else 1, cols.max().item() + 1 if len(cols) > 0 else 1))
    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()
    ok = True
    if not torch.allclose(indptr_npu.cpu(), indptr_cpu):
        print(f"  FAIL [{name}] indptr mismatch", flush=True)
        ok = False
    if not torch.allclose(indices_npu.cpu(), indices_cpu):
        print(f"  FAIL [{name}] indices mismatch", flush=True)
        ok = False
    if ok:
        print(f"  PASS [{name}]", flush=True)
    assert ok, f"Test '{name}' failed"


def test_dist_coo_to_csr_empty_partition_shape():
    """Batch of workers: partition with 0 rows, 0 nnz."""
    if not torch.npu.is_available():
        return
    device = torch.device("npu:0")
    cpu = torch.device("cpu")

    rows = torch.empty(0, dtype=torch.int64)
    cols = torch.empty(0, dtype=torch.int64)
    try:
        mat_npu = from_coo(rows.to(device), cols.to(device), shape=(0, 5))
        indptr_npu, indices_npu, _ = mat_npu.csr()
    except Exception as e:
        print(f"  NPU path FAILED: {e}", flush=True)
        raise
    try:
        mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), shape=(0, 5))
        indptr_cpu, indices_cpu, _ = mat_cpu.csr()
    except Exception as e:
        print(f"  CPU path FAILED: {e}", flush=True)
        raise
    assert len(indptr_npu) == 1
    assert len(indptr_cpu) == 1
    assert int(indptr_npu.cpu()[0].item()) == 0
    assert int(indptr_cpu[0].item()) == 0
    print("  PASS [empty_partition_shape]", flush=True)


def test_dist_coo_to_csr_single_row_partition():
    """Partition with exactly 1 row."""
    if not torch.npu.is_available():
        return
    device = torch.device("npu:0")
    cpu = torch.device("cpu")

    rows = torch.zeros(10, dtype=torch.int64)
    cols = torch.randint(0, 10, (10,), dtype=torch.int64)

    try:
        mat_npu = from_coo(rows.to(device), cols.to(device), shape=(1, 10))
        indptr_npu, indices_npu, _ = mat_npu.csr()
    except Exception as e:
        print(f"  NPU path FAILED: {e}", flush=True)
        raise
    try:
        mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), shape=(1, 10))
        indptr_cpu, indices_cpu, _ = mat_cpu.csr()
    except Exception as e:
        print(f"  CPU path FAILED: {e}", flush=True)
        raise

    assert torch.allclose(indptr_npu.cpu(), indptr_cpu)
    assert torch.allclose(indices_npu.cpu(), indices_cpu)
    val0 = int(indptr_npu.cpu()[0].item())
    val1 = int(indptr_npu.cpu()[1].item())
    assert val0 == 0
    assert val1 == 10
    print("  PASS [single_row_partition]", flush=True)


def test_dist_coo_to_csr_zero_nnz_partition():
    """Partition with rows but zero non-zeros."""
    if not torch.npu.is_available():
        return
    device = torch.device("npu:0")
    cpu = torch.device("cpu")

    rows = torch.empty(0, dtype=torch.int64)
    cols = torch.empty(0, dtype=torch.int64)

    mat_npu = from_coo(rows.to(device), cols.to(device), shape=(5, 10))
    mat_cpu = from_coo(rows.to(cpu), cols.to(cpu), shape=(5, 10))
    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert torch.allclose(indptr_npu.cpu(), indptr_cpu)
    assert indptr_npu.shape == (6,)
    assert indptr_npu.sum().item() == 0
    print("  PASS [zero_nnz_partition]", flush=True)


# ─── Distributed partition scheme tests ──────────────────────────


def test_dist_coo_to_csr_range_partition():
    """Range partition: each worker owns contiguous row block."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    _run_worker_test("range_partition", worker_range_partition, world_size, {
        "global_num_rows": 100,
        "num_cols": 50,
        "nnz_per_part": 500,
        "dtype": torch.int64,
        "seed": 42,
    })


def test_dist_coo_to_csr_range_partition_int32():
    """Range partition with int32 dtype."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    _run_worker_test("range_partition_int32", worker_range_partition, world_size, {
        "global_num_rows": 50,
        "num_cols": 30,
        "nnz_per_part": 200,
        "dtype": torch.int32,
        "seed": 137,
    })


def test_dist_coo_to_csr_uneven_partitions():
    """Imbalanced partitions: some have many rows, others few."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    local_rows = [100, 1, 200, 5][:world_size]
    nnz_actual = [500, 0, 1000, 10][:world_size]
    _run_worker_test("uneven", worker_uneven_partition, world_size, {
        "local_num_rows": local_rows,
        "num_cols": 50,
        "nnz_per_part_actual": nnz_actual,
        "dtype": torch.int64,
        "seed": 73,
    })


def test_dist_coo_to_csr_unsorted():
    """Unsorted COO in each partition."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    _run_worker_test("unsorted", worker_unsorted_partition, world_size, {
        "num_rows": 30,
        "num_cols": 20,
        "nnz": 200,
        "dtype": torch.int64,
        "seed": 2024,
    })


def test_dist_coo_to_csr_empty_partitions():
    """Half of partitions have no edges."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    _run_worker_test("empty", worker_empty_partition, world_size, {
        "num_rows": 20,
        "num_cols": 20,
        "nnz": 100,
        "dtype": torch.int64,
        "seed": 5,
    })


def test_dist_coo_to_csr_single_node_partitions():
    """Each partition holds a single node."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    _run_worker_test("single_node", worker_single_node_partitions, world_size, {
        "num_nodes": 20,
        "num_cols": 15,
        "nnz_per_row": 4,
        "dtype": torch.int64,
        "seed": 99,
    })


def test_dist_coo_to_csr_large_sparse():
    """Power-law row distribution (realistic large-scale distributed)."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    _run_worker_test("large_sparse", worker_large_sparse_partition, world_size, {
        "num_rows": 500,
        "num_cols": 200,
        "nnz": 10000,
        "dtype": torch.int64,
        "seed": 314,
    })


def test_dist_coo_to_csr_large_sparse_int32():
    """Power-law distribution with int32."""
    if not torch.npu.is_available():
        print("  SKIP: NPU not available", flush=True)
        return
    world_size = min(torch.npu.device_count(), 4)
    _run_worker_test("large_sparse_int32", worker_large_sparse_partition, world_size, {
        "num_rows": 500,
        "num_cols": 200,
        "nnz": 10000,
        "dtype": torch.int32,
        "seed": 2718,
    })


# ─── End-to-end tests ────────────────────────────────────────────


def test_e2e_range_partition():
    """Range partition: basic multi-worker distributed training flow."""
    if not torch.npu.is_available():
        return
    _warmup_npu()

    rows, cols = _make_graph(100, 50, 500, torch.int64, 42)
    world_size = min(torch.npu.device_count(), 4)

    ok = _run_e2e_test("range_partition", worker_e2e, world_size, {
        "global_rows": rows, "global_cols": cols,
        "global_num_rows": 100, "global_num_cols": 50,
        "dtype": torch.int64, "name": "e2e_range",
    })
    assert ok, "range_partition e2e test failed"


def test_e2e_int32():
    """Same flow with int32 dtype."""
    if not torch.npu.is_available():
        return
    _warmup_npu()

    rows, cols = _make_graph(80, 40, 300, torch.int32, 137)
    world_size = min(torch.npu.device_count(), 4)

    ok = _run_e2e_test("int32", worker_e2e, world_size, {
        "global_rows": rows, "global_cols": cols,
        "global_num_rows": 80, "global_num_cols": 40,
        "dtype": torch.int32, "name": "e2e_int32",
    })
    assert ok, "int32 e2e test failed"


def test_e2e_imbalanced():
    """Imbalanced graph: few nodes have most edges (power-law)."""
    if not torch.npu.is_available():
        return
    _warmup_npu()

    num_rows, num_cols, nnz = 50, 30, 1000
    torch.manual_seed(2024)
    row_weights = torch.exp(-torch.linspace(0, 4, num_rows))
    row_weights = row_weights / row_weights.sum()
    rows = torch.multinomial(row_weights, nnz, replacement=True).to(torch.int64)
    cols = torch.randint(0, num_cols, (nnz,), dtype=torch.int64)
    world_size = min(torch.npu.device_count(), 4)

    ok = _run_e2e_test("imbalanced", worker_e2e, world_size, {
        "global_rows": rows, "global_cols": cols,
        "global_num_rows": num_rows, "global_num_cols": num_cols,
        "dtype": torch.int64, "name": "e2e_imb",
    })
    assert ok, "imbalanced e2e test failed"


def test_e2e_empty_partitions():
    """Some partitions receive zero edges."""
    if not torch.npu.is_available():
        return
    _warmup_npu()

    num_rows, num_cols, nnz = 8, 5, 20
    rows, cols = _make_graph(num_rows, num_cols, nnz, torch.int64, 7)

    world_size = min(torch.npu.device_count(), 4)
    world_size = max(world_size, num_rows + 1)

    ok = _run_e2e_test("empty_parts", worker_e2e, world_size, {
        "global_rows": rows, "global_cols": cols,
        "global_num_rows": num_rows, "global_num_cols": num_cols,
        "dtype": torch.int64, "name": "e2e_empty",
    })
    assert ok, "empty_parts e2e test failed"


def test_e2e_single_node_parts():
    """Each partition is one node, i.e., extreme fine-grained."""
    if not torch.npu.is_available():
        return
    _warmup_npu()

    num_rows = 20
    num_cols = 15
    nnz = 100
    rows, cols = _make_graph(num_rows, num_cols, nnz, torch.int64, 31)
    world_size = num_rows

    ok = _run_e2e_test("single_node", worker_e2e, world_size, {
        "global_rows": rows, "global_cols": cols,
        "global_num_rows": num_rows, "global_num_cols": num_cols,
        "dtype": torch.int64, "name": "e2e_snode",
    })
    assert ok, "single_node e2e test failed"


def test_e2e_large_graph():
    """Larger graph simulating real distributed training scale."""
    if not torch.npu.is_available():
        return
    _warmup_npu()

    rows, cols = _make_graph(500, 200, 20000, torch.int64, 314159)
    world_size = min(torch.npu.device_count(), 4)

    ok = _run_e2e_test("large", worker_e2e, world_size, {
        "global_rows": rows, "global_cols": cols,
        "global_num_rows": 500, "global_num_cols": 200,
        "dtype": torch.int64, "name": "e2e_large",
    })
    assert ok, "large e2e test failed"


# ─── Main ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=== Distributed COOToCSR tests (merged) ===", flush=True)
    print(f"NPU available: {torch.npu.is_available()}", flush=True)
    if torch.npu.is_available():
        print(f"NPU count: {torch.npu.device_count()}", flush=True)

    tests = [
        # Single-process shape tests
        ("empty_partition_shape", test_dist_coo_to_csr_empty_partition_shape),
        ("single_row_partition", test_dist_coo_to_csr_single_row_partition),
        ("zero_nnz_partition", test_dist_coo_to_csr_zero_nnz_partition),
        # Distributed partition schemes
        ("range_partition", test_dist_coo_to_csr_range_partition),
        ("range_partition_int32", test_dist_coo_to_csr_range_partition_int32),
        ("uneven_partitions", test_dist_coo_to_csr_uneven_partitions),
        ("unsorted", test_dist_coo_to_csr_unsorted),
        ("empty_partitions", test_dist_coo_to_csr_empty_partitions),
        ("single_node_partitions", test_dist_coo_to_csr_single_node_partitions),
        ("large_sparse", test_dist_coo_to_csr_large_sparse),
        ("large_sparse_int32", test_dist_coo_to_csr_large_sparse_int32),
        # End-to-end tests
        ("e2e_range_partition", test_e2e_range_partition),
        ("e2e_int32", test_e2e_int32),
        ("e2e_imbalanced", test_e2e_imbalanced),
        ("e2e_empty_partitions", test_e2e_empty_partitions),
        ("e2e_single_node_parts", test_e2e_single_node_parts),
        ("e2e_large_graph", test_e2e_large_graph),
    ]

    if torch.npu.is_available():
        _ = torch.zeros(1, device=torch.device("npu:0"))
        print("  NPU device warmed up", flush=True)

    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  FAIL [{name}]: {e}", flush=True)
            traceback.print_exc()
            failures += 1

    total = len(tests)
    print(f"\n{'=' * 40}", flush=True)
    print(f"Results: {total - failures}/{total} passed, {failures} failed", flush=True)
    sys.exit(1 if failures > 0 else 0)

