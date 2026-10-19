"""
HCCL multi-card communication + COOToCSR integration test for Ascend NPU.

Tests the full distributed pipeline with real HCCL all-to-all communication:
  1. Each worker creates partitioned COO edges on its NPU
  2. Uses HCCL sparse_all_to_all_push to redistribute edges by destination partition
  3. Each worker converts its received COO to CSR on NPU (COOToCSR)
  4. Verifies CSR results against CPU reference
  5. Uses sparse_all_to_all_pull to verify feature gather operations

Partition schemes: remainder, range
Data types: int32, int64
Edge cases: empty partitions

Usage:
  torchrun --nproc_per_node=4 tests/ascend/test_coo2csr_hccl_npu.py
"""

import os

import sys
import torch
import torch.distributed as dist
from dgl.ascend.hccl import sparse_all_to_all_push, sparse_all_to_all_pull, AscendNDArrayPartitionWrapper
from dgl.sparse import from_coo


def _part_range_bounds(rank, world_size, num_nodes):
    chunk = (num_nodes + world_size - 1) // world_size
    row_start = rank * chunk
    row_end = min((rank + 1) * chunk, num_nodes)
    return row_start, row_end


def _compute_local_shape(num_nodes, world_size, rank, mode):
    if mode == "remainder":
        return num_nodes // world_size + (1 if rank < num_nodes % world_size else 0)
    elif mode == "range":
        start, end = _part_range_bounds(rank, world_size, num_nodes)
        return end - start


# ─── Worker functions ──────────────────────────────────────────────

def worker_push_range(rank, world_size, device, kwargs):
    cpu = torch.device("cpu")
    dtype = kwargs["dtype"]
    num_nodes = kwargs["num_nodes"]
    num_cols = kwargs["num_cols"]
    nnz = kwargs["nnz"]
    seed = kwargs.get("seed", 42) + rank
    torch.manual_seed(seed)

    row_start, row_end = _part_range_bounds(rank, world_size, num_nodes)
    local_rows_count = row_end - row_start

    global_rows = torch.randint(0, num_nodes, (nnz,), dtype=dtype, device=device)
    cols = torch.randint(0, num_cols, (nnz,), dtype=dtype, device=device)
    vals = torch.randn(nnz, device=device)

    packed = torch.stack([cols.float(), vals], dim=1)

    chunk = (num_nodes + world_size - 1) // world_size
    ranges = [min(i * chunk, num_nodes) for i in range(world_size + 1)]
    part_ranges = torch.tensor(ranges, dtype=torch.int64, device=device)
    part = AscendNDArrayPartitionWrapper(num_nodes, world_size, "range", part_ranges=part_ranges)

    recv_rows, recv_packed = sparse_all_to_all_push(global_rows, packed, part)

    if recv_rows.shape[0] == 0:
        local_rows = torch.empty(0, dtype=dtype, device=device)
        recv_cols = torch.empty(0, dtype=dtype, device=device)
        recv_vals = torch.empty(0, device=device)
    else:
        recv_cols = recv_packed[:, 0].long().to(dtype)
        recv_vals = recv_packed[:, 1]
        local_rows = recv_rows - row_start

    mat_npu = from_coo(local_rows, recv_cols, recv_vals, shape=(local_rows_count, num_cols))
    mat_cpu = from_coo(local_rows.cpu(), recv_cols.cpu(), recv_vals.cpu(), shape=(local_rows_count, num_cols))

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert torch.allclose(indptr_npu.cpu(), indptr_cpu), f"[Rank {rank}] indptr mismatch"
    assert torch.allclose(indices_npu.cpu(), indices_cpu), f"[Rank {rank}] indices mismatch"


def worker_push_remainder(rank, world_size, device, kwargs):
    cpu = torch.device("cpu")
    dtype = kwargs["dtype"]
    num_nodes = kwargs["num_nodes"]
    num_cols = kwargs["num_cols"]
    nnz = kwargs["nnz"]
    seed = kwargs.get("seed", 42) + rank
    torch.manual_seed(seed)

    local_rows_count = _compute_local_shape(num_nodes, world_size, rank, "remainder")

    global_rows = torch.randint(0, num_nodes, (nnz,), dtype=dtype, device=device)
    cols = torch.randint(0, num_cols, (nnz,), dtype=dtype, device=device)
    vals = torch.randn(nnz, device=device)

    packed = torch.stack([cols.float(), vals], dim=1)

    part = AscendNDArrayPartitionWrapper(num_nodes, world_size, "remainder")

    recv_rows, recv_packed = sparse_all_to_all_push(global_rows, packed, part)

    if recv_rows.shape[0] == 0:
        local_rows = torch.empty(0, dtype=dtype, device=device)
        recv_cols = torch.empty(0, dtype=dtype, device=device)
        recv_vals = torch.empty(0, device=device)
    else:
        recv_cols = recv_packed[:, 0].long().to(dtype)
        recv_vals = recv_packed[:, 1]
        local_rows = recv_rows // world_size

    mat_npu = from_coo(local_rows, recv_cols, recv_vals, shape=(local_rows_count, num_cols))
    mat_cpu = from_coo(local_rows.cpu(), recv_cols.cpu(), recv_vals.cpu(), shape=(local_rows_count, num_cols))

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert torch.allclose(indptr_npu.cpu(), indptr_cpu), f"[Rank {rank}] indptr mismatch"
    assert torch.allclose(indices_npu.cpu(), indices_cpu), f"[Rank {rank}] indices mismatch"


def worker_push_empty(rank, world_size, device, kwargs):
    cpu = torch.device("cpu")
    dtype = kwargs["dtype"]
    num_nodes = kwargs["num_nodes"]
    num_cols = kwargs["num_cols"]
    seed = kwargs.get("seed", 42) + rank
    torch.manual_seed(seed)

    row_start, row_end = _part_range_bounds(rank, world_size, num_nodes)
    local_rows_count = row_end - row_start

    if rank % 2 == 0:
        nnz = kwargs["nnz"]
        global_rows = torch.randint(0, num_nodes, (nnz,), dtype=dtype, device=device)
        cols = torch.randint(0, num_cols, (nnz,), dtype=dtype, device=device)
        vals = torch.randn(nnz, device=device)
    else:
        global_rows = torch.empty(0, dtype=dtype, device=device)
        cols = torch.empty(0, dtype=dtype, device=device)
        vals = torch.empty(0, device=device)

    if global_rows.shape[0] > 0:
        packed = torch.stack([cols.float(), vals], dim=1)
    else:
        packed = torch.empty(0, 2, device=device)

    chunk = (num_nodes + world_size - 1) // world_size
    ranges = [min(i * chunk, num_nodes) for i in range(world_size + 1)]
    part_ranges = torch.tensor(ranges, dtype=torch.int64, device=device)
    part = AscendNDArrayPartitionWrapper(num_nodes, world_size, "range", part_ranges=part_ranges)

    recv_rows, recv_packed = sparse_all_to_all_push(global_rows, packed, part)

    if recv_rows.shape[0] == 0:
        local_rows = torch.empty(0, dtype=dtype, device=device)
        recv_cols = torch.empty(0, dtype=dtype, device=device)
        recv_vals = torch.empty(0, device=device)
    else:
        recv_cols = recv_packed[:, 0].long().to(dtype)
        recv_vals = recv_packed[:, 1]
        local_rows = recv_rows - row_start

    mat_npu = from_coo(local_rows, recv_cols, recv_vals, shape=(local_rows_count, num_cols))
    mat_cpu = from_coo(local_rows.cpu(), recv_cols.cpu(), recv_vals.cpu(), shape=(local_rows_count, num_cols))

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert torch.allclose(indptr_npu.cpu(), indptr_cpu), f"[Rank {rank}] indptr mismatch"
    assert torch.allclose(indices_npu.cpu(), indices_cpu), f"[Rank {rank}] indices mismatch"


def worker_pull_range(rank, world_size, device, kwargs):
    cpu = torch.device("cpu")
    dtype = kwargs["dtype"]
    num_nodes = kwargs["num_nodes"]
    num_cols = kwargs["num_cols"]
    nnz = kwargs["nnz"]
    feat_dim = kwargs.get("feat_dim", 16)
    seed = kwargs.get("seed", 42) + rank
    torch.manual_seed(seed)

    row_start, row_end = _part_range_bounds(rank, world_size, num_nodes)
    local_rows_count = row_end - row_start

    local_rows = torch.randint(0, local_rows_count, (nnz,), dtype=dtype, device=device)
    cols = torch.randint(0, num_cols, (nnz,), dtype=dtype, device=device)
    vals = torch.randn(nnz, device=device)

    mat_npu = from_coo(local_rows, cols, vals, shape=(local_rows_count, num_cols))
    mat_cpu = from_coo(local_rows.cpu(), cols.cpu(), vals.cpu(), shape=(local_rows_count, num_cols))

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert torch.allclose(indptr_npu.cpu(), indptr_cpu), f"[Rank {rank}] local indptr mismatch"
    assert torch.allclose(indices_npu.cpu(), indices_cpu), f"[Rank {rank}] local indices mismatch"

    local_features = torch.randn(local_rows_count, feat_dim, device=device)
    num_queries = kwargs.get("num_queries", 50)

    chunk = (num_nodes + world_size - 1) // world_size
    ranges = [min(i * chunk, num_nodes) for i in range(world_size + 1)]
    part_ranges = torch.tensor(ranges, dtype=torch.int64, device=device)
    part = AscendNDArrayPartitionWrapper(num_nodes, world_size, "range", part_ranges=part_ranges)

    num_own = min(num_queries, local_rows_count)
    own_query = torch.arange(row_start, row_start + num_own, dtype=torch.int64, device=device)
    own_pulled = sparse_all_to_all_pull(own_query, local_features, part)
    assert torch.allclose(own_pulled, local_features[:num_own]), f"[Rank {rank}] own feature pull mismatch"


def worker_pull_remainder(rank, world_size, device, kwargs):
    cpu = torch.device("cpu")
    dtype = kwargs["dtype"]
    num_nodes = kwargs["num_nodes"]
    num_cols = kwargs["num_cols"]
    nnz = kwargs["nnz"]
    feat_dim = kwargs.get("feat_dim", 32)
    seed = kwargs.get("seed", 42) + rank
    torch.manual_seed(seed)

    local_rows_count = _compute_local_shape(num_nodes, world_size, rank, "remainder")

    if local_rows_count > 0 and nnz > 0:
        local_rows = torch.randint(0, local_rows_count, (nnz,), dtype=dtype, device=device)
        cols = torch.randint(0, num_cols, (nnz,), dtype=dtype, device=device)
        vals = torch.randn(nnz, device=device)
    else:
        local_rows = torch.empty(0, dtype=dtype, device=device)
        cols = torch.empty(0, dtype=dtype, device=device)
        vals = torch.empty(0, device=device)

    mat_npu = from_coo(local_rows, cols, vals, shape=(max(local_rows_count, 1), num_cols))
    mat_cpu = from_coo(local_rows.cpu(), cols.cpu(), vals.cpu(), shape=(max(local_rows_count, 1), num_cols))

    indptr_npu, indices_npu, _ = mat_npu.csr()
    indptr_cpu, indices_cpu, _ = mat_cpu.csr()

    assert torch.allclose(indptr_npu.cpu(), indptr_cpu), f"[Rank {rank}] local indptr mismatch"

    local_features = torch.randn(max(local_rows_count, 1), feat_dim, device=device)
    part = AscendNDArrayPartitionWrapper(num_nodes, world_size, "remainder")

    max_local = num_nodes // world_size
    if rank < num_nodes % world_size:
        max_local += 1

    num_own = min(30, max_local)
    own_query = torch.arange(rank, rank + num_own * world_size, world_size, dtype=torch.int64, device=device)[:num_own]

    if own_query.shape[0] > 0:
        own_pulled = sparse_all_to_all_pull(own_query, local_features, part)
        expected = local_features[:own_pulled.shape[0]]
        assert torch.allclose(own_pulled, expected), f"[Rank {rank}] own feature pull mismatch"


# ─── Test suite definition ─────────────────────────────────────────

TEST_SUITE = [
    ("push_range", worker_push_range, {"num_nodes": 100, "num_cols": 50, "nnz": 200, "dtype": torch.int64, "seed": 42}),
    ("push_range_int32", worker_push_range, {"num_nodes": 80, "num_cols": 40, "nnz": 150, "dtype": torch.int32, "seed": 137}),
    ("push_remainder", worker_push_remainder, {"num_nodes": 100, "num_cols": 50, "nnz": 200, "dtype": torch.int64, "seed": 73}),
    ("push_empty", worker_push_empty, {"num_nodes": 50, "num_cols": 25, "nnz": 100, "dtype": torch.int64, "seed": 5}),
    ("pull_range", worker_pull_range, {"num_nodes": 100, "num_cols": 50, "nnz": 200, "feat_dim": 16, "num_queries": 50, "dtype": torch.int64, "seed": 99}),
    ("pull_remainder", worker_pull_remainder, {"num_nodes": 80, "num_cols": 30, "nnz": 150, "dtype": torch.int64, "seed": 31}),
]


# ─── torchrun entry point ──────────────────────────────────────────

if __name__ == "__main__":
    if "RANK" not in os.environ:
        ws = min(torch.npu.device_count(), 4)
        print("=== HCCL + COOToCSR multi-card integration tests ===")
        print(f"NPU available: {torch.npu.is_available()}, count: {torch.npu.device_count()}")
        print()
        print("Run with torchrun for multi-card HCCL communication:")
        print(f"  torchrun --nproc_per_node={ws} tests/ascend/test_coo2csr_hccl_npu.py")
        print()
        print("Single-card smoke test (no HCCL communication):")
        print(f"  torchrun --nproc_per_node=1 tests/ascend/test_coo2csr_hccl_npu.py")
        sys.exit(0)

    # Pre-cleanup: destroy any stale process group from a prior incomplete run
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_id = local_rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    dist.init_process_group(backend="hccl", device_id=torch.device(f"npu:{device_id}"))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"npu:{device_id}")

    if rank == 0:
        print(f"=== HCCL + COOToCSR tests (world_size={world_size}) ===", flush=True)

    failures = 0
    try:
        for name, worker_fn, kwargs in TEST_SUITE:
            try:
                worker_fn(rank, world_size, device, kwargs)
                if rank == 0:
                    print(f"  PASS [{name}]", flush=True)
            except Exception as e:
                import traceback
                print(f"  [Rank {rank}] FAIL [{name}]: {e}", flush=True)
                traceback.print_exc()
                if "Bind_Failed" in str(e):
                    # Print HCCL port-related env vars to help diagnose
                    port_vars = [
                        "HCCL_IB_PORT", "HCCL_SERVER_PORT",
                        "HCCL_EXEC_PORT", "MULTI_HCCL_PARA_SCHED_PORT",
                        "HCCL_PORT", "HCCL_NET_PREFER",
                    ]
                    port_info = "; ".join(
                        f"{v}={os.environ.get(v, '<not set>')}" for v in port_vars
                    )
                    print(
                        f"  [Rank {rank}] HINT: HCCL ports not released. "
                        f"Run 'pkill -9 python' to clean orphaned HCCL processes, "
                        f"then retry.",
                        flush=True,
                    )
                    print(
                        f"  [Rank {rank}] HCCL port env: {port_info}",
                        flush=True,
                    )
                    print(
                        f"  [Rank {rank}] HCCL log: check ~/ascend_log/ or "
                        f"$ASCEND_LOG_DIR for detailed port binding info",
                        flush=True,
                    )
                failures += 1
    finally:
        # Ensure all ranks finish their HCCL ops before any rank tears down
        torch.npu.synchronize()
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

    if rank == 0:
        total = len(TEST_SUITE)
        passed = total - failures
        print(f"\n{'=' * 40}", flush=True)
        print(f"Results: {passed}/{total} passed, {failures} failed", flush=True)

    sys.exit(1 if failures > 0 else 0)
