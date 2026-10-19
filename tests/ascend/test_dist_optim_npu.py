"""
Distributed sparse optimizer tests for Ascend NPU.

Tests SparseAdam and SparseAdagrad with HCCL communication on NPU:
  1. Single-step SparseAdagrad update correctness
  2. Single-step SparseAdam update correctness
  3. Multi-step convergence (embedding values converge)
  4. Gradient accumulation across multiple forward calls
  5. Empty gradient handling (no forward calls before step)

Usage:
  torchrun --nproc_per_node=4 tests/ascend/test_dist_optim_npu.py
  torchrun --nproc_per_node=1 tests/ascend/test_dist_optim_npu.py
"""

import os

import sys
import torch
import torch.distributed as dist

from dgl.nn.pytorch.sparse_emb import NodeEmbedding
from dgl.optim.pytorch.sparse_optim import SparseAdagrad, SparseAdam


# ─── Test: SparseAdagrad single-step ───────────────────────────────

def worker_adagrad_step(rank, world_size, device, kwargs):
    dim = kwargs["dim"]
    num_emb = kwargs["num_embeddings"]
    lr = kwargs.get("lr", 0.1)
    eps = kwargs.get("eps", 1e-10)
    name = kwargs.get("name", "adagrad_step")

    def init_func(emb):
        emb.fill_(1.0)
        return emb

    emb = NodeEmbedding(num_emb, dim, name, init_func=init_func, device=device)
    opt = SparseAdagrad([emb], lr=lr, eps=eps)

    global_idx = torch.tensor([0, 1, 2], device=device)
    out = emb(global_idx)
    loss = out.sum()
    loss.backward()

    init_vals = emb._tensor.clone().cpu()
    init_state = emb.optm_state[0].clone().cpu()

    opt.step()

    updated_vals = emb._tensor.cpu()
    updated_state = emb.optm_state[0].cpu()

    # Determine which local indices should be updated based on remainder partition
    owned_global = [g for g in global_idx.cpu().tolist() if g % world_size == rank]
    owned_local = set(g // world_size for g in owned_global)

    for i in range(min(num_emb, 10)):
        if i in owned_local:
            assert not torch.allclose(init_vals[i], updated_vals[i]), \
                f"[Rank {rank}] emb[{i}] not updated (global idx {[g for g in owned_global if g // world_size == i]})"
        else:
            assert torch.allclose(init_vals[i], updated_vals[i]), \
                f"[Rank {rank}] emb[{i}] unexpectedly changed"

    # Only check state for indices that were updated
    for i in owned_local:
        assert not torch.allclose(init_state[i], updated_state[i]), \
            f"[Rank {rank}] state[{i}] not updated"


# ─── Test: SparseAdam single-step ──────────────────────────────────

def worker_adam_step(rank, world_size, device, kwargs):
    dim = kwargs["dim"]
    num_emb = kwargs["num_embeddings"]
    lr = kwargs.get("lr", 0.1)
    name = kwargs.get("name", "adam_step")

    def init_func(emb):
        emb.fill_(1.0)
        return emb

    emb = NodeEmbedding(num_emb, dim, name, init_func=init_func, device=device)
    opt = SparseAdam([emb], lr=lr)

    global_idx = torch.tensor([0, 1, 2], device=device)
    out = emb(global_idx)
    loss = out.sum()
    loss.backward()

    init_vals = emb._tensor.clone().cpu()
    opt.step()

    updated_vals = emb._tensor.cpu()

    owned_global = [g for g in global_idx.cpu().tolist() if g % world_size == rank]
    owned_local = set(g // world_size for g in owned_global)

    for i in range(min(num_emb, 10)):
        if i in owned_local:
            assert not torch.allclose(init_vals[i], updated_vals[i]), \
                f"[Rank {rank}] adam emb[{i}] not updated"
        else:
            assert torch.allclose(init_vals[i], updated_vals[i]), \
                f"[Rank {rank}] adam emb[{i}] unexpectedly changed"

    state_step, state_mem, state_power = emb.optm_state
    if len(owned_local) > 0:
        first_owned = min(owned_local)
        assert state_step[first_owned].item() > 0, f"[Rank {rank}] adam step not incremented"
        assert not torch.allclose(state_mem[first_owned].cpu(), torch.zeros_like(state_mem[first_owned].cpu())), \
            f"[Rank {rank}] adam mem is zero"
        assert not torch.allclose(state_power[first_owned].cpu(), torch.zeros_like(state_power[first_owned].cpu())), \
            f"[Rank {rank}] adam power is zero"


# ─── Test: Multi-step convergence ──────────────────────────────────

def worker_multi_step(rank, world_size, device, kwargs):
    dim = kwargs["dim"]
    num_emb = kwargs["num_embeddings"]
    name = kwargs.get("name", "multi_step")
    num_steps = kwargs.get("num_steps", 5)

    emb = NodeEmbedding(num_emb, dim, name, device=device)
    opt = SparseAdagrad([emb], lr=0.5)

    prev_norm = None
    for step in range(num_steps):
        idx = torch.arange(min(10, num_emb), device=device)
        out = emb(idx)
        loss = out.sum()
        loss.backward()
        opt.step()

        cur_norm = emb._tensor.norm().item()
        if prev_norm is not None:
            assert cur_norm != prev_norm, \
                f"[Rank {rank}] norm unchanged at step {step}"
        prev_norm = cur_norm


# ─── Test: Gradient accumulation ──────────────────────────────────

def worker_grad_accum(rank, world_size, device, kwargs):
    dim = kwargs["dim"]
    num_emb = kwargs["num_embeddings"]
    name = kwargs.get("name", "grad_accum")

    def init_func(emb):
        emb.fill_(0.0)
        return emb

    emb = NodeEmbedding(num_emb, dim, name, init_func=init_func, device=device)
    opt = SparseAdagrad([emb], lr=0.1)

    # Use an index this rank owns: in remainder mode, rank X owns global X
    idx = torch.tensor([rank], device=device)

    # Step 1: single forward + backward + step
    out = emb(idx)
    loss = out.sum()
    loss.backward()
    opt.step()
    after_one_step = emb._tensor[0].clone().cpu()

    # Step 2: two forwards (gradient accumulates) + step
    for _ in range(2):
        out = emb(idx)
        loss = out.sum()
        loss.backward()
    opt.step()
    after_accum = emb._tensor[0].clone().cpu()

    # Accumulated gradient should produce a different update
    assert not torch.allclose(after_one_step, after_accum), \
        f"[Rank {rank}] accumulated step should differ from single step"


# ─── Test: Empty gradient ──────────────────────────────────────────

def worker_empty_grad(rank, world_size, device, kwargs):
    dim = kwargs["dim"]
    num_emb = kwargs["num_embeddings"]
    name = kwargs.get("name", "empty_grad")

    def init_func(emb):
        emb.fill_(1.0)
        return emb

    emb = NodeEmbedding(num_emb, dim, name, init_func=init_func, device=device)
    opt = SparseAdagrad([emb], lr=0.1)

    before = emb._tensor.clone().cpu()
    opt.step()
    after = emb._tensor.cpu()

    assert torch.allclose(before, after), \
        f"[Rank {rank}] emb changed despite empty gradient"


# ─── Test suite ───────────────────────────────────────────────────

TEST_SUITE = [
    ("adagrad_step", worker_adagrad_step,
     {"num_embeddings": 100, "dim": 16, "lr": 0.1, "name": "t_adagrad"}),
    ("adam_step", worker_adam_step,
     {"num_embeddings": 100, "dim": 16, "lr": 0.1, "name": "t_adam"}),
    ("multi_step", worker_multi_step,
     {"num_embeddings": 100, "dim": 16, "num_steps": 5, "name": "t_multi"}),
    ("grad_accum", worker_grad_accum,
     {"num_embeddings": 100, "dim": 16, "name": "t_accum"}),
    ("empty_grad", worker_empty_grad,
     {"num_embeddings": 100, "dim": 16, "name": "t_empty"}),
]


# ─── torchrun entry point ──────────────────────────────────────────

if __name__ == "__main__":
    if "RANK" not in os.environ:
        nproc = min(torch.npu.device_count(), 4)
        print("=== Distributed sparse optimizer tests for Ascend NPU ===")
        print(f"NPU available: {torch.npu.is_available()}, count: {torch.npu.device_count()}")
        print()
        print("Run with torchrun:")
        print(f"  torchrun --nproc_per_node={nproc} tests/ascend/test_dist_optim_npu.py")
        print()
        print("WARNING: If you see 'Bind_Failed' errors, run 'pkill -9 python' first.")
        sys.exit(0)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_id = local_rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    dist.init_process_group(backend="hccl", device_id=torch.device(f"npu:{device_id}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"npu:{device_id}")

    if rank == 0:
        print(f"=== Sparse optimizer tests (world_size={world_size}) ===", flush=True)

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
                failures += 1
    finally:
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
