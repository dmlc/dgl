"""
Multi-card distributed GCN training tests on Ascend NPU.

Runs 3 test scenarios sequentially in a single torchrun session:
  1. converge — Full Cora graph, NodeEmbedding, convergence verification
  2. partition — METIS-partitioned Cora + HALO nodes, convergence verification
  3. e2e      — Synthetic graph, SpMM numerical correctness + gradient checks

Usage:
  torchrun --nproc_per_node=2 tests/ascend/test_dist_gcn_converge_npu.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import dgl
import dgl.function as fn
import dgl.distributed
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch.sparse_emb import NodeEmbedding
from dgl.optim.pytorch.sparse_optim import SparseAdagrad


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, h, norm):
        with g.local_scope():
            g.ndata['h'] = (h * norm).half()
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = g.ndata['h'].float() * norm
            return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.layer1 = GCNLayer(in_feats, hidden)
        self.layer2 = GCNLayer(hidden, out_feats)

    def forward(self, g, h, norm):
        h = torch.relu(self.layer1(g, h, norm))
        h = self.layer2(g, h, norm)
        return h


# ─── Scenario 1: full-graph convergence ──────────────────────────


def test_converge(rank, world_size, device):
    """Full Cora graph on each rank, NodeEmbedding by remainder, convergence check."""
    transform = dgl.AddSelfLoop()
    data = None
    if rank == 0:
        data = CoraGraphDataset(transform=transform)
    dist.barrier()
    if rank != 0:
        data = CoraGraphDataset(transform=transform)
    g = data[0]
    g = g.int()
    g = g.formats(['csc', 'csr'])
    g.create_formats_()

    num_nodes = g.num_nodes()
    in_feats = 32
    hidden = 64
    num_classes = data.num_classes
    labels = g.ndata['label'].to(device)

    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).unsqueeze(1)
    g = g.to(device)
    norm = norm.to(device)

    emb = NodeEmbedding(num_nodes, in_feats, "node_feat", device=device)
    opt = SparseAdagrad([emb], lr=0.1)
    model = GCN(in_feats, hidden, num_classes).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fcn = nn.CrossEntropyLoss()

    losses = []
    accs = []
    for epoch in range(50):
        x = emb(torch.arange(num_nodes, device=device), device=device)
        logits = model(g, x, norm.half())
        loss = loss_fcn(logits, labels)
        model_opt.zero_grad()
        loss.backward()
        model_opt.step()
        opt.step()

        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean().item()
        accs.append(acc)

        if rank == 0:
            print(f"  [converge epoch {epoch:02d}] loss={loss.item():.4f}  acc={acc:.4f}", flush=True)

    assert torch.isfinite(torch.tensor(losses)).all(), \
        f"[Rank {rank}] Non-finite loss detected"
    assert losses[-1] < losses[0], \
        f"[Rank {rank}] Loss did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert accs[-1] > accs[0], \
        f"[Rank {rank}] Accuracy did not improve: {accs[0]:.4f} -> {accs[-1]:.4f}"
    assert emb.weight.norm().item() > 0, \
        f"[Rank {rank}] Embedding norm is zero"

    if rank == 0:
        print(f"  PASS [converge]  loss {losses[0]:.4f}->{losses[-1]:.4f}  acc {accs[0]:.4f}->{accs[-1]:.4f}", flush=True)


# ─── Scenario 2: METIS-partitioned convergence ───────────────────


def test_partition(rank, world_size, device):
    """METIS-partitioned Cora with HALO nodes, convergence check."""
    transform = dgl.AddSelfLoop()
    data = CoraGraphDataset(transform=transform)
    g = data[0]
    g = g.formats(['coo', 'csc', 'csr'])
    g.create_formats_()

    num_nodes = g.num_nodes()
    in_feats = 32
    hidden = 64
    num_classes = data.num_classes

    part_dir = "/tmp/cora_partition_test"
    if rank == 0:
        os.makedirs(part_dir, exist_ok=True)
        dgl.distributed.partition_graph(
            g, "cora", world_size, part_dir,
            num_hops=1, part_method="metis",
            graph_formats=['coo', 'csc', 'csr'],
            return_mapping=False,
        )
    dist.barrier()

    part_config = os.path.join(part_dir, "cora.json")
    part_g, node_feats, _, gpb, _, _, _ = dgl.distributed.load_partition(
        part_config, rank, load_feats=True
    )
    part_g = part_g.int()
    part_g = part_g.formats(['coo', 'csc', 'csr'])
    part_g.create_formats_()

    inner_mask = part_g.ndata['inner_node'].bool()
    global_nid = part_g.ndata[dgl.NID]

    local_labels = node_feats['_N/label'].long().to(device)

    degs = part_g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).unsqueeze(1)

    part_g = part_g.to(device)
    norm = norm.to(device)
    global_nid = global_nid.to(device)
    inner_mask_npu = inner_mask.to(device)

    emb = NodeEmbedding(num_nodes, in_feats, "node_feat", device=device)
    opt = SparseAdagrad([emb], lr=0.1)
    model = GCN(in_feats, hidden, num_classes).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fcn = nn.CrossEntropyLoss()

    losses = []
    accs = []
    for epoch in range(50):
        x = emb(global_nid, device=device)
        logits = model(part_g, x, norm)
        local_logits = logits[inner_mask_npu]
        loss = loss_fcn(local_logits, local_labels)

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()
        opt.step()

        losses.append(loss.item())
        pred = local_logits.argmax(dim=1)
        acc = (pred == local_labels).float().mean().item()
        accs.append(acc)

        local_count = inner_mask_npu.sum().item()
        if rank == 0:
            print(f"  [partition epoch {epoch:02d}] loss={loss.item():.4f}  "
                  f"acc={acc:.4f}  local_nodes={local_count}", flush=True)

    assert torch.isfinite(torch.tensor(losses)).all(), \
        f"[Rank {rank}] Non-finite loss detected"
    assert losses[-1] < losses[0], \
        f"[Rank {rank}] Loss did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert accs[-1] > accs[0], \
        f"[Rank {rank}] Accuracy did not improve: {accs[0]:.4f} -> {accs[-1]:.4f}"
    assert emb.weight.norm().item() > 0, \
        f"[Rank {rank}] Embedding norm is zero"

    if rank == 0:
        print(f"  PASS [partition]  loss {losses[0]:.4f}->{losses[-1]:.4f}  acc {accs[0]:.4f}->{accs[-1]:.4f}", flush=True)


# ─── Scenario 3: synthetic e2e with SpMM verification ────────────


def test_e2e(rank, world_size, device):
    """Synthetic graph, per-epoch SpMM CPU reference check, gradient sanity."""
    num_nodes = 200
    num_edges = 2000
    dim = 16
    num_classes = 5
    num_epochs = 10

    torch.manual_seed(42)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst)).int()
    g = g.formats(['csc', 'csr'])
    g.create_formats_()
    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).unsqueeze(1)

    g_cpu = g
    norm_cpu = norm.clone()

    g = g.to(device)
    norm = norm.to(device)

    emb = NodeEmbedding(num_nodes, dim, "node_feat", device=device)
    torch.manual_seed(0)
    all_labels = torch.randint(0, num_classes, (num_nodes,), device=device)

    opt = SparseAdagrad([emb], lr=0.1)
    model = GCN(dim, 16, num_classes).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fcn = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(num_epochs):
        x = emb(torch.arange(num_nodes, device=device), device=device)

        x_cpu = x.cpu().float()
        with g_cpu.local_scope():
            g_cpu.ndata['h'] = x_cpu * norm_cpu
            g_cpu.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            spmm_ref = g_cpu.ndata['h'] * norm_cpu

        h = x
        with g.local_scope():
            g.ndata['h'] = (h * norm).half()
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            spmm1_out = g.ndata['h'].float() * norm
        spmm1_out.retain_grad()

        if rank == 0 and epoch == 0:
            if not torch.allclose(spmm1_out.cpu().float(), spmm_ref.float(), atol=1e-3, rtol=1e-3, equal_nan=True):
                diff = (spmm1_out.cpu().float() - spmm_ref.float()).abs()
                print(f"  [Rank {rank}] SpMM MISMATCH! max_diff={diff.max().item():.6f} "
                      f"NPU_nan_rows={torch.where(~torch.isfinite(spmm1_out).any(dim=1))[0].tolist()}", flush=True)
            else:
                print(f"  [Rank {rank}] SpMM OK (matches CPU)", flush=True)
        h1 = model.layer1.linear(spmm1_out)

        h = torch.relu(h1)
        with g.local_scope():
            g.ndata['h'] = (h * norm).half()
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            spmm2_out = g.ndata['h'].float() * norm
        spmm2_out.retain_grad()
        logits = model.layer2.linear(spmm2_out)

        loss = loss_fcn(logits.float(), all_labels)

        model_opt.zero_grad()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                max_grad = p.grad.float().abs().max().item()
                nan_grad = (~torch.isfinite(p.grad)).float().sum().item()
                if nan_grad > 0 or max_grad > 1e4:
                    print(f"  [Rank {rank}] grad {name}: max_abs={max_grad:.4f} NaN_count={nan_grad}", flush=True)

        model_opt.step()
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"[Rank {rank}] model param {name} has NaN at epoch {epoch}"
        opt.step()
        assert torch.isfinite(emb.weight).all(), f"[Rank {rank}] emb has NaN after opt.step() at epoch {epoch}"
        losses.append(loss.item())
        if rank == 0:
            print(f"  [e2e epoch {epoch:02d}] loss={loss.item():.4f}", flush=True)

    for i, l in enumerate(losses):
        assert torch.isfinite(torch.tensor(l)), \
            f"[Rank {rank}] loss[{i}] is non-finite: {l}"

    local_norm = emb.weight.norm().item()
    assert local_norm > 0, f"[Rank {rank}] embedding norm is zero"
    if rank == 0:
        print(f"  PASS [e2e]  embedding_norm={local_norm:.4f}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    assert "RANK" in os.environ, "Run with torchrun"
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = local_rank % torch.npu.device_count()
    torch.npu.set_device(f"npu:{device_id}")
    dist.init_process_group(backend="hccl", device_id=torch.device(f"npu:{device_id}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"npu:{device_id}")

    if rank == 0:
        print(f"=== Distributed GCN tests (world_size={world_size}) ===", flush=True)

    scenarios = [
        ("converge", test_converge),
        ("partition", test_partition),
        ("e2e", test_e2e),
    ]

    failures = []
    for name, scenario_fn in scenarios:
        try:
            dist.barrier()
            if rank == 0:
                print(f"\n--- [{name}] ---", flush=True)
            scenario_fn(rank, world_size, device)
            if rank == 0:
                print(f"  OK", flush=True)
        except Exception as e:
            import traceback
            print(f"  [Rank {rank}] FAIL [{name}]: {e}", flush=True)
            traceback.print_exc()
            failures.append(name)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        total = len(scenarios)
        passed = total - len(failures)
        print(f"\n{'=' * 40}", flush=True)
        print(f"Results: {passed}/{total} passed", flush=True)
        if failures:
            print(f"Failed: {failures}", flush=True)

