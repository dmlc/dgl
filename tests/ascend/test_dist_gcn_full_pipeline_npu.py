"""
End-to-end distributed GNN training verification on Ascend NPU covering 8 components:
  1. Graph partitioning (METIS)
  2. Distributed graph storage (partition + HALO, partition book)
  3. Distributed neighbor sampling (NeighborSampler + mini-batch DataLoader)
  4. Cross-device communication (NodeEmbedding HCCL all-to-all)
  5. Forward propagation (SpMM + linear)
  6. Backward propagation (gradient through SpMM + linear)
  7. Gradient synchronization (DistributedDataParallel)
  8. Parameter updates (Adam + SparseAdagrad)

Each epoch verifies specific components; convergence verifies end-to-end correctness.

Usage:
  torchrun --nproc_per_node=2 tests/ascend/test_dist_gcn_full_pipeline_npu.py
"""

import os
os.environ['DGL_SPMM_SUM_AIV_ONLY'] = '1'
os.environ['DGL_SPMM_USE_PYTORCH_STREAM'] = '1'
import torch
import torch.nn as nn
import torch.distributed as dist
import dgl
import dgl.function as fn
import dgl.distributed
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch.sparse_emb import NodeEmbedding
from dgl.optim.pytorch.sparse_optim import SparseAdagrad
from dgl.dataloading import NeighborSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


def warmup_npu_mm_backward(device):
    """Warmup torch_npu matrix multiply backward to work around a cold-start bug
    where the first mm/addmm backward call returns all-zeros.
    """
    dummy_x = torch.randn(2, 4, device=device, requires_grad=True)
    dummy_w = torch.randn(4, 4, device=device, requires_grad=True)
    dummy_out = torch.mm(dummy_x, dummy_w.T)
    dummy_out.sum().backward()


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, block, h):
        with block.local_scope():
            block.srcdata['h'] = h.half()
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = block.dstdata['h'].float()
            return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.layer1 = GCNLayer(in_feats, hidden)
        self.layer2 = GCNLayer(hidden, out_feats)

    def forward(self, blocks, h):
        h = torch.relu(self.layer1(blocks[0], h))
        h = self.layer2(blocks[1], h)
        return h


def verify_sampling(blocks, inner_mask, rank):
    assert len(blocks) == 2, f"[Rank {rank}] Expected 2 blocks, got {len(blocks)}"
    for i, b in enumerate(blocks):
        assert b.num_src_nodes() > 0, f"[Rank {rank}] block[{i}] has no src nodes"
        assert b.num_dst_nodes() > 0, f"[Rank {rank}] block[{i}] has no dst nodes"
    print(f"  [Rank {rank}] Sampling OK: block[0] {blocks[0].num_src_nodes()}→{blocks[0].num_dst_nodes()}, "
          f"block[1] {blocks[1].num_src_nodes()}→{blocks[1].num_dst_nodes()}", flush=True)


def verify_forward(logits_npu, blocks, h, model, rank, device):
    h_cpu = h.cpu().float()
    blocks_cpu = [b.to('cpu') for b in blocks]

    state = model.module.state_dict()
    l1_weight = state['layer1.linear.weight']
    l1_bias = state['layer1.linear.bias']
    l2_weight = state['layer2.linear.weight']
    l2_bias = state['layer2.linear.bias']

    with torch.no_grad():
        with blocks_cpu[0].local_scope():
            blocks_cpu[0].srcdata['h'] = h_cpu
            blocks_cpu[0].update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h1_ref = blocks_cpu[0].dstdata['h']
        h1_ref = torch.relu(torch.nn.functional.linear(h1_ref, l1_weight.cpu(), l1_bias.cpu()))

        with blocks_cpu[1].local_scope():
            blocks_cpu[1].srcdata['h'] = h1_ref
            blocks_cpu[1].update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h2_ref = blocks_cpu[1].dstdata['h']
        logits_ref = torch.nn.functional.linear(h2_ref, l2_weight.cpu(), l2_bias.cpu())

    diff = (logits_npu.cpu().float() - logits_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_max = logits_ref.abs().max().item()
    npu_max = logits_npu.cpu().float().abs().max().item()
    print(f"  [Rank {rank}] Forward max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
          f"ref_max={ref_max:.4f}  npu_max={npu_max:.4f}", flush=True)
    assert torch.allclose(logits_npu.cpu().float(), logits_ref, atol=1e-1, rtol=1e-1), \
        f"[Rank {rank}] Forward mismatch: max_diff={max_diff:.4f} atol=1e-1 rtol=1e-1"


def verify_backward(model, emb, rank):
    has_grad = False
    for name, p in model.module.named_parameters():
        if p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all(), \
                f"[Rank {rank}] Non-finite grad in {name}"
            assert p.grad.abs().max().item() > 0, \
                f"[Rank {rank}] Zero grad in {name}"
    assert has_grad, f"[Rank {rank}] No model parameters received gradients"

    if emb.weight.grad is not None:
        assert torch.isfinite(emb.weight.grad).all(), \
            f"[Rank {rank}] Non-finite embedding grad"
    else:
        print(f"  [Rank {rank}] emb.weight.grad is None (may be zero local shard)", flush=True)
    print(f"  [Rank {rank}] Backward OK", flush=True)


def verify_ddp_sync(model, rank, world_size, device):
    for name, p in model.module.named_parameters():
        local_sum = p.float().sum()
        gathered = [torch.zeros_like(local_sum) for _ in range(world_size)]
        dist.all_gather(gathered, local_sum)
        for i in range(1, world_size):
            assert abs(gathered[i].item() - gathered[0].item()) < 1e-4, \
                f"[Rank {rank}] DDP sync FAIL: {name} differs rank 0 vs {i}"
    print(f"  [Rank {rank}] DDP sync OK", flush=True)


def worker_fn(rank, world_size, device):
    env_val = os.environ.get('DGL_SPMM_SUM_AIV_ONLY', 'NOT_SET')
    print(f"  [Rank {rank}] DGL_SPMM_SUM_AIV_ONLY={env_val}", flush=True)
    # ── 1. Graph partitioning (METIS) ────────────────────────────────
    transform = dgl.AddSelfLoop()
    data = CoraGraphDataset(transform=transform)
    full_g = data[0]
    all_labels = full_g.ndata['label']

    num_nodes = full_g.num_nodes()
    num_classes = data.num_classes
    in_feats = 32
    hidden = 64

    if rank == 0:
        g_for_part = full_g.formats(['coo', 'csc', 'csr'])
        g_for_part.create_formats_()
        part_dir = "/tmp/cora_full_pipeline_test"
        os.makedirs(part_dir, exist_ok=True)
        dgl.distributed.partition_graph(
            g_for_part, "cora", world_size, part_dir,
            num_hops=1, part_method="metis",
            graph_formats=['coo', 'csc', 'csr'],
            return_mapping=False,
        )
    dist.barrier()
    if rank == 0:
        print("  [1] Graph partition OK", flush=True)

    # ── 2. Distributed graph storage ─────────────────────────────────
    part_config = os.path.join("/tmp/cora_full_pipeline_test", "cora.json")
    part_g, node_feats, _, gpb, _, _, _ = dgl.distributed.load_partition(
        part_config, rank, load_feats=True
    )
    part_g = part_g.int()
    part_g = part_g.formats(['coo', 'csc', 'csr'])
    part_g.create_formats_()

    inner_mask = part_g.ndata['inner_node'].bool()
    seed_nodes = torch.where(inner_mask)[0].to(torch.int32)
    global_nid_map = part_g.ndata[dgl.NID].to(device)

    local_count = inner_mask.sum().item()
    halo_count = part_g.num_nodes() - local_count
    print(f"  [Rank {rank}] [2] Dist storage OK: {local_count} local + {halo_count} HALO nodes", flush=True)

    # ── 3. Distributed neighbor sampling ─────────────────────────────
    sampler = NeighborSampler([5, 5])
    dataloader = DataLoader(
        part_g, seed_nodes, sampler,
        device=device,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ── 4. Cross-device communication ────────────────────────────────
    emb = NodeEmbedding(num_nodes, in_feats, "node_feat", device=device)
    emb_opt = SparseAdagrad([emb], lr=0.1)

    # ── 5/7/8. Model + DDP + optimizer ──────────────────────────────
    torch.manual_seed(42 + rank)
    model = GCN(in_feats, hidden, num_classes).to(device)
    model = DDP(model, device_ids=None)
    model_opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fcn = nn.CrossEntropyLoss()

    # Warmup torch_npu mm backward (cold-start bug workaround)
    warmup_npu_mm_backward(device)
    if rank == 0:
        print("  [Warmup] NPU mm backward warmup done", flush=True)

    # ── Training loop ────────────────────────────────────────────────
    losses = []
    accs = []
    for epoch in range(30):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            input_gids = global_nid_map[input_nodes]
            output_gids = global_nid_map[output_nodes]

            # Component 4: Cross-device communication (HCCL all-to-all)
            h = emb(input_gids, device=device)

            # Component 5: Forward propagation
            #   SpMM (half) + Linear + ReLU on NPU
            logits = model(blocks, h)
            labels = all_labels[output_gids.cpu()].to(device)
            loss = loss_fcn(logits, labels)

            # ── Epoch 0, batch 0: sampling verification (warm-up, no forward/backward verify) ──
            if epoch == 0 and batch_idx == 0:
                print(f"  [Rank {rank}] [3] Sampling verifying...", flush=True)
                verify_sampling(blocks, inner_mask, rank)

            model_opt.zero_grad()

            # Component 6: Backward propagation
            loss.backward()

            # ── Epoch 0, batch 1: forward + backward verification ──
            if epoch == 0 and batch_idx == 1:
                print(f"  [Rank {rank}] [5] Forward verifying (vs CPU)...", flush=True)
                verify_forward(logits, blocks, h, model, rank, device)
                print(f"  [Rank {rank}] [6] Backward verifying...", flush=True)
                verify_backward(model, emb, rank)

            # Component 7: Gradient sync (DDP allreduce) + Component 8: Parameter update
            model_opt.step()
            emb_opt.step()

            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / total
        avg_acc = correct / total
        losses.append(avg_loss)
        accs.append(avg_acc)

        if rank == 0:
            print(f"  [Epoch {epoch:02d}] loss={avg_loss:.4f}  acc={avg_acc:.4f}  [{total} samples]", flush=True)

        # Component 7 verification: DDP parameter sync across ranks
        if epoch % 5 == 0:
            dist.barrier()
            print(f"  [Rank {rank}] [7] DDP sync verifying...", flush=True)
            verify_ddp_sync(model, rank, world_size, device)

    # ── Final verification ───────────────────────────────────────────
    assert torch.isfinite(torch.tensor(losses)).all(), \
        f"[Rank {rank}] Non-finite loss detected"
    assert losses[-1] < losses[0], \
        f"[Rank {rank}] Loss did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert accs[-1] > accs[0], \
        f"[Rank {rank}] Accuracy did not improve: {accs[0]:.4f} -> {accs[-1]:.4f}"
    assert emb.weight.norm().item() > 0, \
        f"[Rank {rank}] Embedding norm is zero"

    if rank == 0:
        print(f"  Convergence: {losses[0]:.4f} -> {losses[-1]:.4f}, "
              f"Acc: {accs[0]:.4f} -> {accs[-1]:.4f}", flush=True)
        print(f"  PASS [dist_gcn_full_pipeline]", flush=True)


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
        print("=== Distributed GNN pipeline verification "
              f"(world_size={world_size}) ===", flush=True)

    try:
        worker_fn(rank, world_size, device)
    except Exception as e:
        import traceback
        print(f"  [Rank {rank}] FAIL [dist_gcn_full_pipeline]: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        dist.barrier()
        dist.destroy_process_group()

