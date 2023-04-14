import argparse
import os

import dgl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler
from dgl.nn.pytorch import RelGraphConv
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.functional import accuracy


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        # two-layer RGCN
        self.conv1 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )
        self.conv2 = RelGraphConv(
            h_dim,
            out_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )

    def forward(self, g):
        x = self.emb(g[0].srcdata[dgl.NID])
        h = F.relu(
            self.conv1(g[0], x, g[0].edata[dgl.ETYPE], g[0].edata["norm"])
        )
        h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], g[1].edata["norm"])
        return h


def evaluate(model, labels, num_classes, dataloader, inv_target):
    model.eval()
    eval_logits = []
    eval_seeds = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            output_nodes = inv_target[output_nodes]
            for block in blocks:
                block.edata["norm"] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(output_nodes.cpu().detach())
    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    num_seeds = len(eval_seeds)
    loc_sum = accuracy(
        eval_logits.argmax(dim=1),
        labels[eval_seeds].cpu(),
        task="multiclass",
        num_classes=num_classes,
    ) * float(num_seeds)
    return torch.tensor([loc_sum.item(), float(num_seeds)])


def train(
    proc_id,
    device,
    g,
    target_idx,
    labels,
    num_classes,
    train_idx,
    inv_target,
    model,
):
    # define loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # construct sampler and dataloader
    sampler = MultiLayerNeighborSampler([4, 4])
    train_loader = DataLoader(
        g,
        target_idx[train_idx],
        sampler,
        device=device,
        batch_size=100,
        shuffle=True,
        use_ddp=True,
    )
    # no separate validation subset, use train index instead for validation
    val_loader = DataLoader(
        g,
        target_idx[train_idx],
        sampler,
        device=device,
        batch_size=100,
        shuffle=False,
        use_ddp=True,
    )
    for epoch in range(50):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            output_nodes = inv_target[output_nodes]
            for block in blocks:
                block.edata["norm"] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            loss = loss_fcn(logits, labels[output_nodes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # torchmetric accuracy defined as num_correct_labels / num_train_nodes
        # loc_acc_split = [loc_accuracy * loc_num_train_nodes, loc_num_train_nodes]
        loc_acc_split = evaluate(
            model, labels, num_classes, val_loader, inv_target
        ).to(device)
        dist.reduce(loc_acc_split, 0)
        if proc_id == 0:
            acc = loc_acc_split[0] / loc_acc_split[1]
            print(
                "Epoch {:05d} | Loss {:.4f} | Val. Accuracy {:.4f} ".format(
                    epoch, total_loss / (it + 1), acc.item()
                )
            )


def run(proc_id, nprocs, devices, g, data):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )
    (
        num_rels,
        num_classes,
        labels,
        train_idx,
        test_idx,
        target_idx,
        inv_target,
    ) = data
    labels = labels.to(device)
    inv_target = inv_target.to(device)
    # create RGCN model (distributed)
    in_size = g.num_nodes()
    model = RGCN(in_size, 16, num_classes, num_rels).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    train(
        proc_id,
        device,
        g,
        target_idx,
        labels,
        num_classes,
        train_idx,
        inv_target,
        model,
    )
    test_sampler = MultiLayerNeighborSampler(
        [-1, -1]
    )  # -1 for sampling all neighbors
    test_loader = DataLoader(
        g,
        target_idx[test_idx],
        test_sampler,
        device=device,
        batch_size=32,
        shuffle=False,
        use_ddp=True,
    )
    loc_acc_split = evaluate(
        model, labels, num_classes, test_loader, inv_target
    ).to(device)
    dist.reduce(loc_acc_split, 0)
    if proc_id == 0:
        acc = loc_acc_split[0] / loc_acc_split[1]
        print("Test accuracy {:.4f}".format(acc))
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RGCN for entity classification with sampling (multi-gpu)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aifb",
        help="Dataset name ('aifb', 'mutag', 'bgs', 'am').",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    print(
        f"Training with DGL built-in RGCN module with sampling using",
        nprocs,
        f"GPU(s)",
    )

    # load and preprocess dataset at master(parent) process
    if args.dataset == "aifb":
        data = AIFBDataset()
    elif args.dataset == "mutag":
        data = MUTAGDataset()
    elif args.dataset == "bgs":
        data = BGSDataset()
    elif args.dataset == "am":
        data = AMDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    num_rels = len(g.canonical_etypes)
    category = data.predict_category
    labels = g.nodes[category].data.pop("labels")
    train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    # find target category and node id
    category_id = g.ntypes.index(category)
    g = dgl.to_homogeneous(g)
    node_ids = torch.arange(g.num_nodes())
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    # rename the fields as they can be changed by DataLoader
    g.ndata["ntype"] = g.ndata.pop(dgl.NTYPE)
    g.ndata["type_id"] = g.ndata.pop(dgl.NID)
    # find the mapping (inv_target) from global node IDs to type-specific node IDs
    inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64)
    inv_target[target_idx] = torch.arange(
        0, target_idx.shape[0], dtype=inv_target.dtype
    )
    # avoid creating certain graph formats and train/test indexes in each sub-process to save momory
    g.create_formats_()
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)

    data = (
        num_rels,
        data.num_classes,
        labels,
        train_idx,
        test_idx,
        target_idx,
        inv_target,
    )
    mp.spawn(run, args=(nprocs, devices, g, data), nprocs=nprocs)
