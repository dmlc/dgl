"""
This script trains and tests a GraphSAGE model for node classification on
multiple GPUs with distributed data-parallel training (DDP).

Before reading this example, please familiar yourself with graphsage node
classification using neighbor sampling by reading the example in the
`examples/sampling/node_classification.py`

This flowchart describes the main functional sequence of the provided example.
main
│
├───> Load and preprocess dataset
│
└───> run (multiprocessing) 
      │
      ├───> Init process group and build distributed SAGE model (HIGHLIGHT)
      │
      ├───> train
      │     │
      │     ├───> NeighborSampler
      │     │
      │     └───> Training loop
      │           │
      │           ├───> SAGE.forward
      │           │
      │           └───> Collect validation accuracy (HIGHLIGHT)
      │
      └───> layerwise_infer
            │
            └───> SAGE.inference
                  │
                  ├───> MultiLayerFullNeighborSampler
                  │
                  └───> Use a shared output tensor
"""
import argparse
import os
import time

import dgl
import dgl.nn as dglnn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,  # use DDP
                use_uva=use_uva,
            )
            # In order to prevent running out of GPU memory, allocate a shared
            # output tensor 'y' in host memory.
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size
                    if l != len(self.layers) - 1
                    else self.out_size,
                )
            )
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # Non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # Use a barrier to make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y


def evaluate(model, g, num_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(
    proc_id, device, g, num_classes, nid, model, use_uva, batch_size=2**10
):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        print(f"Test accuracy {acc.item():.4f}")


def train(
    proc_id,
    nprocs,
    device,
    g,
    num_classes,
    train_idx,
    val_idx,
    model,
    use_uva,
    num_epochs,
):
    # Instantiate a neighbor sampler
    sampler = NeighborSampler(
        [10, 10, 10], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,  # To split the set for each process
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()  # Gradients are synchronized in DDP
            total_loss += loss
        #####################################################################
        # (HIGHLIGHT) Collect accuracy values from sub-processes and obtain
        # overall accuracy.
        #
        # `torch.distributed.reduce` is used to reduce tensors from all the
        # sub-processes to a specified process, ReduceOp.SUM is used by default.
        #
        # Other multiprocess functions supported by the backend are also
        # available. Please refer to
        # https://pytorch.org/docs/stable/distributed.html
        # for more information.
        #####################################################################
        acc = (
            evaluate(model, g, num_classes, val_dataloader).to(device) / nprocs
        )
        t1 = time.time()
        # Reduce `acc` tensors to process 0.
        dist.reduce(tensor=acc, dst=0)
        if proc_id == 0:
            print(
                f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | "
                f"Accuracy {acc.item():.4f} | Time {t1 - t0:.4f}"
            )


def run(proc_id, nprocs, devices, g, data, mode, num_epochs):
    # Find corresponding device for current process.
    device = devices[proc_id]
    torch.cuda.set_device(device)
    #########################################################################
    # (HIGHLIGHT) Build a data-parallel distributed GraphSAGE model.
    #
    # DDP in PyTorch provides data parallelism across the devices specified
    # by the `process_group`. Gradients are synchronized across each model
    # replica.
    #
    # To prepare a training sub-process, there are four steps involved:
    # 1. Initialize the process group
    # 2. Unpack data for the sub-process.
    # 3. Instantiate a GraphSAGE model on the corresponding device.
    # 4. Parallelize the model with `DistributedDataParallel`.
    #
    # For the detailed usage of `DistributedDataParallel`, please refer to
    # PyTorch documentation.
    #########################################################################
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )
    num_classes, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    g = g.to(device if mode == "puregpu" else "cpu")
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 256, num_classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    # Training.
    use_uva = mode == "mixed"
    if proc_id == 0:
        print("Training...")
    train(
        proc_id,
        nprocs,
        device,
        g,
        num_classes,
        train_idx,
        val_idx,
        model,
        use_uva,
        num_epochs,
    )

    # Testing.
    if proc_id == 0:
        print("Testing...")
    layerwise_infer(proc_id, device, g, num_classes, test_idx, model, use_uva)

    # Cleanup the process group.
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs for train.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ogbn-products",
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Root directory of dataset.",
    )
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # Load and preprocess the dataset.
    print("Loading data")
    dataset = AsNodePredDataset(
        DglNodePropPredDataset(args.dataset_name, root=args.dataset_dir)
    )
    g = dataset[0]
    # Explicitly create desired graph formats before multi-processing to avoid
    # redundant creation in each sub-process and to save memory.
    g.create_formats_()
    if args.dataset_name == "ogbn-arxiv":
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )

    # To use DDP with n GPUs, spawn up n processes.
    mp.spawn(
        run,
        args=(nprocs, devices, g, data, args.mode, args.num_epochs),
        nprocs=nprocs,
    )
