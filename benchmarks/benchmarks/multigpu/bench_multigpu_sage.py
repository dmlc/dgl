import argparse
import math
import time
from types import SimpleNamespace
from typing import NamedTuple

import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

from .. import utils


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(dev_id)
    batch_labels = labels[seeds].to(dev_id)
    return batch_inputs, batch_labels


# Entry point
@utils.thread_wrapped_func
def run(result_queue, proc_id, n_gpus, args, devices, data):
    dev_id = devices[proc_id]
    timing_records = []
    if n_gpus > 1:
        dist_init_method = "tcp://{master_ip}:{master_port}".format(
            master_ip="127.0.0.1", master_port="12345"
        )
        world_size = n_gpus
        th.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=world_size,
            rank=proc_id,
        )
    th.cuda.set_device(dev_id)

    n_classes, train_g, _, _ = data

    train_nfeat = train_g.ndata.pop("feat")
    train_labels = train_g.ndata.pop("label")

    train_nfeat = train_nfeat.to(dev_id)
    train_labels = train_labels.to(dev_id)

    in_feats = train_nfeat.shape[1]

    train_mask = train_g.ndata["train_mask"]
    train_nid = train_mask.nonzero().squeeze()

    # Split train_nid
    train_nid = th.split(train_nid, math.ceil(len(train_nid) / n_gpus))[proc_id]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Define model and optimizer
    model = SAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[dev_id], output_device=dev_id
        )
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        if proc_id == 0:
            tic_step = time.time()

        batch_inputs, batch_labels = load_subtensor(
            train_nfeat, train_labels, seeds, input_nodes, dev_id
        )
        blocks = [block.int().to(dev_id) for block in blocks]
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if proc_id == 0:
            timing_records.append(time.time() - tic_step)

        if step >= 50:
            break

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        result_queue.put(np.array(timing_records))


@utils.benchmark("time", timeout=600)
@utils.skip_if_not_4gpu()
@utils.parametrize("data", ["reddit", "ogbn-products"])
def track_time(data):
    args = SimpleNamespace(
        num_hidden=16,
        fan_out="10,25",
        batch_size=1000,
        lr=0.003,
        dropout=0.5,
        num_layers=2,
        num_workers=4,
    )

    devices = [0, 1, 2, 3]
    n_gpus = len(devices)
    data = utils.process_data(data)
    g = data[0]
    n_classes = data.num_classes
    train_g = val_g = test_g = g

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # Pack data
    data = n_classes, train_g, val_g, test_g

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for proc_id in range(n_gpus):
        p = ctx.Process(
            target=run,
            args=(result_queue, proc_id, n_gpus, args, devices, data),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    time_records = result_queue.get(block=False)
    num_exclude = 10  # exclude first 10 iterations
    if len(time_records) < 15:
        # exclude less if less records
        num_exclude = int(len(time_records) * 0.3)
    return np.mean(time_records[num_exclude:])
