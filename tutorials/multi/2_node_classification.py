"""
Single Machine Multi-GPU Minibatch Node Classification
======================================================

In this tutorial, you will learn how to use multiple GPUs in training a
graph neural network (GNN) for node classification.

This tutorial assumes that you have read the `Stochastic GNN Training for Node
Classification in DGL <../../notebooks/stochastic_training/node_classification.ipynb>`__.
It also assumes that you know the basics of training general
models with multi-GPU with ``DistributedDataParallel``.

.. note::

   See `this tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__
   from PyTorch for general multi-GPU training with ``DistributedDataParallel``.  Also,
   see the first section of :doc:`the multi-GPU graph classification
   tutorial <1_graph_classification>`
   for an overview of using ``DistributedDataParallel`` with DGL.

"""

######################################################################
# Importing Packages
# ---------------
#
# We use ``torch.distributed`` to initialize a distributed training context
# and ``torch.multiprocessing`` to spawn multiple processes for each GPU.
#

import os

os.environ["DGLBACKEND"] = "pytorch"
import time

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm


######################################################################
# Defining Model
# --------------
#
# The model will be again identical to `Stochastic GNN Training for Node
# Classification in DGL <../../notebooks/stochastic_training/node_classification.ipynb>`__.
#


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean.
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hidden_size = hidden_size
        self.out_size = out_size
        # Set the dtype for the layers manually.
        self.float()

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x


######################################################################
# Mini-batch Data Loading
# -----------------------
#
# The major difference from the previous tutorial is that we will use
# ``DistributedItemSampler`` instead of ``ItemSampler`` to sample mini-batches
# of nodes.  ``DistributedItemSampler`` is a distributed version of
# ``ItemSampler`` that works with ``DistributedDataParallel``.  It is
# implemented as a wrapper around ``ItemSampler`` and will sample the same
# minibatch on all replicas.  It also supports dropping the last non-full
# minibatch to avoid the need for padding.
#


def create_dataloader(
    graph,
    features,
    itemset,
    device,
    is_train,
):
    datapipe = gb.DistributedItemSampler(
        item_set=itemset,
        batch_size=1024,
        drop_last=is_train,
        shuffle=is_train,
        drop_uneven_inputs=is_train,
    )
    datapipe = datapipe.copy_to(device)
    # Now that we have moved to device, sample_neighbor and fetch_feature steps
    # will be executed on GPUs.
    datapipe = datapipe.sample_neighbor(graph, [10, 10, 10])
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])
    return gb.DataLoader(datapipe)


def weighted_reduce(tensor, weight, dst=0):
    ########################################################################
    # (HIGHLIGHT) Collect accuracy and loss values from sub-processes and
    # obtain overall average values.
    #
    # `torch.distributed.reduce` is used to reduce tensors from all the
    # sub-processes to a specified process, ReduceOp.SUM is used by default.
    #
    # Because the GPUs may have differing numbers of processed items, we
    # perform a weighted mean to calculate the exact loss and accuracy.
    ########################################################################
    dist.reduce(tensor=tensor, dst=dst)
    weight = torch.tensor(weight, device=tensor.device)
    dist.reduce(tensor=weight, dst=dst)
    return tensor / weight


######################################################################
# Evaluation Loop
# ---------------
#
# The evaluation loop is almost identical to the previous tutorial.
#


@torch.no_grad()
def evaluate(rank, model, graph, features, itemset, num_classes, device):
    model.eval()
    y = []
    y_hats = []
    dataloader = create_dataloader(
        graph,
        features,
        itemset,
        device,
        is_train=False,
    )

    for data in tqdm(dataloader) if rank == 0 else dataloader:
        blocks = data.blocks
        x = data.node_features["feat"]
        y.append(data.labels)
        y_hats.append(model.module(blocks, x))

    res = MF.accuracy(
        torch.cat(y_hats),
        torch.cat(y),
        task="multiclass",
        num_classes=num_classes,
    )

    return res.to(device), sum(y_i.size(0) for y_i in y)


######################################################################
# Training Loop
# -------------
#
# The training loop is also almost identical to the previous tutorial except
# that we use Join Context Manager to solve the uneven input problem. The
# mechanics of Distributed Data Parallel (DDP) training in PyTorch requires
# the number of inputs are the same for all ranks, otherwise the program may
# error or hang. To solve it, PyTorch provides Join Context Manager. Please
# refer to `this tutorial <https://pytorch.org/tutorials/advanced/generic_join.html>`__
# for detailed information.
#


def train(
    rank,
    graph,
    features,
    train_set,
    valid_set,
    num_classes,
    model,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Create training data loader.
    dataloader = create_dataloader(
        graph,
        features,
        train_set,
        device,
        is_train=True,
    )

    for epoch in range(5):
        epoch_start = time.time()

        model.train()
        total_loss = torch.tensor(0, dtype=torch.float, device=device)
        num_train_items = 0
        with Join([model]):
            for data in tqdm(dataloader) if rank == 0 else dataloader:
                # The input features are from the source nodes in the first
                # layer's computation graph.
                x = data.node_features["feat"]

                # The ground truth labels are from the destination nodes
                # in the last layer's computation graph.
                y = data.labels

                blocks = data.blocks

                y_hat = model(blocks, x)

                # Compute loss.
                loss = F.cross_entropy(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.detach() * y.size(0)
                num_train_items += y.size(0)

        # Evaluate the model.
        if rank == 0:
            print("Validating...")
        acc, num_val_items = evaluate(
            rank,
            model,
            graph,
            features,
            valid_set,
            num_classes,
            device,
        )
        total_loss = weighted_reduce(total_loss, num_train_items)
        acc = weighted_reduce(acc * num_val_items, num_val_items)

        # We synchronize before measuring the epoch time.
        torch.cuda.synchronize()
        epoch_end = time.time()
        if rank == 0:
            print(
                f"Epoch {epoch:05d} | "
                f"Average Loss {total_loss.item():.4f} | "
                f"Accuracy {acc.item():.4f} | "
                f"Time {epoch_end - epoch_start:.4f}"
            )


######################################################################
# Defining Traning and Evaluation Procedures
# ------------------------------------------
#
# The following code defines the main function for each process. It is
# similar to the previous tutorial except that we need to initialize a
# distributed training context with ``torch.distributed`` and wrap the model
# with ``torch.nn.parallel.DistributedDataParallel``.
#


def run(rank, world_size, devices, dataset):
    # Set up multiprocessing environment.
    device = devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )

    # Pin the graph and features in-place to enable GPU access.
    graph = dataset.graph.pin_memory_()
    features = dataset.feature.pin_memory_()
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    num_classes = dataset.tasks[0].metadata["num_classes"]

    in_size = features.size("node", None, "feat")[0]
    hidden_size = 256
    out_size = num_classes

    # Create GraphSAGE model. It should be copied onto a GPU as a replica.
    model = SAGE(in_size, hidden_size, out_size).to(device)
    model = DDP(model)

    # Model training.
    if rank == 0:
        print("Training...")
    train(
        rank,
        graph,
        features,
        train_set,
        valid_set,
        num_classes,
        model,
        device,
    )

    # Test the model.
    if rank == 0:
        print("Testing...")
    test_set = dataset.tasks[0].test_set
    test_acc, num_test_items = evaluate(
        rank,
        model,
        graph,
        features,
        itemset=test_set,
        num_classes=num_classes,
        device=device,
    )
    test_acc = weighted_reduce(test_acc * num_test_items, num_test_items)

    if rank == 0:
        print(f"Test Accuracy {test_acc.item():.4f}")


######################################################################
# Spawning Trainer Processes
# --------------------------
#
# The following code spawns a process for each GPU and calls the ``run``
# function defined above.
#


def main():
    if not torch.cuda.is_available():
        print("No GPU found!")
        return

    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]
    world_size = len(devices)

    print(f"Training with {world_size} gpus.")

    # Load and preprocess dataset.
    dataset = gb.BuiltinDataset("ogbn-arxiv").load()

    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)

    mp.set_sharing_strategy("file_system")
    mp.spawn(
        run,
        args=(world_size, devices, dataset),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
