"""
This script trains and tests a GraphSAGE model for node classification on
multiple GPUs using distributed data-parallel training (DDP) and GraphBolt
data loader. 

Before reading this example, please familiar yourself with graphsage node
classification using GtaphBolt data loader by reading the example in the
`examples/graphbolt/node_classification.py`.

For the usage of DDP provided by PyTorch, please read its documentation:
https://pytorch.org/tutorials/beginner/dist_overview.html and
https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParal
lel.html

This flowchart describes the main functional sequence of the provided example:
main
│
├───> OnDiskDataset pre-processing
│
└───> run (multiprocessing) 
      │
      ├───> Init process group and build distributed SAGE model (HIGHLIGHT)
      │
      ├───> train
      │     │
      │     ├───> Get GraphBolt dataloader with DistributedItemSampler
      │     │     (HIGHLIGHT)
      │     │
      │     └───> Training loop
      │           │
      │           ├───> SAGE.forward
      │           │
      │           ├───> Validation set evaluation
      │           │
      │           └───> Collect accuracy and loss from all ranks (HIGHLIGHT)
      │
      └───> Test set evaluation
"""
import argparse
import os
import time

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP


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
        self.set_layer_dtype(torch.float32)

    def set_layer_dtype(self, dtype):
        for layer in self.layers:
            for param in layer.parameters():
                param.data = param.data.to(dtype)

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x


def create_dataloader(
    args,
    graph,
    features,
    itemset,
    device,
    is_train,
):
    ############################################################################
    # [HIGHLIGHT]
    # Get a GraphBolt dataloader for node classification tasks with multi-gpu
    # distributed training. DistributedItemSampler instead of ItemSampler should
    # be used.
    ############################################################################

    ############################################################################
    # [Note]:
    # gb.DistributedItemSampler()
    # [Input]:
    # 'item_set': The current dataset. (e.g. `train_set` or `valid_set`)
    # 'batch_size': Specifies the number of samples to be processed together,
    # referred to as a 'mini-batch'. (The term 'mini-batch' is used here to
    # indicate a subset of the entire dataset that is processed together. This
    # is in contrast to processing the entire dataset, known as a 'full batch'.)
    # 'drop_last': Determines whether the last non-full minibatch should be
    # dropped.
    # 'shuffle': Determines if the items should be shuffled.
    # 'num_replicas': Specifies the number of replicas.
    # 'drop_uneven_inputs': Determines whether the numbers of minibatches on all
    # ranks should be kept the same by dropping uneven minibatches.
    # [Output]:
    # An DistributedItemSampler object for handling mini-batch sampling on
    # multiple replicas.
    ############################################################################
    datapipe = gb.DistributedItemSampler(
        item_set=itemset,
        batch_size=args.batch_size,
        drop_last=is_train,
        shuffle=is_train,
        drop_uneven_inputs=is_train,
    )
    ############################################################################
    # [Note]:
    # datapipe.copy_to() / gb.CopyTo()
    # [Input]:
    # 'device': The specified device that data should be copied to.
    # [Output]:
    # A CopyTo object copying data in the datapipe to a specified device.\
    ############################################################################
    if args.storage_device != "cpu":
        datapipe = datapipe.copy_to(device)
    datapipe = datapipe.sample_neighbor(
        graph,
        args.fanout,
        overlap_fetch=args.storage_device == "pinned",
        asynchronous=args.storage_device != "cpu",
    )
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])
    if args.storage_device == "cpu":
        datapipe = datapipe.copy_to(device)

    dataloader = gb.DataLoader(datapipe, args.num_workers)

    # Return the fully-initialized DataLoader object.
    return dataloader


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


@torch.no_grad()
def evaluate(rank, model, dataloader, num_classes, device):
    model.eval()
    y = []
    y_hats = []

    for data in tqdm.tqdm(dataloader) if rank == 0 else dataloader:
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


def train(
    rank,
    args,
    train_dataloader,
    valid_dataloader,
    num_classes,
    model,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()
        total_loss = torch.tensor(0, dtype=torch.float, device=device)
        num_train_items = 0
        ########################################################################
        # (HIGHLIGHT) Use Join Context Manager to solve uneven input problem.
        #
        # The mechanics of Distributed Data Parallel (DDP) training in PyTorch
        # requires the number of inputs are the same for all ranks, otherwise
        # the program may error or hang. To solve it, PyTorch provides Join
        # Context Manager. Please refer to
        # https://pytorch.org/tutorials/advanced/generic_join.html for detailed
        # information.
        #
        # Another method is to set `drop_uneven_inputs` as True in GraphBolt's
        # DistributedItemSampler, which will solve this problem by dropping
        # uneven inputs.
        ########################################################################
        with Join([model]):
            for data in (
                tqdm.tqdm(train_dataloader) if rank == 0 else train_dataloader
            ):
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
            valid_dataloader,
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


def run(rank, world_size, args, devices, dataset):
    # Set up multiprocessing environment.
    device = devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )

    # Pin the graph and features to enable GPU access.
    if args.storage_device == "pinned":
        graph = dataset.graph.pin_memory_()
        feature = dataset.feature.pin_memory_()
    else:
        graph = dataset.graph.to(args.storage_device)
        feature = dataset.feature.to(args.storage_device)

    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    args.fanout = list(map(int, args.fanout.split(",")))
    num_classes = dataset.tasks[0].metadata["num_classes"]

    in_size = feature.size("node", None, "feat")[0]
    hidden_size = 256
    out_size = num_classes

    if args.gpu_cache_size > 0 and args.storage_device != "cuda":
        feature[("node", None, "feat")] = gb.gpu_cached_feature(
            feature[("node", None, "feat")],
            args.gpu_cache_size,
        )

    # Create GraphSAGE model. It should be copied onto a GPU as a replica.
    model = SAGE(in_size, hidden_size, out_size).to(device)
    model = DDP(model)

    # Create data loaders.
    train_dataloader = create_dataloader(
        args,
        graph,
        feature,
        train_set,
        device,
        is_train=True,
    )
    valid_dataloader = create_dataloader(
        args,
        graph,
        feature,
        valid_set,
        device,
        is_train=False,
    )
    test_dataloader = create_dataloader(
        args,
        graph,
        feature,
        test_set,
        device,
        is_train=False,
    )

    # Model training.
    if rank == 0:
        print("Training...")
    train(
        rank,
        args,
        train_dataloader,
        valid_dataloader,
        num_classes,
        model,
        device,
    )

    # Test the model.
    if rank == 0:
        print("Testing...")
    test_acc, num_test_items = evaluate(
        rank,
        model,
        test_dataloader,
        num_classes,
        device,
    )
    test_acc = weighted_reduce(test_acc * num_test_items, num_test_items)

    if rank == 0:
        print(f"Test Accuracy {test_acc.item():.4f}")
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script does a multi-gpu training on a GraphSAGE model "
        "for node classification using GraphBolt dataloader."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for optimization.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for training."
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="10,10,10",
        help="Fan-out of neighbor sampling. It is IMPORTANT to keep len(fanout)"
        " identical with the number of layers in your model. Default: 10,10,10",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="The number of processes."
    )
    parser.add_argument(
        "--gpu-cache-size",
        type=int,
        default=0,
        help="The capacity of the GPU cache in bytes.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"],
        help="The dataset we can use for node classification example. Currently"
        " ogbn-products, ogbn-arxiv, ogbn-papers100M datasets are supported.",
    )
    parser.add_argument(
        "--mode",
        default="pinned-cuda",
        choices=["cpu-cuda", "pinned-cuda", "cuda-cuda"],
        help="Dataset storage placement and Train device: 'cpu' for CPU and RAM"
        ", 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        print(f"Multi-gpu training needs to be in gpu mode.")
        exit(0)
    args.storage_device, _ = args.mode.split("-")

    devices = list(map(int, args.gpu.split(",")))
    world_size = len(devices)

    print(f"Training with {world_size} gpus.")

    # Load and preprocess dataset.
    dataset = gb.BuiltinDataset(args.dataset).load()

    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)

    mp.set_sharing_strategy("file_system")
    mp.spawn(
        run,
        args=(world_size, args, devices, dataset),
        nprocs=world_size,
        join=True,
    )
