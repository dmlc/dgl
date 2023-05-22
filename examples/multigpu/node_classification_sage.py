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


"""
This flowchart describes the main functional sequence of the provided example.
main
│
└───> run
      │
      ├───> train
      │     │
      │     ├───> evaluate
      │     │
      │     └───> SAGE.forward
      │
      └───> layerwise_infer
            │
            └───> SAGE.inference
"""


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.hid_size = hid_size
        self.out_size = out_size

        # Three-layer GraphSAGE-mean.
        self.layers = nn.ModuleList(
            [
                dglnn.SAGEConv(in_size, hid_size, "mean"),
                dglnn.SAGEConv(hid_size, hid_size, "mean"),
                dglnn.SAGEConv(hid_size, out_size, "mean"),
            ]
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        """
        Forward propagation of the SAGE model.

        Parameters:
        -----------
        blocks : list of dgl.Block objects
            List of blocks.

            A block is a graph consisting of two sets of nodes: the
            source nodes and destination nodes. The source and destination
            nodes can have multiple node types. All the edges connect from
            source nodes to destination nodes.
            For more details: https://discuss.dgl.ai/t/what-is-the-block/2932.

        x : torch.Tensor
            Initial node features.

        Returns:
        --------
        torch.Tensor
            Output feature tensor.
        """
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # If not the last layer.
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
                use_ddp=True,
                use_uva=use_uva,
            )
            # In order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory.
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
                # non_blocking (with pinned memory) to accelerate data transfer.
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # Make sure all GPUs are done writing to 'y'.
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y


def evaluate(model, g, num_classes, dataloader):
    """
    The evaluation function for the model's performance on a given DataLoader.
    This function is used during the training phase to assess the model's
    performance on the validation set.

    Parameters:
    ----------
    model : SAGE
        The model to be evaluated.

    g : DGLGraph
        The graph on which the model is trained and evaluated.

    num_classes : int
        Number of output classes.

    dataloader : DataLoader
        DataLoader to fetch data during the evaluation.

    Returns:
    --------
    float
        The model's accuracy on the given DataLoader.
    """

    model.eval()
    # Initialize the list to store true labels.
    ys = []
    # Initialize the list to store predicted labels.
    y_hats = []

    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))

    # Concatenate all true labels and all predictions, then calculate
    # the accuracy of the predictions using the function MF.accuracy.
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
        # nid: Node IDs on which the inference is performed. (test_idx)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        print("Test accuracy {:.4f}".format(acc.item()))


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
    """
    Training function for GraphSAGE model.

    Parameters:
    -----------
    proc_id : int
        Process ID. This will be a unique number for each spawned process.

    nprocs : int
        Total number of processes involved in the distributed training.

    device : int
        The device (e.g., 0) on which to perform computations.

    g : dgl.DGLGraph
        The input graph on which to perform node classification.

    num_classes : int
        The number of classes for the classification task.

    train_idx : LongTensor
        The indices of nodes in the training set.

    val_idx : LongTensor
        The indices of nodes in the validation set.

    model : SAGE
        The GraphSAGE model to train.

    use_uva : bool
        If True, uses Unified Virtual Addressing (UVA) for CUDA computation.

    num_epochs : int
        The number of epochs for which to train the model.

    Returns:
    --------
    None
    """
    # Define the neighborhood sampler. The arguments [10, 10, 10]
    # specify that we sample 10 neighbors at each layer for 3 layers.
    # Prefetching node features and labels for speedup.
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
        use_ddp=True,
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
    # Define the optimizer.
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # Main training loop.
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        for it, (_, _, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        acc = (
            evaluate(model, g, num_classes, val_dataloader).to(device) / nprocs
        )
        t1 = time.time()

        # Reduce validation accuracy from all processes.
        dist.reduce(acc, 0)
        if proc_id == 0:
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | "
                "Time {:.4f}".format(
                    epoch, total_loss / (it + 1), acc.item(), t1 - t0
                )
            )


def run(proc_id, nprocs, devices, g, data, mode, num_epochs):
    """
    Main training function to be run in each spawned process.

    Parameters:
    ----------
    proc_id : int
        Process ID. This will be a unique number for each spawned process.

    nprocs : int
        Total number of processes.

    devices : list[int]
        List of devices (GPUs) available for each process.

    g : DGLGraph
        The input graph.

    data : tuple
        The data needed for training and testing. It contains:
          - The number of classes (int)
          - The training indices (torch.Tensor)
          - The validation indices (torch.Tensor)
          - The test indices (torch.Tensor)

    mode : str
        Training mode. It could be either 'mixed' for CPU-GPU
        mixed training, or 'puregpu' for pure-GPU training.

    num_epochs : int
        Number of training epochs.

    Returns:
    --------
    None
    """
    # The rank of the current process.
    device = devices[proc_id]

    # Set the device for this process.
    torch.cuda.set_device(device)

    # Initialize process group and unpack data for sub-processes.
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )

    # Fetch the data for training/validation/testing.
    num_classes, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    g = g.to(device if mode == "puregpu" else "cpu")

    # Create GraphSAGE model (distributed).
    in_size = g.ndata["feat"].shape[1]
    # Hidden_size: 256
    model = SAGE(in_size, 256, num_classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    # Training + testing.
    # Wether turn on CUDA UVA(Unified Virtual Addressing) optimization.
    use_uva = mode == "mixed"
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
    # After training, perform inference on the test data.
    layerwise_infer(proc_id, device, g, num_classes, test_idx, model, use_uva)

    # Cleanup process group.
    dist.destroy_process_group()


def main(args):
    # Get the device list
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."

    # Load and preprocess dataset.
    print("Loading data")
    dataset = AsNodePredDataset(
        DglNodePropPredDataset(args.dataset_name, root=args.dataset_dir)
    )
    g = dataset[0]
    # Avoid creating certain graph formats in each sub-process to save momory.
    g.create_formats_()
    if args.dataset_name == "ogbn-arxiv":
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)

    # 'data' contain the various pieces of data needed for training and testing.
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )

    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")
    # Spawn multiple processes using 'mp.spawn'.
    # This will start the function 'run' for each process.
    mp.spawn(
        # The function to be run in each spawned process.
        run,
        # The arguments to be passed to the function 'run'.
        args=(nprocs, devices, g, data, args.mode, args.num_epochs),
        # The number of processes to spawn.
        nprocs=nprocs,
    )


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
    main(args)
