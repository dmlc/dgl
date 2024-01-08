"""
This script trains and tests a GraphSAGE model for node classification on
large graphs using efficient neighbor sampling.

Paper: [Inductive Representation Learning on Large Graphs]
(https://arxiv.org/abs/1706.02216)

Before reading this example, please familiar yourself with graphsage node
classification by reading the example in the
`examples/core/graphsage/node_classification.py`

If you want to train graphsage on a large graph in a distributed fashion, read
the example in the `examples/distributed/graphsage/`.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> Load and preprocess dataset
│
├───> Instantiate SAGE model
│
├───> train
│     │
│     ├───> NeighborSampler (HIGHLIGHT)
│     │
│     └───> Training loop
│           │
│           └───> SAGE.forward
│
└───> layerwise_infer
      │
      └───> SAGE.inference
            │
            └───> MultiLayerFullNeighborSampler (HIGHLIGHT)
"""

import argparse
import time

import dgl
import dgl.nn as dglnn
import torch
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
from ogb.nodeproppred import DglNodePropPredDataset


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

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x

    def inference(self, g, device, batch_size, fused_sampling: bool = True):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        #####################################################################
        # (HIGHLIGHT) Creating a MultiLayerFullNeighborSampler instance.
        # This sampler is used in the Graph Neural Networks (GNN) training
        # process to provide neighbor sampling, which is crucial for
        # efficient training of GNN on large graphs.
        #
        # The first argument '1' indicates the number of layers for
        # the neighbor sampling. In this case, it's set to 1, meaning
        # only the direct neighbors of each node will be included in the
        # sampling.
        #
        # The 'prefetch_node_feats' parameter specifies the node features
        # that need to be pre-fetched during sampling. In this case, the
        # feature named 'feat' will be pre-fetched.
        #
        # `prefetch` in DGL initiates data fetching operations in parallel
        # with model computations. This ensures data is ready when the
        # computation needs it, thereby eliminating waiting times between
        # fetching and computing steps and reducing the I/O overhead during
        # the training process.
        #
        # The difference between whether to use prefetch or not is shown:
        #
        # Without Prefetch:
        # Fetch1 ──> Compute1 ──> Fetch2 ──> Compute2 ──> Fetch3 ──> Compute3
        #
        # With Prefetch:
        # Fetch1 ──> Fetch2 ──> Fetch3
        #    │          │          │
        #    └─Compute1 └─Compute2 └─Compute3
        #####################################################################
        sampler = MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"], fused=fused_sampling
        )

        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        # Enable pin_memory for faster CPU to GPU data transfer if the
        # model is running on a GPU.
        pin_memory = buffer_device != device

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1
            y = torch.empty(
                g.num_nodes(),
                self.out_size if is_last_layer else self.hidden_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                hidden_x = layer(blocks[0], x)  # len(blocks) = 1
                if layer_idx != len(self.layers) - 1:
                    hidden_x = F.relu(hidden_x)
                    hidden_x = self.dropout(hidden_x)
                # By design, our output nodes are contiguous.
                y[output_nodes[0] : output_nodes[-1] + 1] = hidden_x.to(
                    buffer_device
                )
            feat = y
        return y


@torch.no_grad()
def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata["feat"]
        ys.append(blocks[-1].dstdata["label"])
        y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


@torch.no_grad()
def layerwise_infer(
    device, graph, nid, model, num_classes, batch_size, fused_sampling
):
    model.eval()
    pred = model.inference(
        graph, device, batch_size, fused_sampling
    )  # pred in buffer_device.
    pred = pred[nid]
    label = graph.ndata["label"][nid].to(pred.device)
    return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)


def train(device, g, dataset, model, num_classes, use_uva, fused_sampling):
    # Create sampler & dataloader.
    train_idx = dataset.train_idx.to(g.device if not use_uva else device)
    val_idx = dataset.val_idx.to(g.device if not use_uva else device)
    #####################################################################
    # (HIGHLIGHT) Instantiate a NeighborSampler object for efficient
    # training of Graph Neural Networks (GNNs) on large-scale graphs.
    #
    # The argument [10, 10, 10] sets the number of neighbors (fanout)
    # to be sampled at each layer. Here, we have three layers, and
    # 10 neighbors will be randomly selected for each node at each
    # layer.
    #
    # The 'prefetch_node_feats' and 'prefetch_labels' parameters
    # specify the node features and labels that need to be pre-fetched
    # during sampling. More details about `prefetch` can be found in the
    # `SAGE.inference` function.
    #####################################################################
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
        fused=fused_sampling,
    )

    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        # If `g` is on gpu or `use_uva` is True, `num_workers` must be zero,
        # otherwise it will cause error.
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        # No need to shuffle for validation.
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        t0 = time.time()
        model.train()
        total_loss = 0
        # A block is a graph consisting of two sets of nodes: the
        # source nodes and destination nodes. The source and destination
        # nodes can have multiple node types. All the edges connect from
        # source nodes to destination nodes.
        # For more details: https://discuss.dgl.ai/t/what-is-the-block/2932.
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # The input features from the source nodes in the first layer's
            # computation graph.
            x = blocks[0].srcdata["feat"]

            # The ground truth labels from the destination nodes
            # in the last layer's computation graph.
            y = blocks[-1].dstdata["label"]

            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        t1 = time.time()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | "
            f"Accuracy {acc.item():.4f} | Time {t1 - t0:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "gpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for "
        "CPU-GPU mixed training, 'gpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--compare-to-graphbolt",
        default="false",
        choices=["false", "true"],
        help="Whether comparing to GraphBolt or not, 'false' by default.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # Load and preprocess dataset.
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))

    g = dataset[0]
    if args.compare_to_graphbolt == "false":
        g = g.to("cuda" if args.mode == "gpu" else "cpu")
    num_classes = dataset.num_classes
    # Whether use Unified Virtual Addressing (UVA) for CUDA computation.
    use_uva = args.mode == "mixed"
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    fused_sampling = args.compare_to_graphbolt == "false"

    # Create GraphSAGE model.
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # Model training.
    print("Training...")
    train(device, g, dataset, model, num_classes, use_uva, fused_sampling)

    # Test the model.
    print("Testing...")
    acc = layerwise_infer(
        device,
        g,
        dataset.test_idx,
        model,
        num_classes,
        batch_size=4096,
        fused_sampling=fused_sampling,
    )
    print(f"Test accuracy {acc.item():.4f}")
