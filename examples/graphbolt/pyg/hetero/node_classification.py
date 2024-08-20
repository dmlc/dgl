"""
This script is a PyG counterpart of ``/examples/graphbolt/rgcn/hetero_rgcn.py``.
"""

import argparse
import time

import dgl.graphbolt as gb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv
from tqdm import tqdm


def accuracy(out, labels):
    assert out.ndim == 2
    assert out.size(0) == labels.size(0)
    assert labels.ndim == 1 or (labels.ndim == 2 and labels.size(1) == 1)
    labels = labels.flatten()
    predictions = torch.argmax(out, 1)
    return (labels == predictions).sum(dtype=torch.float64) / labels.size(0)


def create_dataloader(
    graph,
    features,
    itemset,
    batch_size,
    fanout,
    device,
    job,
):
    """Create a GraphBolt dataloader for training, validation or testing."""

    ###########################################################################
    # Initialize the ItemSampler to sample mini-batches from the dataset.
    # `item_set`:
    #   The set of items to sample from. This is typically the
    #   training, validation or test set.
    # `batch_size`:
    #   The number of nodes to sample in each mini-batch.
    # `shuffle`:
    #   Whether to shuffle the items in the dataset before sampling.
    datapipe = gb.ItemSampler(
        itemset,
        batch_size=batch_size,
        shuffle=(job == "train"),
        drop_last=(job == "train"),
    )
    need_copy = True
    # Copy the data to the specified device.
    if args.graph_device != "cpu" and need_copy:
        datapipe = datapipe.copy_to(device=device)
        need_copy = False
    # Sample neighbors for each node in the mini-batch.
    datapipe = datapipe.sample_neighbor(
        graph,
        fanout if job != "infer" else [-1],
        overlap_fetch=args.overlap_graph_fetch,
        asynchronous=args.graph_device != "cpu",
    )
    # Copy the data to the specified device.
    if args.feature_device != "cpu" and need_copy:
        datapipe = datapipe.copy_to(device=device)
        need_copy = False

    if args.dataset == "ogb-lsc-mag240m":
        node_feature_keys = {
            "paper": ["feat"],
            "author": ["feat"],
            "institution": ["feat"],
        }
    # Fetch node features for the sampled subgraph.
    datapipe = datapipe.fetch_feature(features, node_feature_keys)

    # Copy the data to the specified device.
    if need_copy:
        datapipe = datapipe.copy_to(device=device)
    # Create and return a DataLoader to handle data loading.
    return gb.DataLoader(datapipe, num_workers=args.num_workers)


def convert_to_pyg(h, subgraph):
    #####################################################################
    # (HIGHLIGHT) Convert given features to be consumed by a PyG layer.
    #
    #   We convert the provided sampled edges in CSC format from GraphBolt and
    #   convert to COO via using gb.expand_indptr.
    #####################################################################
    h_dst_dict = {}
    edge_index_dict = {}
    sizes_dict = {}
    for etype, sampled_csc in subgraph.sampled_csc.items():
        src = sampled_csc.indices
        dst = gb.expand_indptr(
            sampled_csc.indptr,
            dtype=src.dtype,
            output_size=src.size(0),
        )
        edge_index = torch.stack([src, dst], dim=0).long()
        dst_size = sampled_csc.indptr.size(0) - 1
        # h and h[:dst_size] correspond to source and destination features resp.
        src_ntype, _, dst_ntype = gb.etype_str_to_tuple(etype)
        h_dst_dict[dst_ntype] = h[dst_ntype][:dst_size]
        edge_index_dict[etype] = edge_index
        sizes_dict[etype] = (h[src_ntype].size(0), dst_size)

    return (h, h_dst_dict), edge_index_dict, sizes_dict


class RelGraphConvLayer(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        ntypes,
        etypes,
        activation,
        dropout=0.0,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation

        ########################################################################
        # (HIGHLIGHT) HeteroGraphConv is a graph convolution operator over
        # heterogeneous graphs. A dictionary is passed where the key is the
        # relation name and the value is the instance of GraphConv. norm="right"
        # is to divide the aggregated messages by each node’s in-degrees, which
        # is equivalent to averaging the received messages. weight=False and
        # bias=False as we will use our own weight matrices defined later.
        ########################################################################
        self.convs = nn.ModuleDict(
            {etype: SimpleConv(aggr="mean") for etype in etypes}
        )

        # Create a separate Linear layer for each relationship. Each
        # relationship has its own weights which will be applied to the node
        # features before performing convolution.
        self.weight = nn.ModuleDict(
            {
                etype: nn.Linear(in_size, out_size, bias=False)
                for etype in etypes
            }
        )

        # Create a separate Linear layer for each node type.
        # loop_weights are used to update the output embedding of each target node
        # based on its own features, thereby allowing the model to refine the node
        # representations. Note that this does not imply the existence of self-loop
        # edges in the graph. It is similar to residual connection.
        self.loop_weights = nn.ModuleDict(
            {ntype: nn.Linear(in_size, out_size, bias=True) for ntype in ntypes}
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, subgraph, x):
        # Create a dictionary of node features for the destination nodes in
        # the graph. We slice the node features according to the number of
        # destination nodes of each type. This is necessary because when
        # incorporating the effect of self-loop edges, we perform computations
        # only on the destination nodes' features. By doing so, we ensure the
        # feature dimensions match and prevent any misuse of incorrect node
        # features.
        (h, h_dst), edge_index, size = convert_to_pyg(x, subgraph)

        h_out = {}
        for etype in edge_index:
            src_ntype, _, dst_ntype = gb.etype_str_to_tuple(etype)
            # h_dst is unused in SimpleConv.
            t = self.convs[etype]((h[src_ntype], h_dst[dst_ntype]), edge_index[etype], size=size[etype])
            t = self.weight[etype](t)
            if dst_ntype in h_out:
                h_out[dst_ntype] += t
            else:
                h_out[dst_ntype] = t

        def _apply(ntype, x):
            # Apply the `loop_weight` to the input node features, effectively
            # acting as a residual connection. This allows the model to refine
            # node embeddings based on its current features.
            x = x + self.loop_weights[ntype](h_dst[ntype])
            return self.dropout(self.activation(x))

        # Apply the function defined above for each node type. This will update
        # the node features using the `loop_weights`, apply the activation
        # function and dropout.
        return {ntype: _apply(ntype, h) for ntype, h in h_out.items()}


class EntityClassify(nn.Module):
    def __init__(self, graph, in_size, hidden_size, out_size, n_layers):
        super(EntityClassify, self).__init__()
        self.layers = nn.ModuleList()
        sizes = [in_size] + [hidden_size] * (n_layers - 1) + [out_size]
        for i in range(n_layers):
            self.layers.append(
                RelGraphConvLayer(
                    sizes[i],
                    sizes[i + 1],
                    graph.node_type_to_id.keys(),
                    graph.edge_type_to_id.keys(),
                    activation=F.relu if i != n_layers - 1 else lambda x: x,
                    dropout=0.5,
                )
            )

    def forward(self, subgraphs, h):
        for layer, subgraph in zip(self.layers, subgraphs):
            h = layer(subgraph, h)
        return h


@torch.compile
def evaluate_step(minibatch, model):
    category = "paper"
    node_features = {
        ntype: feat.float()
        for (ntype, name), feat in minibatch.node_features.items()
        if name == "feat"
    }
    labels = minibatch.labels[category].long()
    out = model(minibatch.sampled_subgraphs, node_features)[category]
    num_correct = accuracy(out, labels) * labels.size(0)
    return num_correct, labels.size(0)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = torch.zeros(1, dtype=torch.float64, device=device)
    total_samples = 0
    for minibatch in tqdm(dataloader, desc="Evaluating"):
        num_correct, num_samples = evaluate_step(minibatch, model)
        total_correct += num_correct
        total_samples += num_samples
    
    return total_correct / total_samples


@torch.compile
def train_step(minibatch, optimizer, model, loss_fn):
    category = "paper"
    node_features = {
        ntype: feat.float()
        for (ntype, name), feat in minibatch.node_features.items()
        if name == "feat"
    }
    labels = minibatch.labels[category].long()
    optimizer.zero_grad()
    out = model(minibatch.sampled_subgraphs, node_features)[category]
    loss = loss_fn(out, labels)
    num_correct = accuracy(out, labels) * labels.size(0)
    loss.backward()
    optimizer.step()
    return loss.detach(), num_correct, labels.size(0)

def train_helper(dataloader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, dtype=torch.float64, device=device)
    total_samples = 0
    start = time.time()
    for minibatch in tqdm(dataloader, "Training"):
        loss, num_correct, num_samples = train_step(minibatch, optimizer, model, loss_fn)
        total_loss += loss * num_samples
        total_correct += num_correct
        total_samples += num_samples
    loss = total_loss / total_samples
    acc = total_correct / total_samples
    end = time.time()
    return loss, acc, end - start

def train(train_dataloader, valid_dataloader, model, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc, duration = train_helper(
            train_dataloader, model, optimizer, loss_fn, device
        )
        val_acc = evaluate(model, valid_dataloader, device)
        print(
            f"Epoch: {epoch:02d}, Loss: {train_loss.item():.4f}, "
            f"Approx. Train: {train_acc.item():.4f}, "
            f"Approx. Val: {val_acc.item():.4f}, "
            f"Time: {duration}s"
        )


def main():
    torch.set_float32_matmul_precision(args.precision)
    if not torch.cuda.is_available():
        args.mode = "cpu-cpu-cpu"
    print(f"Training in {args.mode} mode.")
    args.graph_device, args.feature_device, args.device = args.mode.split("-")
    args.overlap_feature_fetch = args.feature_device == "pinned"
    args.overlap_graph_fetch = args.graph_device == "pinned"

    # Load dataset.
    dataset = gb.BuiltinDataset(args.dataset).load()
    print("Dataset loaded")

    # Move the dataset to the selected storage.
    graph = (
        dataset.graph.pin_memory_()
        if args.graph_device == "pinned"
        else dataset.graph.to(args.graph_device)
    )
    features = (
        dataset.feature.pin_memory_()
        if args.feature_device == "pinned"
        else dataset.feature.to(args.feature_device)
    )

    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    args.fanout = list(map(int, args.fanout.split(",")))

    num_classes = dataset.tasks[0].metadata["num_classes"]
    num_etypes = len(graph.num_edges)

    train_dataloader, valid_dataloader, test_dataloader = (
        create_dataloader(
            graph=graph,
            features=features,
            itemset=itemset,
            batch_size=args.batch_size,
            fanout=[torch.full((num_etypes,), fanout) for fanout in args.fanout],
            device=args.device,
            job=job,
        )
        for itemset, job in zip([train_set, valid_set, test_set], ["train", "evaluate", "evaluate"])
    )

    feat_size = features.size("node", "paper", "feat")[0]
    hidden_channels = 256

    # Initialize the entity classification model.
    model = EntityClassify(graph, feat_size, hidden_channels, num_classes, 3).to(
        args.device
    )

    print(
        "Number of model parameters: "
        f"{sum(p.numel() for p in model.parameters())}"
    )

    train(train_dataloader, valid_dataloader, model, args.device)

    print("Testing...")
    test_acc = evaluate(model, test_dataloader, args.device)
    print(f"Test accuracy {test_acc.item():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="GraphBolt PyG R-SAGE")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate for optimization.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for training."
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogb-lsc-mag240m",
        choices=["ogb-lsc-mag240m"],
        help="Dataset name. Possible values: ogb-lsc-mag240m",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="10,25",
        help="Fan-out of neighbor sampling. It is IMPORTANT to keep len(fanout)"
        " identical with the number of layers in your model. Default: ",
    )
    parser.add_argument(
        "--mode",
        default="pinned-pinned-cuda",
        choices=[
            "cpu-cpu-cpu",
            "cpu-cpu-cuda",
            "cpu-pinned-cuda",
            "pinned-pinned-cuda",
            "cuda-pinned-cuda",
            "cuda-cuda-cuda",
        ],
        help="Graph storage - feature storage - Train device: 'cpu' for CPU and RAM,"
        " 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
    )

    parser.add_argument("--precision", type=str, default="high")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main()