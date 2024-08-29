"""
This script trains and tests a GraphSAGE model for link prediction on
large graphs using graphbolt dataloader. It is the PyG counterpart of the
example in `examples/graphbolt/link_prediction.py`.

Paper: [Inductive Representation Learning on Large Graphs]
(https://arxiv.org/abs/1706.02216)

While node classification predicts labels for nodes based on their
local neighborhoods, link prediction assesses the likelihood of an edge
existing between two nodes, necessitating different sampling strategies
that account for pairs of nodes and their joint neighborhoods.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> OnDiskDataset pre-processing
│
├───> Instantiate SAGE model
│
├───> train
│     │
│     ├───> Get graphbolt dataloader (HIGHLIGHT)
|     |
|     |───> Define a PyG GNN model for link prediction (HIGHLIGHT)
│     │
│     └───> Training loop
│           │
│           ├───> SAGE.forward
│
└───> Validation and test set evaluation
"""
import argparse
import time
from functools import partial

import dgl.graphbolt as gb
import torch

# For torch.compile until https://github.com/pytorch/pytorch/issues/121197 is
# resolved.
import torch._inductor.codecache

torch._dynamo.config.cache_size_limit = 32

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torchmetrics.retrieval import RetrievalMRR
from tqdm import tqdm, trange


class GraphSAGE(torch.nn.Module):
    #####################################################################
    # (HIGHLIGHT) Define the GraphSAGE model architecture.
    #
    # - This class inherits from `torch.nn.Module`.
    # - Two convolutional layers are created using the SAGEConv class from PyG.
    # - The forward method defines the computation performed at every call.
    #####################################################################
    def __init__(self, in_size, hidden_size, n_layers):
        super(GraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList()
        sizes = [in_size] + [hidden_size] * n_layers
        for i in range(n_layers):
            self.layers.append(SAGEConv(sizes[i], sizes[i + 1]))
        self.hidden_size = hidden_size
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, subgraphs, x):
        h = x
        for i, (layer, subgraph) in enumerate(zip(self.layers, subgraphs)):
            #####################################################################
            # (HIGHLIGHT) Convert given features to be consumed by a PyG layer.
            #
            #   PyG layers have two modes, bipartite and normal. We slice the
            #   given features to get src and dst features to use the PyG layers
            #   in the more efficient bipartite mode.
            #####################################################################
            h, edge_index, size = subgraph.to_pyg(h)
            h = layer(h, edge_index, size=size)
            if i != len(subgraphs) - 1:
                h = F.relu(h)
        return h

    def inference(self, graph, features, dataloader, storage_device):
        """Conduct layer-wise inference to get all the node embeddings."""
        pin_memory = storage_device == "pinned"
        buffer_device = torch.device("cpu" if pin_memory else storage_device)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1

            y = torch.empty(
                graph.total_num_nodes,
                self.hidden_size,
                dtype=torch.float32,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            for data in tqdm(dataloader, "Inferencing"):
                # len(data.sampled_subgraphs) = 1
                h, edge_index, size = data.sampled_subgraphs[0].to_pyg(
                    data.node_features["feat"]
                )
                hidden_x = layer(h, edge_index, size=size)
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
                # By design, our output nodes are contiguous.
                y[data.seeds[0] : data.seeds[-1] + 1] = hidden_x.to(
                    buffer_device
                )
            if not is_last_layer:
                features.update("node", None, "feat", y)

        return y


def create_dataloader(
    graph, features, itemset, batch_size, fanout, device, job
):
    #####################################################################
    # (HIGHLIGHT) Create a data loader for efficiently loading graph data.
    #
    # - 'ItemSampler' samples mini-batches of node IDs from the dataset.
    # - 'CopyTo' copies the fetched data to the specified device.
    # - 'sample_neighbor' performs neighbor sampling on the graph.
    # - 'FeatureFetcher' fetches node features based on the sampled subgraph.

    #####################################################################
    # Create a datapipe for mini-batch sampling with a specific neighbor fanout.
    # Here, [10, 10, 10] specifies the number of neighbors sampled for each node at each layer.
    # We're using `sample_neighbor` for consistency with DGL's sampling API.
    # Note: GraphBolt offers additional sampling methods, such as `sample_layer_neighbor`,
    # which could provide further optimization and efficiency for GNN training.
    # Users are encouraged to explore these advanced features for potentially improved performance.

    # Initialize an ItemSampler to sample mini-batches from the dataset.
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
    # Sample negative edges.
    if job == "train":
        datapipe = datapipe.sample_uniform_negative(graph, args.neg_ratio)
    # Sample neighbors for each node in the mini-batch.
    datapipe = getattr(datapipe, args.sample_mode)(
        graph,
        fanout if job != "infer" else [-1],
        overlap_fetch=args.overlap_graph_fetch,
        asynchronous=args.graph_device != "cpu",
    )
    if job == "train" and args.exclude_edges:
        datapipe = datapipe.exclude_seed_edges(
            include_reverse_edges=True,
            asynchronous=args.graph_device != "cpu",
        )
    # Copy the data to the specified device.
    if args.feature_device != "cpu" and need_copy:
        datapipe = datapipe.copy_to(device=device)
        need_copy = False
    # Fetch node features for the sampled subgraph.
    datapipe = datapipe.fetch_feature(
        features,
        node_feature_keys=["feat"],
        overlap_fetch=args.overlap_feature_fetch,
    )
    # Copy the data to the specified device.
    if need_copy:
        datapipe = datapipe.copy_to(device=device)
    # Create and return a DataLoader to handle data loading.
    return gb.DataLoader(datapipe, num_workers=args.num_workers)


@torch.compile
def predictions_step(model, h_src, h_dst):
    return model.predictor(h_src * h_dst).squeeze()


def compute_predictions(model, node_emb, seeds, device):
    """Compute the predictions for given source and destination nodes.

    This function computes the predictions for a set of node pairs, dividing the
    task into batches to handle potentially large graphs.
    """

    preds = torch.empty(seeds.shape[0], device=device)
    seeds_src, seeds_dst = seeds.T
    # The constant number is 1001, due to negtive ratio in the `ogbl-citation2`
    # dataset is 1000.
    eval_size = args.eval_batch_size * 1001
    # Loop over node pairs in batches.
    for start in trange(0, seeds_src.shape[0], eval_size, desc="Evaluate"):
        end = min(start + eval_size, seeds_src.shape[0])

        # Fetch embeddings for current batch of source and destination nodes.
        h_src = node_emb[seeds_src[start:end]].to(device, non_blocking=True)
        h_dst = node_emb[seeds_dst[start:end]].to(device, non_blocking=True)

        # Compute prediction scores using the model.
        preds[start:end] = predictions_step(model, h_src, h_dst)
    return preds


@torch.no_grad()
def evaluate(model, graph, features, all_nodes_set, valid_set, test_set):
    """Evaluate the model on validation and test sets."""
    model.eval()

    dataloader = create_dataloader(
        graph,
        features,
        all_nodes_set,
        args.eval_batch_size,
        [-1],
        args.device,
        job="infer",
    )

    # Compute node embeddings for the entire graph.
    node_emb = model.inference(graph, features, dataloader, args.feature_device)
    results = []

    # Loop over both validation and test sets.
    for split in [valid_set, test_set]:
        # Unpack the item set.
        seeds = split._items[0].to(node_emb.device)
        labels = split._items[1].to(node_emb.device)
        indexes = split._items[2].to(node_emb.device)

        preds = compute_predictions(model, node_emb, seeds, indexes.device)
        # Compute MRR values for the current split.
        results.append(RetrievalMRR()(preds, labels, indexes))
    return results


@torch.compile
def train_step(minibatch, optimizer, model):
    node_features = minibatch.node_features["feat"]
    compacted_seeds = minibatch.compacted_seeds.T
    labels = minibatch.labels
    optimizer.zero_grad()
    y = model(minibatch.sampled_subgraphs, node_features)
    logits = model.predictor(
        y[compacted_seeds[0]] * y[compacted_seeds[1]]
    ).squeeze()
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.detach(), labels.size(0)


def train_helper(dataloader, model, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = torch.zeros(1, device=device)  # Accumulator for the total loss
    total_samples = 0  # Accumulator for the total number of samples processed
    start = time.time()
    for step, minibatch in tqdm(enumerate(dataloader), "Training"):
        loss, num_samples = train_step(minibatch, optimizer, model)
        total_loss += loss * num_samples
        total_samples += num_samples
        if step + 1 == args.early_stop:
            break
    train_loss = total_loss / total_samples
    end = time.time()
    return train_loss, end - start


def train(dataloader, model, device):
    #####################################################################
    # (HIGHLIGHT) Train the model for one epoch.
    #
    # - Iterates over the data loader, fetching mini-batches of graph data.
    # - For each mini-batch, it performs a forward pass, computes loss, and
    #   updates the model parameters.
    # - The function returns the average loss and accuracy for the epoch.
    #
    # Parameters:
    #   dataloader: DataLoader that provides mini-batches of graph data.
    #   model: The GraphSAGE model.
    #   device: The device (CPU/GPU) to run the training on.
    #####################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, duration = train_helper(
            dataloader, model, optimizer, device
        )
        print(
            f"Epoch {epoch:02d}, Loss: {train_loss.item():.4f}, "
            f"Time: {duration}s"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Which dataset are you going to use?"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate for optimization.",
    )
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for training."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="0 means no early stop, otherwise stop at the input-th step",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbl-citation2",
        choices=["ogbl-citation2"],
        help="The dataset we can use for link prediction. Currently"
        " only ogbl-citation2 dataset is supported.",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="10,10,10",
        help="Fan-out of neighbor sampling. It is IMPORTANT to keep len(fanout)"
        " identical with the number of layers in your model. Default: 10,10,10",
    )
    parser.add_argument(
        "--exclude-edges",
        type=bool,
        default=True,
        help="Whether to exclude reverse edges during sampling. Default: True",
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
    parser.add_argument(
        "--gpu-cache-size",
        type=int,
        default=0,
        help="The capacity of the GPU cache in bytes.",
    )
    parser.add_argument(
        "--sample-mode",
        default="sample_neighbor",
        choices=["sample_neighbor", "sample_layer_neighbor"],
        help="The sampling function when doing layerwise sampling.",
    )
    parser.add_argument("--precision", type=str, default="high")
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision(args.precision)
    if not torch.cuda.is_available():
        args.mode = "cpu-cpu-cpu"
    print(f"Training in {args.mode} mode.")
    args.graph_device, args.feature_device, args.device = args.mode.split("-")
    args.overlap_feature_fetch = args.feature_device == "pinned"
    args.overlap_graph_fetch = args.graph_device == "pinned"

    # Load and preprocess dataset.
    print("Loading data...")
    dataset = gb.BuiltinDataset(args.dataset).load()

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
    all_nodes_set = dataset.all_nodes_set
    args.fanout = list(map(int, args.fanout.split(",")))

    if args.gpu_cache_size > 0 and args.feature_device != "cuda":
        features._features[("node", None, "feat")] = gb.gpu_cached_feature(
            features._features[("node", None, "feat")],
            args.gpu_cache_size,
        )

    train_dataloader = create_dataloader(
        graph=graph,
        features=features,
        itemset=train_set,
        batch_size=args.train_batch_size,
        fanout=args.fanout,
        device=args.device,
        job="train",
    )

    in_channels = features.size("node", None, "feat")[0]
    hidden_channels = 256
    model = GraphSAGE(in_channels, hidden_channels, len(args.fanout)).to(
        args.device
    )
    assert len(args.fanout) == len(model.layers)

    train(train_dataloader, model, args.device)

    # Test the model.
    print("Testing...")
    valid_mrr, test_mrr = evaluate(
        model,
        graph,
        features,
        all_nodes_set,
        valid_set,
        test_set,
    )
    print(
        f"Validation MRR {valid_mrr.item():.4f}, Test MRR {test_mrr.item():.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main()
