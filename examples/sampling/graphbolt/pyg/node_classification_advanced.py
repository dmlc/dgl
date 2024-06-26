"""
This script demonstrates node classification with GraphSAGE on large graphs, 
merging GraphBolt (GB) and PyTorch Geometric (PyG). GraphBolt efficiently manages 
data loading for large datasets, crucial for mini-batch processing. Post data 
loading, PyG's user-friendly framework takes over for training, showcasing seamless 
integration with GraphBolt. This combination offers an efficient alternative to 
traditional Deep Graph Library (DGL) methods, highlighting adaptability and 
scalability in handling large-scale graph data for diverse real-world applications.



Key Features:
- Implements the GraphSAGE model, a scalable GNN, for node classification on large graphs.
- Utilizes GraphBolt, an efficient framework for large-scale graph data processing.
- Integrates with PyTorch Geometric for building and training the GraphSAGE model.
- The script is well-documented, providing clear explanations at each step.

This flowchart describes the main functional sequence of the provided example.
main: 

main
│
├───> Load and preprocess dataset (GraphBolt)
│     │
│     └───> Utilize GraphBolt's BuiltinDataset for dataset handling
│
├───> Instantiate the SAGE model (PyTorch Geometric)
│     │
│     └───> Define the GraphSAGE model architecture
│
├───> Train the model
│     │
│     ├───> Mini-Batch Processing with GraphBolt
│     │     │
│     │     └───> Efficient handling of mini-batches using GraphBolt's utilities
│     │
│     └───> Training Loop
│           │
│           ├───> Forward and backward passes
│           │
│           └───> Parameters optimization
│
└───> Evaluate the model
      │
      └───> Performance assessment on validation and test datasets
            │
            └───> Accuracy and other relevant metrics calculation


"""

import argparse
import time

import dgl.graphbolt as gb
import torch

# Needed until https://github.com/pytorch/pytorch/issues/121197 is resolved to
# use the `--torch-compile` cmdline option reliably.
import torch._inductor.codecache
import torch.nn.functional as F
import torchmetrics.functional as MF
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


def convert_to_pyg(h, subgraph):
    #####################################################################
    # (HIGHLIGHT) Convert given features to be consumed by a PyG layer.
    #
    #   We convert the provided sampled edges in CSC format from GraphBolt and
    #   convert to COO via using gb.expand_indptr.
    #####################################################################
    src = subgraph.sampled_csc.indices
    dst = gb.expand_indptr(
        subgraph.sampled_csc.indptr,
        dtype=src.dtype,
        output_size=src.size(0),
    )
    edge_index = torch.stack([src, dst], dim=0).long()
    dst_size = subgraph.sampled_csc.indptr.size(0) - 1
    # h and h[:dst_size] correspond to source and destination features resp.
    return (h, h[:dst_size]), edge_index, (h.size(0), dst_size)


class GraphSAGE(torch.nn.Module):
    #####################################################################
    # (HIGHLIGHT) Define the GraphSAGE model architecture.
    #
    # - This class inherits from `torch.nn.Module`.
    # - Two convolutional layers are created using the SAGEConv class from PyG.
    # - 'in_size', 'hidden_size', 'out_size' are the sizes of
    #   the input, hidden, and output features, respectively.
    # - The forward method defines the computation performed at every call.
    #####################################################################
    def __init__(self, in_size, hidden_size, out_size):
        super(GraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hidden_size))
        self.layers.append(SAGEConv(hidden_size, hidden_size))
        self.layers.append(SAGEConv(hidden_size, out_size))
        self.hidden_size = hidden_size
        self.out_size = out_size

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
            h, edge_index, size = convert_to_pyg(h, subgraph)
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
                self.out_size if is_last_layer else self.hidden_size,
                dtype=torch.float32,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            for data in tqdm(dataloader, "Inferencing"):
                # len(data.sampled_subgraphs) = 1
                h, edge_index, size = convert_to_pyg(
                    data.node_features["feat"], data.sampled_subgraphs[0]
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
    # Copy the data to the specified device.
    if args.graph_device != "cpu":
        datapipe = datapipe.copy_to(device=device)
    # Sample neighbors for each node in the mini-batch.
    datapipe = getattr(datapipe, args.sample_mode)(
        graph, fanout if job != "infer" else [-1]
    )
    # Copy the data to the specified device.
    if args.feature_device != "cpu":
        datapipe = datapipe.copy_to(device=device)
    # Fetch node features for the sampled subgraph.
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])
    # Copy the data to the specified device.
    if args.feature_device == "cpu":
        datapipe = datapipe.copy_to(device=device)
    # Create and return a DataLoader to handle data loading.
    return gb.DataLoader(
        datapipe,
        num_workers=args.num_workers,
        overlap_graph_fetch=args.overlap_graph_fetch,
        num_gpu_cached_edges=args.num_gpu_cached_edges,
        gpu_cache_threshold=args.gpu_graph_caching_threshold,
    )


def train(train_dataloader, valid_dataloader, num_classes, model, device):
    #####################################################################
    # (HIGHLIGHT) Train the model for one epoch.
    #
    # - Iterates over the data loader, fetching mini-batches of graph data.
    # - For each mini-batch, it performs a forward pass, computes loss, and
    #   updates the model parameters.
    # - The function returns the average loss and accuracy for the epoch.
    #
    # Parameters:
    #   model: The GraphSAGE model.
    #   dataloader: DataLoader that provides mini-batches of graph data.
    #   optimizer: Optimizer used for updating model parameters.
    #   criterion: Loss function used for training.
    #   device: The device (CPU/GPU) to run the training on.
    #####################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()  # Set the model to training mode
    total_loss = torch.zeros(1, device=device)  # Accumulator for the total loss
    total_correct = 0  # Accumulator for the total number of correct predictions
    total_samples = 0  # Accumulator for the total number of samples processed
    num_batches = 0  # Counter for the number of mini-batches processed

    for epoch in range(args.epochs):
        start = time.time()
        for minibatch in tqdm(train_dataloader, "Training"):
            node_features = minibatch.node_features["feat"]
            labels = minibatch.labels
            optimizer.zero_grad()
            out = model(minibatch.sampled_subgraphs, node_features)
            loss = criterion(out, labels)
            total_loss += loss.detach()
            total_correct += MF.accuracy(
                out, labels, task="multiclass", num_classes=num_classes
            ) * labels.size(0)
            total_samples += labels.size(0)
            loss.backward()
            optimizer.step()
            num_batches += 1
        train_loss = total_loss / num_batches
        train_acc = total_correct / total_samples
        end = time.time()
        val_acc = evaluate(model, valid_dataloader, num_classes)
        print(
            f"Epoch {epoch:02d}, Loss: {train_loss.item():.4f}, "
            f"Approx. Train: {train_acc:.4f}, Approx. Val: {val_acc:.4f}, "
            f"Time: {end - start}s"
        )


@torch.no_grad()
def layerwise_infer(
    args, graph, features, test_set, all_nodes_set, model, num_classes
):
    model.eval()
    dataloader = create_dataloader(
        graph=graph,
        features=features,
        itemset=all_nodes_set,
        batch_size=4 * args.batch_size,
        fanout=[-1],
        device=args.device,
        job="infer",
    )
    pred = model.inference(graph, features, dataloader, args.feature_device)
    pred = pred[test_set._items[0]]
    label = test_set._items[1].to(pred.device)

    return MF.accuracy(
        pred,
        label,
        task="multiclass",
        num_classes=num_classes,
    )


@torch.no_grad()
def evaluate(model, dataloader, num_classes):
    model.eval()
    y_hats = []
    ys = []
    for minibatch in tqdm(dataloader, "Evaluating"):
        node_features = minibatch.node_features["feat"]
        labels = minibatch.labels
        out = model(minibatch.sampled_subgraphs, node_features)
        y_hats.append(out)
        ys.append(labels)

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
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
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"],
        help="The dataset we can use for node classification example. Currently"
        " ogbn-products, ogbn-arxiv, ogbn-papers100M datasets are supported.",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="5,10,15",
        help="Fan-out of neighbor sampling. It is IMPORTANT to keep len(fanout)"
        " identical with the number of layers in your model. Default: 5,10,15",
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
    parser.add_argument(
        "--overlap-graph-fetch",
        action="store_true",
        help="An option for enabling overlap_graph_fetch in graphbolt dataloader."
        "If True, the data loader will overlap the UVA graph fetching operations"
        "with the rest of operations by using an alternative CUDA stream. Disabled"
        "by default.",
    )
    parser.add_argument(
        "--num-gpu-cached-edges",
        type=int,
        default=0,
        help="The number of edges to be cached from the graph on the GPU.",
    )
    parser.add_argument(
        "--gpu-graph-caching-threshold",
        type=int,
        default=1,
        help="The number of accesses after which a vertex neighborhood will be cached.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Uses torch.compile() on the trained GNN model. Requires "
        "torch>=2.2.0 to enable this option.",
    )
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        args.mode = "cpu-cpu-cpu"
    print(f"Training in {args.mode} mode.")
    args.graph_device, args.feature_device, args.device = args.mode.split("-")

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

    num_classes = dataset.tasks[0].metadata["num_classes"]

    if args.gpu_cache_size > 0 and args.feature_device != "cuda":
        features._features[("node", None, "feat")] = gb.GPUCachedFeature(
            features._features[("node", None, "feat")],
            args.gpu_cache_size,
        )

    train_dataloader, valid_dataloader = (
        create_dataloader(
            graph=graph,
            features=features,
            itemset=itemset,
            batch_size=args.batch_size,
            fanout=args.fanout,
            device=args.device,
            job=job,
        )
        for itemset, job in zip([train_set, valid_set], ["train", "evaluate"])
    )

    in_channels = features.size("node", None, "feat")[0]
    hidden_channels = 256
    model = GraphSAGE(in_channels, hidden_channels, num_classes).to(args.device)
    assert len(args.fanout) == len(model.layers)
    if args.torch_compile:
        torch._dynamo.config.cache_size_limit = 32
        model = torch.compile(model, fullgraph=True, dynamic=True)

    train(train_dataloader, valid_dataloader, num_classes, model, args.device)

    # Test the model.
    print("Testing...")
    test_acc = layerwise_infer(
        args,
        graph,
        features,
        test_set,
        all_nodes_set,
        model,
        num_classes,
    )
    print(f"Test accuracy {test_acc.item():.4f}")


if __name__ == "__main__":
    args = parse_args()
    main()
