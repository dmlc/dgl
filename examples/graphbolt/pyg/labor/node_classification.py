import argparse
import time

from copy import deepcopy

import dgl.graphbolt as gb
import torch

# Needed until https://github.com/pytorch/pytorch/issues/121197 is resolved to
# use the `--torch-compile` cmdline option reliably.
import torch._inductor.codecache
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from load_dataset import load_dataset
from sage_conv import SAGEConv
from torch.torch_version import TorchVersion
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
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        sizes = [in_size] + [hidden_size] * n_layers
        for i in range(n_layers):
            self.layers.append(SAGEConv(sizes[i], sizes[i + 1]))
        self.linear = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.out_size = out_size

    def forward(self, subgraphs, x):
        h = x
        for layer, subgraph in zip(self.layers, subgraphs):
            h, edge_index, size = convert_to_pyg(h, subgraph)
            h = layer(h, edge_index, size=size)
            h = F.gelu(h)
            h = self.dropout(h)
        return self.linear(h)

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
                hidden_x = F.gelu(hidden_x)
                if is_last_layer:
                    hidden_x = self.linear(hidden_x)
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
    kwargs = (
        {
            "layer_dependency": args.layer_dependency,
            "batch_dependency": args.batch_dependency,
        }
        if args.sample_mode == "sample_layer_neighbor"
        else {}
    )
    datapipe = getattr(datapipe, args.sample_mode)(
        graph, fanout if job != "infer" else [-1], **kwargs
    )
    # Copy the data to the specified device.
    if args.feature_device != "cpu":
        datapipe = datapipe.copy_to(device=device)
    # Fetch node features for the sampled subgraph.
    datapipe = datapipe.fetch_feature(
        features,
        node_feature_keys=["feat"],
        overlap_fetch=args.overlap_feature_fetch,
    )
    # Copy the data to the specified device.
    if args.feature_device == "cpu":
        datapipe = datapipe.copy_to(device=device)
    # Create and return a DataLoader to handle data loading.
    return gb.DataLoader(
        datapipe,
        num_workers=args.num_workers,
        overlap_graph_fetch=args.overlap_graph_fetch,
    )


def train(
    train_dataloader,
    valid_dataloader,
    model,
    multilabel,
    kwargs,
    gpu_cache_miss_rate_fn,
    cpu_cache_miss_rate_fn,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    total_loss = torch.zeros(1, device=device)  # Accumulator for the total loss
    total_correct = 0  # Accumulator for the total number of correct predictions
    total_samples = 0  # Accumulator for the total number of samples processed
    num_batches = 0  # Counter for the number of mini-batches processed

    best_model = None
    best_model_acc = 0
    best_model_epoch = -1

    for epoch in range(args.epochs):
        model.train()  # Set the model to training mode
        start = time.time()
        train_dataloader_tqdm = tqdm(train_dataloader, "Training")
        for minibatch in train_dataloader_tqdm:
            node_features = minibatch.node_features["feat"]
            labels = minibatch.labels
            optimizer.zero_grad()
            out = model(minibatch.sampled_subgraphs, node_features)
            label_dtype = out.dtype if multilabel else None
            loss = criterion(out, labels.to(label_dtype))
            total_loss += loss.detach()
            total_correct += MF.f1_score(out, labels, **kwargs) * labels.size(0)
            total_samples += labels.size(0)
            loss.backward()
            optimizer.step()
            num_batches += 1
            train_dataloader_tqdm.set_postfix(
                {
                    "num_nodes": node_features.size(0),
                    "gpu_cache_miss": gpu_cache_miss_rate_fn(),
                    "cpu_cache_miss": cpu_cache_miss_rate_fn(),
                }
            )
        train_loss = total_loss / num_batches
        train_acc = total_correct / total_samples
        end = time.time()
        val_acc = evaluate(
            model,
            valid_dataloader,
            kwargs,
            gpu_cache_miss_rate_fn,
            cpu_cache_miss_rate_fn,
        )
        if val_acc > best_model_acc:
            best_model_acc = val_acc
            best_model = deepcopy(model.state_dict())
            best_model_epoch = epoch
        print(
            f"Epoch {epoch:02d}, Loss: {train_loss.item():.4f}, "
            f"Approx. Train: {train_acc:.4f}, Approx. Val: {val_acc:.4f}, "
            f"Time: {end - start}s"
        )
        if best_model_epoch + args.early_stopping_patience < epoch:
            break
    return best_model


@torch.no_grad()
def layerwise_infer(
    args,
    graph,
    features,
    itemsets,
    all_nodes_set,
    model,
    kwargs,
):
    model.eval()
    dataloader = create_dataloader(
        graph=graph,
        features=features,
        itemset=all_nodes_set,
        batch_size=args.batch_size,
        fanout=[-1],
        device=args.device,
        job="infer",
    )
    pred = model.inference(graph, features, dataloader, args.feature_device)

    metrics = {}
    for split_name, itemset in itemsets.items():
        nid, labels = itemset[:]
        acc = MF.f1_score(
            pred[nid.to(pred.device)],
            labels.to(pred.device),
            **kwargs,
        )
        metrics[split_name] = acc.item()

    return metrics


@torch.no_grad()
def evaluate(
    model, dataloader, kwargs, gpu_cache_miss_rate_fn, cpu_cache_miss_rate_fn
):
    model.eval()
    y_hats = []
    ys = []
    val_dataloader_tqdm = tqdm(dataloader, "Evaluating")
    for minibatch in val_dataloader_tqdm:
        node_features = minibatch.node_features["feat"]
        labels = minibatch.labels
        out = model(minibatch.sampled_subgraphs, node_features)
        y_hats.append(out)
        ys.append(labels)
        val_dataloader_tqdm.set_postfix(
            {
                "num_nodes": node_features.size(0),
                "gpu_cache_miss": gpu_cache_miss_rate_fn(),
                "cpu_cache_miss": cpu_cache_miss_rate_fn(),
            }
        )

    return MF.f1_score(torch.cat(y_hats), torch.cat(ys), **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Which dataset are you going to use?"
    )
    parser.add_argument(
        "--epochs", type=int, default=9999999, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for optimization.",
    )
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
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
        choices=[
            "ogbn-arxiv",
            "ogbn-products",
            "ogbn-papers100M",
            "reddit",
            "yelp",
            "flickr",
        ],
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="10,10,10",
        help="Fan-out of neighbor sampling. len(fanout) determines the number of"
        " GNN layers in your model. Default: 10,10,10",
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
    parser.add_argument("--layer-dependency", action="store_true")
    parser.add_argument("--batch-dependency", type=int, default=1)
    parser.add_argument(
        "--cpu-feature-cache-policy",
        type=str,
        default="sieve",
        choices=["s3-fifo", "sieve", "lru", "clock"],
        help="The cache policy for the CPU feature cache.",
    )
    parser.add_argument(
        "--num-cpu-cached-features",
        type=int,
        default=0,
        help="The capacity of the CPU cache, the number of features to store.",
    )
    parser.add_argument(
        "--num-gpu-cached-features",
        type=int,
        default=0,
        help="The capacity of the GPU cache, the number of features to store.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument(
        "--sample-mode",
        default="sample_layer_neighbor",
        choices=["sample_neighbor", "sample_layer_neighbor"],
        help="The sampling function when doing layerwise sampling.",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        default=TorchVersion(torch.__version__) < TorchVersion("2.2.0a0"),
        help="Disables torch.compile() on the trained GNN model because it is "
        "enabled by default for torch>=2.2.0 without this option.",
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
    # For now, only sample_layer_neighbor is faster with this option
    args.overlap_graph_fetch = (
        args.sample_mode == "sample_layer_neighbor"
        and args.graph_device == "pinned"
    )

    # Load and preprocess dataset.
    print("Loading data...")
    dataset, multilabel = load_dataset(args.dataset)

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

    feature_index_device = (
        args.feature_device if args.feature_device != "pinned" else None
    )
    feature_num_bytes = (
        features[("node", None, "feat")]
        # Read a single row to query its size in bytes.
        .read(torch.zeros(1, device=feature_index_device).long()).nbytes
    )
    if args.num_cpu_cached_features > 0 and isinstance(
        features[("node", None, "feat")], gb.DiskBasedFeature
    ):
        features[("node", None, "feat")] = gb.CPUCachedFeature(
            features[("node", None, "feat")],
            args.num_cpu_cached_features * feature_num_bytes,
            args.cpu_feature_cache_policy,
            args.feature_device == "pinned",
        )
        cpu_cached_feature = features[("node", None, "feat")]
        cpu_cache_miss_rate_fn = lambda: cpu_cached_feature._feature.miss_rate
    else:
        cpu_cache_miss_rate_fn = lambda: 1
    if args.num_gpu_cached_features > 0 and args.feature_device != "cuda":
        features[("node", None, "feat")] = gb.GPUCachedFeature(
            features[("node", None, "feat")],
            args.num_gpu_cached_features * feature_num_bytes,
        )
        gpu_cached_feature = features[("node", None, "feat")]
        gpu_cache_miss_rate_fn = lambda: gpu_cached_feature._feature.miss_rate
    else:
        gpu_cache_miss_rate_fn = lambda: 1

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
    model = GraphSAGE(
        in_channels,
        args.num_hidden,
        num_classes,
        len(args.fanout),
        args.dropout,
    ).to(args.device)
    assert len(args.fanout) == len(model.layers)
    if not args.disable_torch_compile:
        torch._dynamo.config.cache_size_limit = 32
        model = torch.compile(model, fullgraph=True, dynamic=True)

    kwargs = {
        "num_labels" if multilabel else "num_classes": num_classes,
        "task": "multilabel" if multilabel else "multiclass",
        "validate_args": False,
    }

    best_model = train(
        train_dataloader,
        valid_dataloader,
        model,
        multilabel,
        kwargs,
        gpu_cache_miss_rate_fn,
        cpu_cache_miss_rate_fn,
        args.device,
    )
    model.load_state_dict(best_model)

    # Test the model.
    print("Testing...")
    itemsets = {"train": train_set, "val": valid_set, "test": test_set}
    final_acc = layerwise_infer(
        args,
        graph,
        features,
        itemsets,
        all_nodes_set,
        model,
        kwargs,
    )
    print("Final accuracy values:")
    print(final_acc)


if __name__ == "__main__":
    args = parse_args()
    main()
