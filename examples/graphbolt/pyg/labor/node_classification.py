import argparse
import time

from copy import deepcopy
from functools import partial

import dgl.graphbolt as gb
import torch

# For torch.compile until https://github.com/pytorch/pytorch/issues/121197 is
# resolved.
import torch._inductor.codecache

torch._dynamo.config.cache_size_limit = 32

import torch.nn as nn
import torchmetrics.functional as MF
from load_dataset import load_dataset
from sage_conv import SAGEConv as CustomSAGEConv
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


def accuracy(out, labels):
    assert out.ndim == 2
    assert out.size(0) == labels.size(0)
    assert labels.ndim == 1 or (labels.ndim == 2 and labels.size(1) == 1)
    labels = labels.flatten()
    predictions = torch.argmax(out, 1)
    return (labels == predictions).sum(dtype=torch.float64) / labels.size(0)


class GraphSAGE(torch.nn.Module):
    def __init__(
        self, in_size, hidden_size, out_size, n_layers, dropout, variant
    ):
        super().__init__()
        assert variant in ["original", "custom"]
        self.layers = torch.nn.ModuleList()
        if variant == "custom":
            sizes = [in_size] + [hidden_size] * n_layers
            for i in range(n_layers):
                self.layers.append(CustomSAGEConv(sizes[i], sizes[i + 1]))
            self.linear = nn.Linear(hidden_size, out_size)
            self.activation = nn.GELU()
        else:
            sizes = [in_size] + [hidden_size] * (n_layers - 1) + [out_size]
            for i in range(n_layers):
                self.layers.append(SAGEConv(sizes[i], sizes[i + 1]))
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.variant = variant

    def forward(self, subgraphs, x):
        h = x
        for i, (layer, subgraph) in enumerate(zip(self.layers, subgraphs)):
            h, edge_index, size = subgraph.to_pyg(h)
            h = layer(h, edge_index, size=size)
            if self.variant == "custom":
                h = self.activation(h)
                h = self.dropout(h)
            elif i != len(subgraphs) - 1:
                h = self.activation(h)
        return self.linear(h) if self.variant == "custom" else h

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
                h, edge_index, size = data.sampled_subgraphs[0].to_pyg(
                    data.node_features["feat"]
                )
                hidden_x = layer(h, edge_index, size=size)
                if self.variant == "custom":
                    hidden_x = self.activation(hidden_x)
                    if is_last_layer:
                        hidden_x = self.linear(hidden_x)
                elif not is_last_layer:
                    hidden_x = self.activation(hidden_x)
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
    need_copy = True
    # Copy the data to the specified device.
    if args.graph_device != "cpu" and need_copy:
        datapipe = datapipe.copy_to(device=device)
        need_copy = False
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
        graph,
        fanout if job != "infer" else [-1],
        overlap_fetch=args.overlap_graph_fetch,
        asynchronous=args.graph_device != "cpu",
        **kwargs,
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
def train_step(minibatch, optimizer, model, loss_fn, multilabel, eval_fn):
    node_features = minibatch.node_features["feat"]
    labels = minibatch.labels
    optimizer.zero_grad()
    out = model(minibatch.sampled_subgraphs, node_features)
    label_dtype = out.dtype if multilabel else None
    loss = loss_fn(out, labels.to(label_dtype))
    num_correct = eval_fn(out, labels) * labels.size(0)
    loss.backward()
    optimizer.step()
    return loss.detach(), num_correct, labels.size(0)


def train_helper(
    dataloader,
    model,
    optimizer,
    loss_fn,
    multilabel,
    eval_fn,
    gpu_cache_miss_rate_fn,
    cpu_cache_miss_rate_fn,
    device,
):
    model.train()  # Set the model to training mode
    total_loss = torch.zeros(1, device=device)  # Accumulator for the total loss
    # Accumulator for the total number of correct predictions
    total_correct = torch.zeros(1, dtype=torch.float64, device=device)
    total_samples = 0  # Accumulator for the total number of samples processed
    num_batches = 0  # Counter for the number of mini-batches processed
    start = time.time()
    dataloader = tqdm(dataloader, "Training")
    for step, minibatch in enumerate(dataloader):
        loss, num_correct, num_samples = train_step(
            minibatch, optimizer, model, loss_fn, multilabel, eval_fn
        )
        total_loss += loss
        total_correct += num_correct
        total_samples += num_samples
        num_batches += 1
        if step % 25 == 0:
            # log every 25 steps for performance.
            dataloader.set_postfix(
                {
                    "num_nodes": minibatch.node_ids().size(0),
                    "gpu_cache_miss": gpu_cache_miss_rate_fn(),
                    "cpu_cache_miss": cpu_cache_miss_rate_fn(),
                }
            )
    train_loss = total_loss / num_batches
    train_acc = total_correct / total_samples
    end = time.time()
    return train_loss, train_acc, end - start


def train(
    train_dataloader,
    valid_dataloader,
    model,
    multilabel,
    eval_fn,
    gpu_cache_miss_rate_fn,
    cpu_cache_miss_rate_fn,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    best_model = None
    best_model_acc = 0
    best_model_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc, duration = train_helper(
            train_dataloader,
            model,
            optimizer,
            loss_fn,
            multilabel,
            eval_fn,
            gpu_cache_miss_rate_fn,
            cpu_cache_miss_rate_fn,
            device,
        )
        val_acc = evaluate(
            model,
            valid_dataloader,
            eval_fn,
            gpu_cache_miss_rate_fn,
            cpu_cache_miss_rate_fn,
            device,
        )
        if val_acc > best_model_acc:
            best_model_acc = val_acc
            best_model = deepcopy(model.state_dict())
            best_model_epoch = epoch
        print(
            f"Epoch {epoch:02d}, Loss: {train_loss.item():.4f}, "
            f"Approx. Train: {train_acc.item():.4f}, "
            f"Approx. Val: {val_acc.item():.4f}, "
            f"Time: {duration}s"
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
    eval_fn,
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
        acc = eval_fn(
            pred[nid.to(pred.device)],
            labels.to(pred.device),
        )
        metrics[split_name] = acc.item()

    return metrics


@torch.compile
def evaluate_step(minibatch, model, eval_fn):
    node_features = minibatch.node_features["feat"]
    labels = minibatch.labels
    out = model(minibatch.sampled_subgraphs, node_features)
    num_correct = eval_fn(out, labels) * labels.size(0)
    return num_correct, labels.size(0)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    eval_fn,
    gpu_cache_miss_rate_fn,
    cpu_cache_miss_rate_fn,
    device,
):
    model.eval()
    total_correct = torch.zeros(1, dtype=torch.float64, device=device)
    total_samples = 0
    dataloader = tqdm(dataloader, "Evaluating")
    for step, minibatch in enumerate(dataloader):
        num_correct, num_samples = evaluate_step(minibatch, model, eval_fn)
        total_correct += num_correct
        total_samples += num_samples
        if step % 25 == 0:
            dataloader.set_postfix(
                {
                    "num_nodes": minibatch.node_ids().size(0),
                    "gpu_cache_miss": gpu_cache_miss_rate_fn(),
                    "cpu_cache_miss": cpu_cache_miss_rate_fn(),
                }
            )

    return total_correct / total_samples


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
            "igb-hom-tiny",
            "igb-hom-small",
            "igb-hom-medium",
            "igb-hom-large",
            "igb-hom",
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
        help="Graph storage - feature storage - Train device: 'cpu' for CPU and"
        " RAM, 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
    )
    parser.add_argument("--layer-dependency", action="store_true")
    parser.add_argument("--batch-dependency", type=int, default=1)
    parser.add_argument(
        "--cpu-feature-cache-policy",
        type=str,
        default=None,
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
        "--sage-model-variant",
        default="custom",
        choices=["custom", "original"],
        help="The custom SAGE GNN model provides higher accuracy with lower"
        " runtime performance.",
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
    disk_based_feature_keys = None
    if args.num_cpu_cached_features > 0:
        disk_based_feature_keys = [("node", None, "feat")]
    dataset, multilabel = load_dataset(args.dataset, disk_based_feature_keys)

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
        features[("node", None, "feat")] = gb.cpu_cached_feature(
            features[("node", None, "feat")],
            args.num_cpu_cached_features * feature_num_bytes,
            args.cpu_feature_cache_policy,
            args.feature_device == "pinned",
        )
        cpu_cached_feature = features[("node", None, "feat")]
        cpu_cache_miss_rate_fn = lambda: cpu_cached_feature.miss_rate
    else:
        cpu_cache_miss_rate_fn = lambda: 1
    if args.num_gpu_cached_features > 0 and args.feature_device != "cuda":
        features[("node", None, "feat")] = gb.gpu_cached_feature(
            features[("node", None, "feat")],
            args.num_gpu_cached_features * feature_num_bytes,
        )
        gpu_cached_feature = features[("node", None, "feat")]
        gpu_cache_miss_rate_fn = lambda: gpu_cached_feature.miss_rate
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
        args.sage_model_variant,
    ).to(args.device)
    assert len(args.fanout) == len(model.layers)

    eval_fn = (
        partial(
            # TODO @mfbalin: Find an implementation that does not synchronize.
            MF.f1_score,
            task="multilabel",
            num_labels=num_classes,
            validate_args=False,
        )
        if multilabel
        else accuracy
    )

    best_model = train(
        train_dataloader,
        valid_dataloader,
        model,
        multilabel,
        eval_fn,
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
        eval_fn,
    )
    print("Final accuracy values:")
    print(final_acc)


if __name__ == "__main__":
    args = parse_args()
    main()
