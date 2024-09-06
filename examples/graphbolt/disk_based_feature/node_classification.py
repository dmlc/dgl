"""
This example references examples/graphbolt/pyg/labor/node_classification.py
"""

import argparse
import time

from copy import deepcopy

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def accuracy(out, labels):
    assert out.ndim == 2
    assert out.size(0) == labels.size(0)
    assert labels.ndim == 1 or (labels.ndim == 2 and labels.size(1) == 1)
    labels = labels.flatten()
    predictions = torch.argmax(out, 1)
    return (labels == predictions).sum(dtype=torch.float64) / labels.size(0)


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean.
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, out_size, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.out_size = out_size
        # Set the dtype for the layers manually.
        self.set_layer_dtype(torch.float32)

    def set_layer_dtype(self, _dtype):
        for layer in self.layers:
            for param in layer.parameters():
                param.data = param.data.to(_dtype)

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x

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
            for data in tqdm(dataloader):
                # len(blocks) = 1
                hidden_x = layer(data.blocks[0], data.node_features["feat"])
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
                    hidden_x = self.dropout(hidden_x)
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
            # Layer dependency makes it so that the sampled neighborhoods across layers
            # become correlated, reducing the total number of sampled unique nodes in a
            # minibatch, thus reducing the amount of feature data requested.
            "layer_dependency": args.layer_dependency,
            # Batch dependency makes it so that the sampled neighborhoods across minibatches
            # become correlated, reducing the total number of sampled unique nodes across
            # minibatches, thus increasing temporal locality and reducing cache miss rates.
            "batch_dependency": args.batch_dependency,
        }
        if args.sample_mode == "sample_layer_neighbor"
        else {}
    )
    datapipe = getattr(datapipe, args.sample_mode)(
        graph,
        fanout if job != "infer" else [-1],
        overlap_fetch=args.overlap_graph_fetch,
        **kwargs,
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
    return gb.DataLoader(datapipe, num_workers=args.num_workers)


def train_step(minibatch, optimizer, model, loss_fn):
    node_features = minibatch.node_features["feat"]
    labels = minibatch.labels
    optimizer.zero_grad()
    out = model(minibatch.blocks, node_features)
    loss = loss_fn(out, labels)
    num_correct = accuracy(out, labels) * labels.size(0)
    loss.backward()
    optimizer.step()
    return loss.detach(), num_correct, labels.size(0)


def train_helper(
    dataloader,
    model,
    optimizer,
    loss_fn,
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
            minibatch, optimizer, model, loss_fn
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
    gpu_cache_miss_rate_fn,
    cpu_cache_miss_rate_fn,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_model = None
    best_model_acc = 0
    best_model_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc, duration = train_helper(
            train_dataloader,
            model,
            optimizer,
            loss_fn,
            gpu_cache_miss_rate_fn,
            cpu_cache_miss_rate_fn,
            device,
        )
        val_acc = evaluate(
            model,
            valid_dataloader,
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
        acc = accuracy(
            pred[nid.to(pred.device)],
            labels.to(pred.device),
        )
        metrics[split_name] = acc.item()

    return metrics


def evaluate_step(minibatch, model):
    node_features = minibatch.node_features["feat"]
    labels = minibatch.labels
    out = model(minibatch.blocks, node_features)
    num_correct = accuracy(out, labels) * labels.size(0)
    return num_correct, labels.size(0)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    gpu_cache_miss_rate_fn,
    cpu_cache_miss_rate_fn,
    device,
):
    model.eval()
    total_correct = torch.zeros(1, dtype=torch.float64, device=device)
    total_samples = 0
    val_dataloader_tqdm = tqdm(dataloader, "Evaluating")
    for step, minibatch in enumerate(val_dataloader_tqdm):
        num_correct, num_samples = evaluate_step(minibatch, model)
        total_correct += num_correct
        total_samples += num_samples
        if step % 25 == 0:
            val_dataloader_tqdm.set_postfix(
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
    parser.add_argument("--dropout", type=float, default=0.2)
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
        ],
    )
    parser.add_argument("--root", type=str, default="datasets")
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
        "--cpu-cache-size-in-gigabytes",
        type=float,
        default=0,
        help="The capacity of the CPU cache in GiB.",
    )
    parser.add_argument(
        "--gpu-cache-size-in-gigabytes",
        type=float,
        default=0,
        help="The capacity of the GPU cache in GiB.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument(
        "--sample-mode",
        default="sample_neighbor",
        choices=["sample_neighbor", "sample_layer_neighbor"],
        help="The sampling function when doing layerwise sampling.",
    )
    parser.add_argument("--precision", type=str, default="high")
    parser.add_argument("--enable-inference", action="store_true")
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision(args.precision)
    if not torch.cuda.is_available():
        args.mode = "cpu-cpu-cpu"
    print(f"Training in {args.mode} mode.")
    args.graph_device, args.feature_device, args.device = args.mode.split("-")
    args.overlap_feature_fetch = args.feature_device == "pinned"
    args.overlap_graph_fetch = args.graph_device == "pinned"

    """
    Load and preprocess on-disk dataset.
    We inspect the in_memory field of the feature_data in the YAML file and modify
    it to False. This will make sure the feature_data is loaded as DiskBasedFeature.
    """
    print("Loading data...")
    disk_based_feature_keys = None
    if args.cpu_cache_size_in_gigabytes > 0:
        disk_based_feature_keys = [("node", None, "feat")]

    dataset = gb.BuiltinDataset(args.dataset, root=args.root)
    if disk_based_feature_keys is None:
        disk_based_feature_keys = set()
    for feature in dataset.yaml_data["feature_data"]:
        feature_key = (feature["domain"], feature["type"], feature["name"])
        # Set the in_memory setting to False without modifying YAML file.
        if feature_key in disk_based_feature_keys:
            feature["in_memory"] = False
    dataset = dataset.load()

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

    """
    If the CPU cache size is greater than 0, we wrap the DiskBasedFeature to be
    a CPUCachedFeature. This internally manages the CPU feature cache by the
    specified cache replacement policy. This will reduce the amount of data
    transferred during disk read operations for this feature.
    
    Note: It is advised to set the CPU cache size to be at least 4 times the number
    of sampled nodes in a mini-batch, otherwise the feature fetcher might get into
    a deadlock, causing a hang.
    """
    if args.cpu_cache_size_in_gigabytes > 0 and isinstance(
        features[("node", None, "feat")], gb.DiskBasedFeature
    ):
        features[("node", None, "feat")] = gb.cpu_cached_feature(
            features[("node", None, "feat")],
            int(args.cpu_cache_size_in_gigabytes * 1024 * 1024 * 1024),
            args.cpu_feature_cache_policy,
            args.feature_device == "pinned",
        )
        cpu_cached_feature = features[("node", None, "feat")]
        cpu_cache_miss_rate_fn = lambda: cpu_cached_feature.miss_rate
    else:
        cpu_cache_miss_rate_fn = lambda: 1

    """
    If the GPU cache size is greater than 0, we wrap the underlying feature store
    to be a GPUCachedFeature. This will reduce the amount of data transferred during
    host-to-device copy operations for this feature.
    """
    if args.gpu_cache_size_in_gigabytes > 0 and args.feature_device != "cuda":
        features[("node", None, "feat")] = gb.gpu_cached_feature(
            features[("node", None, "feat")],
            int(args.gpu_cache_size_in_gigabytes * 1024 * 1024 * 1024),
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
    model = SAGE(
        in_channels,
        args.num_hidden,
        num_classes,
        len(args.fanout),
        args.dropout,
    ).to(args.device)
    assert len(args.fanout) == len(model.layers)

    best_model = train(
        train_dataloader,
        valid_dataloader,
        model,
        gpu_cache_miss_rate_fn,
        cpu_cache_miss_rate_fn,
        args.device,
    )
    model.load_state_dict(best_model)

    if args.enable_inference:
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
        )
        print("Final accuracy values:")
        print(final_acc)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main()
