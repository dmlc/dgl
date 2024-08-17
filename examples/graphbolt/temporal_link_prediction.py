"""
This script trains and tests a Heterogeneous GraphSAGE model for link
prediction with temporal information using graphbolt dataloader.

While node classification predicts labels for nodes based on their
local neighborhoods, link prediction assesses the likelihood of an edge
existing between two nodes, necessitating different sampling strategies
that account for pairs of nodes and their joint neighborhoods.

An additional temporal attribute is provided in both graph and TVT sets,
ensuring that during sampling, only neighbors whose timestamps are earlier
than the seed timestamp will be sampled.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> OnDiskDataset pre-processing
│
├───> Instantiate HeteroSAGE model
│
├───> train
│     │
│     ├───> Get graphbolt dataloader (HIGHLIGHT)
│     │
│     └───> Training loop
│           │
│           ├───> HeteroSAGE.forward
│           │
│           └───> Validation set evaluation
│
└───> Test set evaluation
"""
import argparse
import os
import time

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data.utils import download, extract_archive


TIMESTAMP_FEATURE_NAME = "__timestamp__"
NODE_FEATURE_KEYS = {
    "Product": ["categoryId"],
    "Query": ["categoryId"],
}

TARGET_TYPE = ("Query", "Click", "Product")
ALL_TYPES = [
    TARGET_TYPE,
    ("Product", "reverse_Click", "Query"),
    ("Product", "reverse_QueryResult", "Query"),
    ("Query", "QueryResult", "Product"),
]


class CategoricalEncoder(nn.Module):
    def __init__(
        self,
        num_categories,
        out_size,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_categories, out_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, input_feat: torch.Tensor):
        return self.embed(input_feat.view(-1))


class HeteroSAGE(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()
        sizes = [in_size, hidden_size]
        for size in sizes:
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        etype: dglnn.SAGEConv(
                            size,
                            hidden_size,
                            "mean",
                        )
                        for etype in ALL_TYPES
                    },
                    aggregate="sum",
                )
            )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, blocks, X_node_dict):
        H_node_dict = X_node_dict
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            H_node_dict = layer(block, H_node_dict)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                H_node_dict = {
                    ntype: F.relu(H) for ntype, H in H_node_dict.items()
                }
        return H_node_dict


def create_dataloader(args, graph, features, itemset, is_train=True):
    datapipe = gb.ItemSampler(
        itemset,
        batch_size=args.train_batch_size if is_train else args.eval_batch_size,
        shuffle=is_train,
    )

    if args.storage_device != "cpu":
        datapipe = datapipe.copy_to(device=args.device)

    ############################################################################
    # [Input]:
    # 'datapipe' is either 'ItemSampler' or 'UniformNegativeSampler' depending
    # on whether training is needed ('is_train'),
    # 'graph': The network topology for sampling.
    # 'args.fanout': Number of neighbors to sample per node.
    # [Output]:
    # A NeighborSampler object to sample neighbors.
    # [Role]:
    # Initialize a neighbor sampler for sampling the neighborhoods of nodes with
    # considering of temporal information. Only neighbors that is earlier than
    # the seed will be sampled.
    ############################################################################
    datapipe = getattr(datapipe, args.sample_mode)(
        graph,
        args.fanout if is_train else [-1],
        node_timestamp_attr_name=TIMESTAMP_FEATURE_NAME,
        edge_timestamp_attr_name=TIMESTAMP_FEATURE_NAME,
    )

    datapipe = datapipe.fetch_feature(
        features, node_feature_keys=NODE_FEATURE_KEYS
    )

    if args.storage_device == "cpu":
        datapipe = datapipe.copy_to(device=args.device)

    dataloader = gb.DataLoader(
        datapipe,
        num_workers=args.num_workers,
    )

    # Return the fully-initialized DataLoader object.
    return dataloader


def train(args, model, graph, features, train_set, encoders):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloader = create_dataloader(args, graph, features, train_set)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_epoch_time = time.time()
        for step, data in tqdm.tqdm(enumerate(dataloader)):
            # Get node pairs with labels for loss calculation.
            compacted_seeds = data.compacted_seeds[
                gb.etype_tuple_to_str(TARGET_TYPE)
            ].T
            labels = data.labels

            node_feature = {}
            for ntype, keys in NODE_FEATURE_KEYS.items():
                ntype, feat = ntype, keys[0]
                node_feature[ntype] = data.node_features[
                    (ntype, feat)
                ].squeeze()

            blocks = data.blocks

            # Get the embeddings of the input nodes.
            X_node_dict = {
                ntype: encoders[ntype](feat)
                for ntype, feat in node_feature.items()
            }
            X_node_dict = model(blocks, X_node_dict)
            src_type, _, dst_type = TARGET_TYPE
            logits = model.predictor(
                X_node_dict[src_type][compacted_seeds[0]]
                * X_node_dict[dst_type][compacted_seeds[1]]
            ).squeeze()

            # Compute loss.
            loss = F.binary_cross_entropy_with_logits(
                logits, labels[gb.etype_tuple_to_str(TARGET_TYPE)].float()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step + 1 == args.early_stop:
                # Early stopping requires a new dataloader to reset its state.
                dataloader = create_dataloader(args, graph, features, train_set)
                break

        end_epoch_time = time.time()
        print(
            f"Epoch {epoch:05d} | "
            f"Loss {(total_loss) / (step + 1):.4f} | "
            f"Time {(end_epoch_time - start_epoch_time):.4f} s"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="diginetica-r2ne (GraphBolt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dataset",
        default="diginetica-r2ne",
        choices=["diginetica-r2ne"],
        help="Dataset.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="0 means no early stop, otherwise stop at the input-th step",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="20,20",
        help="Fan-out of neighbor sampling. Default: 20, 20",
    )
    parser.add_argument(
        "--exclude-edges",
        type=int,
        default=1,
        help="Whether to exclude reverse edges during sampling. Default: 1",
    )
    parser.add_argument(
        "--mode",
        default="cpu-cuda",
        choices=["cpu-cpu", "cpu-cuda", "cuda-cuda"],
        help="Dataset storage placement and Train device: 'cpu' for CPU and RAM,"
        " 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
    )
    parser.add_argument(
        "--sample-mode",
        default="temporal_sample_neighbor",
        choices=["temporal_sample_neighbor", "temporal_sample_layer_neighbor"],
        help="The sampling function when doing layerwise sampling.",
    )
    return parser.parse_args()


def download_datasets(name, root="datasets"):
    url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/"
    dataset_dir = os.path.join(root, name)
    if not os.path.exists(dataset_dir):
        url += name + ".zip"
        os.makedirs(root, exist_ok=True)
        zip_file_path = os.path.join(root, name + ".zip")
        download(url, path=zip_file_path)
        extract_archive(zip_file_path, root, overwrite=True)
        os.remove(zip_file_path)
    return dataset_dir


def main(args):
    if not torch.cuda.is_available():
        args.mode = "cpu-cpu"
    print(f"Training in {args.mode} mode.")
    args.storage_device, args.device = args.mode.split("-")
    args.device = torch.device(args.device)

    # Load and preprocess dataset.
    print("Loading data")
    # TODO: Add the datasets to built-in.
    dataset_path = download_datasets(args.dataset)
    dataset = gb.OnDiskDataset(dataset_path).load()

    # Move the dataset to the selected storage.
    graph = dataset.graph.to(args.storage_device)
    features = dataset.feature.to(args.storage_device)

    train_set = dataset.tasks[0].train_set
    # TODO: Fix the dataset so that this modification is not needed. node_pairs
    # needs to be cast into graph.indices.dtype, which is int32.
    train_set._itemsets["Query:Click:Product"]._items = tuple(
        item.to(graph.indices.dtype if i == 0 else None)
        for i, item in enumerate(
            train_set._itemsets["Query:Click:Product"]._items
        )
    )

    args.fanout = list(map(int, args.fanout.split(",")))

    in_size = 128
    hidden_channels = 256
    query_size = features.metadata("node", "Query", "categoryId")[
        "num_categories"
    ]
    product_size = features.metadata("node", "Product", "categoryId")[
        "num_categories"
    ]
    args.device = torch.device(args.device)
    model = HeteroSAGE(in_size, hidden_channels).to(args.device)
    encoders = {
        "Query": CategoricalEncoder(query_size, in_size).to(args.device),
        "Product": CategoricalEncoder(product_size, in_size).to(args.device),
    }

    # Model training.
    print("Training...")
    train(args, model, graph, features, train_set, encoders)


if __name__ == "__main__":
    args = parse_args()
    main(args)
