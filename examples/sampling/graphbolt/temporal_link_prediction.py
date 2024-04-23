"""
This script trains and tests a Heterogeneous GraphSAGE model for link
prediction with temporal information using graphbolt dataloader.

Unlike previous dgl examples, we've utilized the newly defined dataloader
from GraphBolt. This example will help you grasp how to build an end-to-end
training pipeline using GraphBolt.

While node classification predicts labels for nodes based on their
local neighborhoods, link prediction assesses the likelihood of an edge
existing between two nodes, necessitating different sampling strategies
that account for pairs of nodes and their joint neighborhoods.

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
from functools import partial

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data.utils import download, extract_archive


TIMESTAMP_FEATURE_NAME = "__timestamp__"
node_feature_keys = {
    "Product": ["categoryId"],
    "Query": ["categoryId"],
}

target_type = ("Query", "Click", "Product")
all_connected_types = [
    target_type,
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
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.SAGEConv(
                        in_size,
                        hidden_size,
                        "mean",
                    )
                    for etype in all_connected_types
                },
                aggregate="sum",
            )
        )
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.SAGEConv(
                        hidden_size,
                        hidden_size,
                        "mean",
                    )
                    for etype in all_connected_types
                },
                aggregate="sum",
            )
        )
        self.hidden_size = hidden_size
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
    """Get a GraphBolt version of a dataloader for link prediction tasks. This
    function demonstrates how to utilize functional forms of datapipes in
    GraphBolt. Alternatively, you can create a datapipe using its class
    constructor. For a more detailed tutorial, please read the examples in
    `dgl/notebooks/graphbolt/walkthrough.ipynb`.
    """

    ############################################################################
    # [Input]:
    # 'itemset': The current dataset.
    # 'args.batch_size': Specify the number of samples to be processed together,
    # referred to as a 'mini-batch'. (The term 'mini-batch' is used here to
    # indicate a subset of the entire dataset that is processed together. This
    # is in contrast to processing the entire dataset, known as a 'full batch'.)
    # 'is_train': Determining if data should be shuffled. (Shuffling is
    # generally used only in training to improve model generalization. It's
    # not used in validation and testing as the focus there is to evaluate
    # performance rather than to learn from the data.)
    # [Output]:
    # An ItemSampler object for handling mini-batch sampling.
    # [Role]:
    # Initialize the ItemSampler to sample mini-batche from the dataset.
    ############################################################################
    datapipe = gb.ItemSampler(
        itemset,
        batch_size=args.train_batch_size if is_train else args.eval_batch_size,
        shuffle=is_train,
    )

    ############################################################################
    # [Input]:
    # 'datapipe' is either 'ItemSampler' or 'UniformNegativeSampler' depending
    # on whether training is needed ('is_train'),
    # 'graph': The network topology for sampling.
    # 'args.fanout': Number of neighbors to sample per node.
    # [Output]:
    # A NeighborSampler object to sample neighbors.
    # [Role]:
    # Initialize a neighbor sampler for sampling the neighborhoods of nodes.
    ############################################################################
    datapipe = datapipe.temporal_sample_neighbor(
        graph,
        args.fanout if is_train else [-1],
        node_timestamp_attr_name=TIMESTAMP_FEATURE_NAME,
        edge_timestamp_attr_name=TIMESTAMP_FEATURE_NAME,
    )

    ############################################################################
    # [Input]:
    # 'features': The node features.
    # 'node_feature_keys': The node feature keys (list) to be fetched.
    # [Output]:
    # A FeatureFetcher object to fetch node features.
    # [Role]:
    # Initialize a feature fetcher for fetching features of the sampled
    # subgraphs.
    ############################################################################
    datapipe = datapipe.fetch_feature(
        features, node_feature_keys=node_feature_keys
    )

    ############################################################################
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device.
    ############################################################################
    if args.storage_device == "cpu":
        datapipe = datapipe.copy_to(device=args.device)

    ############################################################################
    # [Input]:
    # 'datapipe': The datapipe object to be used for data loading.
    # 'args.num_workers': The number of processes to be used for data loading.
    # [Output]:
    # A DataLoader object to handle data loading.
    # [Role]:
    # Initialize a multi-process dataloader to load the data in parallel.
    ############################################################################
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
                gb.etype_tuple_to_str(target_type)
            ].T
            labels = data.labels

            node_feature = {}
            for ntype, keys in node_feature_keys.items():
                ntype, feat = ntype, keys[0]
                node_feature[ntype] = data.node_features[
                    (ntype, feat)
                ].squeeze()

            blocks = data.blocks

            # Get the embeddings of the input nodes.
            H_node_dict = {
                ntype: encoders[ntype](feat)
                for ntype, feat in node_feature.items()
            }
            seed_embeds = model(blocks, H_node_dict)
            src_type, _, dst_type = target_type
            logits = model.predictor(
                seed_embeds[src_type][compacted_seeds[0]]
                * seed_embeds[dst_type][compacted_seeds[1]]
            ).squeeze()

            # Compute loss.
            loss = F.binary_cross_entropy_with_logits(
                logits, labels[gb.etype_tuple_to_str(target_type)].float()
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
        default=0,
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
        choices=["cpu-cpu", "cpu-cuda"],
        help="Dataset storage placement and Train device: 'cpu' for CPU and RAM,"
        " 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
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
    dataset_path = download_datasets(args.dataset)
    dataset = gb.OnDiskDataset(dataset_path).load()

    # Move the dataset to the selected storage.
    graph = dataset.graph.to(args.storage_device)
    features = dataset.feature.to(args.storage_device)

    train_set = dataset.tasks[0].train_set
    args.fanout = list(map(int, args.fanout.split(",")))

    in_size = 128
    hidden_channels = 256
    args.device = torch.device(args.device)
    model = HeteroSAGE(in_size, hidden_channels).to(args.device)
    encoders = {
        "Query": CategoricalEncoder(1102, in_size).to(args.device),
        "Product": CategoricalEncoder(1218, in_size).to(args.device),
    }

    # Model training.
    print("Training...")
    train(args, model, graph, features, train_set, encoders)


if __name__ == "__main__":
    args = parse_args()
    main(args)
