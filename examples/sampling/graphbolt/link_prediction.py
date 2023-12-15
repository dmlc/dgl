"""
This script trains and tests a GraphSAGE model for link prediction on
large graphs using graphbolt dataloader.

Paper: [Inductive Representation Learning on Large Graphs]
(https://arxiv.org/abs/1706.02216)

Unlike previous dgl examples, we've utilized the newly defined dataloader
from GraphBolt. This example will help you grasp how to build an end-to-end
training pipeline using GraphBolt.

While node classification predicts labels for nodes based on their
local neighborhoods, link prediction assesses the likelihood of an edge
existing between two nodes, necessitating different sampling strategies
that account for pairs of nodes and their joint neighborhoods.

TODO: Add the link_prediction.py example to core/graphsage.
Before reading this example, please familiar yourself with graphsage link
prediction by reading the example in the
`examples/core/graphsage/link_prediction.py`

If you want to train graphsage on a large graph in a distributed fashion, read
the example in the `examples/distributed/graphsage/`.

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
│     │
│     └───> Training loop
│           │
│           ├───> SAGE.forward
│           │
│           └───> Validation set evaluation
│
└───> Test set evaluation
"""
import argparse
import time
from functools import partial

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.linkproppred import Evaluator


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.hidden_size = hidden_size
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
        return hidden_x

    def inference(self, graph, features, dataloader, device):
        """Conduct layer-wise inference to get all the node embeddings."""
        feature = features.read("node", None, "feat")

        buffer_device = torch.device("cpu")
        # Enable pin_memory for faster CPU to GPU data transfer if the
        # model is running on a GPU.
        pin_memory = buffer_device != device

        print("Start node embedding inference.")
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1

            y = torch.empty(
                graph.total_num_nodes,
                self.hidden_size,
                dtype=torch.float32,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feature = feature.to(device)
            for step, data in tqdm.tqdm(enumerate(dataloader)):
                data = data.to_dgl()
                x = feature[data.input_nodes]
                hidden_x = layer(data.blocks[0], x)  # len(blocks) = 1
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
                # By design, our output nodes are contiguous.
                y[
                    data.output_nodes[0] : data.output_nodes[-1] + 1
                ] = hidden_x.to(buffer_device, non_blocking=True)
            feature = y

        return y


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
    # 'args.neg_ratio': Specify the ratio of negative to positive samples.
    # (E.g., if neg_ratio is 1, for each positive sample there will be 1
    # negative sample.)
    # 'graph': The overall network topology for negative sampling.
    # [Output]:
    # A UniformNegativeSampler object that will handle the generation of
    # negative samples for link prediction tasks.
    # [Role]:
    # Initialize the UniformNegativeSampler for negative sampling in link
    # prediction.
    # [Note]:
    # If 'is_train' is False, the UniformNegativeSampler will not be used.
    # Since, in validation and testing, the itemset already contains the
    # negative edges information.
    ############################################################################
    if is_train:
        datapipe = datapipe.sample_uniform_negative(graph, args.neg_ratio)

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
    datapipe = datapipe.sample_neighbor(graph, args.fanout)

    ############################################################################
    # [Input]:
    # 'gb.exclude_seed_edges': Function to exclude seed edges, optionally
    # including their reverse edges, from the sampled subgraphs in the
    # minibatch.
    # [Output]:
    # A MiniBatchTransformer object with excluded seed edges.
    # [Role]:
    # During the training phase of link prediction, negative edges are
    # sampled. It's essential to exclude the seed edges from the process
    # to ensure that positive samples are not inadvertently included within
    # the negative samples.
    ############################################################################
    if is_train and args.exclude_edges:
        datapipe = datapipe.transform(
            partial(gb.exclude_seed_edges, include_reverse_edges=True)
        )

    ############################################################################
    # [Input]:
    # 'features': The node features.
    # 'node_feature_keys': The node feature keys (list) to be fetched.
    # [Output]:
    # A FeatureFetcher object to fetch node features.
    # [Role]:
    # Initialize a feature fetcher for fetching features of the sampled
    # subgraphs. This step is skipped in evaluation/inference because features
    # are updated as a whole during it, thus storing features in minibatch is
    # unnecessary.
    ############################################################################
    if is_train:
        datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])

    ############################################################################
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device.
    ############################################################################
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


def to_binary_link_dgl_computing_pack(data: gb.DGLMiniBatch):
    """Convert the minibatch to a training pair and a label tensor."""
    pos_src, pos_dst = data.positive_node_pairs
    neg_src, neg_dst = data.negative_node_pairs
    node_pairs = (
        torch.cat((pos_src, neg_src), dim=0),
        torch.cat((pos_dst, neg_dst), dim=0),
    )
    pos_label = torch.ones_like(pos_src)
    neg_label = torch.zeros_like(neg_src)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return (node_pairs, labels.float())


@torch.no_grad()
def compute_mrr(args, model, evaluator, node_emb, src, dst, neg_dst):
    """Compute the Mean Reciprocal Rank (MRR) for given source and destination
    nodes.

    This function computes the MRR for a set of node pairs, dividing the task
    into batches to handle potentially large graphs.
    """
    rr = torch.zeros(src.shape[0])
    # Loop over node pairs in batches.
    for start in tqdm.trange(
        0, src.shape[0], args.eval_batch_size, desc="Evaluate"
    ):
        end = min(start + args.eval_batch_size, src.shape[0])

        # Concatenate positive and negative destination nodes.
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)

        # Fetch embeddings for current batch of source and destination nodes.
        h_src = node_emb[src[start:end]][:, None, :].to(args.device)
        h_dst = (
            node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(args.device)
        )

        # Compute prediction scores using the model.
        pred = model.predictor(h_src * h_dst).squeeze(-1)

        # Evaluate the predictions to obtain MRR values.
        input_dict = {"y_pred_pos": pred[:, 0], "y_pred_neg": pred[:, 1:]}
        rr[start:end] = evaluator.eval(input_dict)["mrr_list"]
    return rr.mean()


@torch.no_grad()
def evaluate(args, model, graph, features, all_nodes_set, valid_set, test_set):
    """Evaluate the model on validation and test sets."""
    model.eval()
    evaluator = Evaluator(name="ogbl-citation2")

    # Since we need to use all neghborhoods for evaluation, we set the fanout
    # to -1.
    args.fanout = [-1]
    dataloader = create_dataloader(
        args, graph, features, all_nodes_set, is_train=False
    )

    # Compute node embeddings for the entire graph.
    node_emb = model.inference(graph, features, dataloader, args.device)
    results = []

    # Loop over both validation and test sets.
    for split in [valid_set, test_set]:
        # Unpack the item set.
        src = split._items[0][:, 0].to(node_emb.device)
        dst = split._items[0][:, 1].to(node_emb.device)
        neg_dst = split._items[1].to(node_emb.device)

        # Compute MRR values for the current split.
        results.append(
            compute_mrr(args, model, evaluator, node_emb, src, dst, neg_dst)
        )
    return results


def train(args, model, graph, features, train_set):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloader = create_dataloader(args, graph, features, train_set)

    for epoch in tqdm.trange(args.epochs):
        model.train()
        total_loss = 0
        start_epoch_time = time.time()
        for step, data in enumerate(dataloader):
            # Convert data to DGL format.
            data = data.to_dgl()

            # Unpack MiniBatch.
            compacted_pairs, labels = to_binary_link_dgl_computing_pack(data)
            node_feature = data.node_features["feat"]
            # Convert sampled subgraphs to DGL blocks.
            blocks = data.blocks

            # Get the embeddings of the input nodes.
            y = model(blocks, node_feature)
            logits = model.predictor(
                y[compacted_pairs[0]] * y[compacted_pairs[1]]
            ).squeeze()

            # Compute loss.
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step + 1 == args.early_stop:
                break

        end_epoch_time = time.time()
        print(
            f"Epoch {epoch:05d} | "
            f"Loss {(total_loss) / (step + 1):.4f} | "
            f"Time {(end_epoch_time - start_epoch_time):.4f} s"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="OGBL-Citation2 (GraphBolt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="0 means no early stop, otherwise stop at the input-th step",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="15,10,5",
        help="Fan-out of neighbor sampling. Default: 15,10,5",
    )
    parser.add_argument(
        "--exclude-edges",
        type=int,
        default=1,
        help="Whether to exclude reverse edges during sampling. Default: 1",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Train device: 'cpu' for CPU, 'cuda' for GPU.",
    )
    return parser.parse_args()


def main(args):
    if not torch.cuda.is_available():
        args.device = "cpu"
    print(f"Training in {args.device} mode.")

    # Load and preprocess dataset.
    print("Loading data")
    dataset = gb.BuiltinDataset("ogbl-citation2").load()
    graph = dataset.graph
    features = dataset.feature
    train_set = dataset.tasks[0].train_set
    args.fanout = list(map(int, args.fanout.split(",")))

    in_size = features.size("node", None, "feat")[0]
    hidden_channels = 256
    args.device = torch.device(args.device)
    model = SAGE(in_size, hidden_channels).to(args.device)

    # Model training.
    print("Training...")
    train(args, model, graph, features, train_set)

    # Test the model.
    print("Testing...")
    test_set = dataset.tasks[0].test_set
    valid_set = dataset.tasks[0].validation_set
    all_nodes_set = dataset.all_nodes_set
    valid_mrr, test_mrr = evaluate(
        args, model, graph, features, all_nodes_set, valid_set, test_set
    )
    print(
        f"Validation MRR {valid_mrr.item():.4f}, "
        f"Test MRR {test_mrr.item():.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
