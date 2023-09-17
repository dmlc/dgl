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

import dgl
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


def create_dataloader(args, graph, features, itemset, is_train=True):
    """Get a GraphBolt version of a dataloader for link prediction tasks. This
    function demonstrates how to utilize functional forms of datapipes in
    GraphBolt. Alternatively, you can create a datapipe using its class
    constructor.
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
        itemset, batch_size=args.batch_size, shuffle=is_train
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
    # 'features': The node features.
    # 'node_feature_keys': The node feature keys (list) to be fetched.
    # [Output]:
    # A FeatureFetcher object to fetch node features.
    # [Role]:
    # Initialize a feature fetcher for fetching features of the sampled
    # subgraphs.
    ############################################################################
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
    # A MultiProcessDataLoader object to handle data loading.
    # [Role]:
    # Initialize a multi-process dataloader to load the data in parallel.
    ############################################################################
    dataloader = gb.MultiProcessDataLoader(
        datapipe,
        num_workers=args.num_workers,
    )

    # Return the fully-initialized DataLoader object.
    return dataloader


# TODO[Keli]: Remove this helper function later.
def to_binary_link_dgl_computing_pack(data: gb.MiniBatch):
    """Convert the minibatch to a training pair and a label tensor."""
    batch_size = data.compacted_node_pairs[0].shape[0]
    neg_ratio = data.compacted_negative_dsts.shape[0] // batch_size

    pos_src, pos_dst = data.compacted_node_pairs
    if data.compacted_negative_srcs is None:
        neg_src = pos_src.repeat_interleave(neg_ratio, dim=0)
    else:
        neg_src = data.compacted_negative_srcs
    neg_dst = data.compacted_negative_dsts

    node_pairs = (
        torch.cat((pos_src, neg_src), dim=0),
        torch.cat((pos_dst, neg_dst), dim=0),
    )
    pos_label = torch.ones_like(pos_src)
    neg_label = torch.zeros_like(neg_src)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return (node_pairs, labels.float())


# TODO[Keli]: Remove this helper function later.
def to_dgl_blocks(sampled_subgraphs: gb.SampledSubgraphImpl):
    """Convert sampled subgraphs to DGL blocks."""
    blocks = [
        dgl.create_block(
            sampled_subgraph.node_pairs,
            num_src_nodes=sampled_subgraph.reverse_row_node_ids.shape[0],
            num_dst_nodes=sampled_subgraph.reverse_column_node_ids.shape[0],
        )
        for sampled_subgraph in sampled_subgraphs
    ]
    return blocks


@torch.no_grad()
def evaluate(args, graph, features, itemset, model):
    evaluator = Evaluator(name="ogbl-citation2")

    # Since we need to evaluate the model, we need to set the number
    # of layers to 1 and the fanout to -1.
    args.fanout = [torch.LongTensor([-1])]
    dataloader = create_dataloader(
        args, graph, features, itemset, is_train=False
    )
    pos_pred = []
    neg_pred = []

    model.eval()
    for step, data in tqdm.tqdm(enumerate(dataloader)):
        # Unpack MiniBatch.
        compacted_pairs, _ = to_binary_link_dgl_computing_pack(data)
        node_feature = data.node_features["feat"].float()
        blocks = to_dgl_blocks(data.sampled_subgraphs)

        # Get the embeddings of the input nodes.
        y = model(blocks, node_feature)
        # Calculate the score for positive and negative edges.
        score = (
            model.predictor(y[compacted_pairs[0]] * y[compacted_pairs[1]])
            .squeeze()
            .detach()
        )

        # Split the score into positive and negative parts.
        pos_score = score[: data.compacted_node_pairs[0].shape[0]]
        neg_score = score[data.compacted_node_pairs[0].shape[0] :]

        # Append the score to the list.
        pos_pred.append(pos_score)
        neg_pred.append(neg_score)
    pos_pred = torch.cat(pos_pred, dim=0)
    neg_pred = torch.cat(neg_pred, dim=0).view(pos_pred.shape[0], -1)

    input_dict = {"y_pred_pos": pos_pred, "y_pred_neg": neg_pred}
    mrr = evaluator.eval(input_dict)["mrr_list"]
    return mrr.mean()


def train(args, graph, features, train_set, valid_set, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloader = create_dataloader(args, graph, features, train_set)

    for epoch in tqdm.trange(args.epochs):
        model.train()
        total_loss = 0
        for step, data in enumerate(dataloader):
            # Unpack MiniBatch.
            compacted_pairs, labels = to_binary_link_dgl_computing_pack(data)
            node_feature = data.node_features["feat"].float()
            # Convert sampled subgraphs to DGL blocks.
            blocks = to_dgl_blocks(data.sampled_subgraphs)

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
            if (step % 100 == 0) and (step != 0):
                print(
                    f"Epoch {epoch:05d} | "
                    f"Step {step:05d} | "
                    f"Loss {(total_loss) / (step + 1):.4f}",
                    end="\n",
                )

    # Evaluate the model.
    print("Validation")
    valid_mrr = evaluate(args, graph, features, valid_set, model)
    print(f"Valid MRR {valid_mrr.item():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="OGBL-Citation2 (GraphBolt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--fanout",
        type=str,
        default="15,10,5",
        help="Fan-out of neighbor sampling. Default: 15,10,5",
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
    valid_set = dataset.tasks[0].validation_set
    args.fanout = list(map(int, args.fanout.split(",")))

    in_size = 128
    hidden_channels = 256
    model = SAGE(in_size, hidden_channels)

    # Model training.
    print("Training...")
    train(args, graph, features, train_set, valid_set, model)

    # Test the model.
    print("Testing...")
    test_set = dataset.tasks[0].test_set
    test_mrr = evaluate(args, graph, features, test_set, model)
    print(f"Test MRR {test_mrr.item():.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
