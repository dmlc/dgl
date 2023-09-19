"""
This script trains and tests a GraphSAGE model for node classification
on large graphs using GraphBolt dataloader.

Paper: [Inductive Representation Learning on Large Graphs]
(https://arxiv.org/abs/1706.02216)

Unlike previous dgl examples, we've utilized the newly defined dataloader
from GraphBolt. This example will help you grasp how to build an end-to-end
training pipeline using GraphBolt.

Before reading this example, please familiar yourself with graphsage node
classification by reading the example in the
`examples/core/graphsage/node_classification.py`. This introduction,
[A Blitz Introduction to Node Classification with DGL]
(https://docs.dgl.ai/tutorials/blitz/1_introduction.html), might be helpful.

If you want to train graphsage on a large graph in a distributed fashion,
please read the example in the `examples/distributed/graphsage/`.

This flowchart describes the main functional sequence of the provided example:
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

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean.
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hidden_size = hidden_size
        self.out_size = out_size

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x


def create_dataloader(args, graph, features, itemset, is_train=True):
    """
    [HIGHLIGHT]
    Get a GraphBolt version of a dataloader for node classification tasks.
    This function demonstrates how to utilize functional forms of datapipes in
    GraphBolt.
    Alternatively, you can create a datapipe using its class constructor.
    """

    ############################################################################
    # [Step-1]:
    # gb.ItemSampler()
    # [Input]:
    # 'itemset': The current dataset. (e.g. `train_set` or `valid_set`)
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
    # [Step-2]:
    # self.sample_neighbor()
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
    # [Step-3]:
    # self.fetch_feature()
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
    # [Step-4]:
    # gb.MultiProcessDataLoader()
    # [Input]:
    # 'datapipe': The datapipe object to be used for data loading.
    # 'args.num_workers': The number of processes to be used for data loading.
    # [Output]:
    # A MultiProcessDataLoader object to handle data loading.
    # [Role]:
    # Initialize a multi-process dataloader to load the data in parallel.
    ############################################################################
    dataloader = gb.MultiProcessDataLoader(
        datapipe, num_workers=args.num_workers
    )

    # Return the fully-initialized DataLoader object.
    return dataloader


@torch.no_grad()
def evaluate(args, model, graph, features, itemset, num_classes):
    model.eval()
    ys = []
    y_hats = []
    dataloader = create_dataloader(
        args, graph, features, itemset, is_train=False
    )

    for step, data in tqdm.tqdm(enumerate(dataloader)):
        blocks = data.to_dgl_blocks()

        x = data.node_features["feat"].float()
        ys.append(data.labels)
        y_hats.append(model(blocks, x))

    res = MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )

    return res


def train(args, graph, features, train_set, valid_set, num_classes, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloader = create_dataloader(
        args, graph, features, train_set, is_train=True
    )

    for epoch in tqdm.trange(args.epochs):
        model.train()
        total_loss = 0
        for step, data in tqdm.tqdm(enumerate(dataloader)):
            # The input features from the source nodes in the first layer's
            # computation graph.
            x = data.node_features["feat"].float()

            # The ground truth labels from the destination nodes
            # in the last layer's computation graph.
            y = data.labels

            # The predicted labels.
            y_hat = model(data.to_dgl_blocks(), x)

            # Compute loss.
            loss = F.cross_entropy(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate the model.
        print("Validating...")
        acc = evaluate(args, model, graph, features, valid_set, num_classes)
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (step + 1):.4f} | "
            f"Accuracy {acc.item():.4f} "
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script trains and tests a GraphSAGE model "
        "for node classification using GraphBolt dataloader."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate for optimization.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for training."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="15,10,5",
        help="Fan-out of neighbor sampling. It is IMPORTANT to keep len(fanout)"
        " identical with the number of layers in your model. Default: 15,10,5",
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
    dataset = gb.BuiltinDataset("ogbn-products").load()

    graph = dataset.graph
    features = dataset.feature
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    args.fanout = list(map(int, args.fanout.split(",")))

    num_classes = dataset.tasks[0].metadata["num_classes"]

    in_size = features.read("node", None, "feat").shape[
        1
    ]  # Size of feature of a single node.
    hidden_size = 128
    out_size = num_classes

    model = SAGE(in_size, hidden_size, out_size)

    # Model training.
    print("Training...")
    train(args, graph, features, train_set, valid_set, num_classes, model)

    # Test the model.
    print("Testing...")
    test_set = dataset.tasks[0].test_set
    test_acc = evaluate(
        args, model, graph, features, itemset=test_set, num_classes=num_classes
    )
    print(f"Test Accuracy is {test_acc.item():.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
