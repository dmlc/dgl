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
└───> All nodes set inference & Test set evaluation
"""
import argparse
import time

import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from tqdm import tqdm


def create_dataloader(
    graph, features, itemset, batch_size, fanout, device, num_workers, job
):
    """
    [HIGHLIGHT]
    Get a GraphBolt version of a dataloader for node classification tasks.
    This function demonstrates how to utilize functional forms of datapipes in
    GraphBolt. For a more detailed tutorial, please read the examples in
    `dgl/notebooks/graphbolt/walkthrough.ipynb`.
    Alternatively, you can create a datapipe using its class constructor.

    Parameters
    ----------
    job : one of ["train", "evaluate", "infer"]
        The stage where dataloader is created, with options "train", "evaluate"
        and "infer".
    Other parameters are explicated in the comments below.
    """

    ############################################################################
    # [Step-1]:
    # gb.ItemSampler()
    # [Input]:
    # 'itemset': The current dataset. (e.g. `train_set` or `valid_set`)
    # 'batch_size': Specify the number of samples to be processed together,
    # referred to as a 'mini-batch'. (The term 'mini-batch' is used here to
    # indicate a subset of the entire dataset that is processed together. This
    # is in contrast to processing the entire dataset, known as a 'full batch'.)
    # 'job': Determines whether data should be shuffled. (Shuffling is
    # generally used only in training to improve model generalization. It's
    # not used in validation and testing as the focus there is to evaluate
    # performance rather than to learn from the data.)
    # [Output]:
    # An ItemSampler object for handling mini-batch sampling.
    # [Role]:
    # Initialize the ItemSampler to sample mini-batche from the dataset.
    ############################################################################
    datapipe = gb.ItemSampler(
        itemset, batch_size=batch_size, shuffle=(job == "train")
    )

    ############################################################################
    # [Step-2]:
    # self.copy_to()
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device. Copying here
    # ensures that the rest of the operations run on the GPU.
    ############################################################################
    if args.storage_device != "cpu":
        datapipe = datapipe.copy_to(device=device)

    ############################################################################
    # [Step-3]:
    # self.sample_neighbor()
    # [Input]:
    # 'graph': The network topology for sampling.
    # '[-1] or fanout': Number of neighbors to sample per node. In
    # training or validation, the length of `fanout` should be equal to the
    # number of layers in the model. In inference, this parameter is set to
    # [-1], indicating that all neighbors of a node are sampled.
    # [Output]:
    # A NeighborSampler object to sample neighbors.
    # [Role]:
    # Initialize a neighbor sampler for sampling the neighborhoods of nodes.
    ############################################################################
    datapipe = getattr(datapipe, args.sample_mode)(
        graph,
        fanout if job != "infer" else [-1],
        overlap_fetch=args.storage_device == "pinned",
        asynchronous=args.storage_device != "cpu",
    )

    ############################################################################
    # [Step-4]:
    # self.fetch_feature()
    # [Input]:
    # 'features': The node features.
    # 'node_feature_keys': The keys of the node features to be fetched.
    # [Output]:
    # A FeatureFetcher object to fetch node features.
    # [Role]:
    # Initialize a feature fetcher for fetching features of the sampled
    # subgraphs.
    ############################################################################
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])

    ############################################################################
    # [Step-5]:
    # self.copy_to()
    # [Input]:
    # 'device': The device to copy the data to.
    # [Output]:
    # A CopyTo object to copy the data to the specified device.
    ############################################################################
    if args.storage_device == "cpu":
        datapipe = datapipe.copy_to(device=device)

    ############################################################################
    # [Step-6]:
    # gb.DataLoader()
    # [Input]:
    # 'datapipe': The datapipe object to be used for data loading.
    # 'num_workers': The number of processes to be used for data loading.
    # [Output]:
    # A DataLoader object to handle data loading.
    # [Role]:
    # Initialize a multi-process dataloader to load the data in parallel.
    ############################################################################
    dataloader = gb.DataLoader(datapipe, num_workers=num_workers)

    # Return the fully-initialized DataLoader object.
    return dataloader


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
        num_workers=args.num_workers,
        job="infer",
    )
    pred = model.inference(graph, features, dataloader, args.storage_device)
    pred = pred[test_set._items[0]]
    label = test_set._items[1].to(pred.device)

    return MF.accuracy(
        pred,
        label,
        task="multiclass",
        num_classes=num_classes,
    )


@torch.no_grad()
def evaluate(args, model, graph, features, itemset, num_classes):
    model.eval()
    y = []
    y_hats = []
    dataloader = create_dataloader(
        graph=graph,
        features=features,
        itemset=itemset,
        batch_size=args.batch_size,
        fanout=args.fanout,
        device=args.device,
        num_workers=args.num_workers,
        job="evaluate",
    )

    for step, data in tqdm(enumerate(dataloader), "Evaluating"):
        x = data.node_features["feat"]
        y.append(data.labels)
        y_hats.append(model(data.blocks, x))

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(y),
        task="multiclass",
        num_classes=num_classes,
    )


def train(args, graph, features, train_set, valid_set, num_classes, model):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    dataloader = create_dataloader(
        graph=graph,
        features=features,
        itemset=train_set,
        batch_size=args.batch_size,
        fanout=args.fanout,
        device=args.device,
        num_workers=args.num_workers,
        job="train",
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        for step, data in tqdm(enumerate(dataloader), "Training"):
            # The input features from the source nodes in the first layer's
            # computation graph.
            x = data.node_features["feat"]

            # The ground truth labels from the destination nodes
            # in the last layer's computation graph.
            y = data.labels

            y_hat = model(data.blocks, x)

            # Compute loss.
            loss = F.cross_entropy(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        t1 = time.time()
        # Evaluate the model.
        acc = evaluate(args, model, graph, features, valid_set, num_classes)
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (step + 1):.4f} | "
            f"Accuracy {acc.item():.4f} | Time {t1 - t0:.4f}"
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
        default=1e-3,
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
        "--fanout",
        type=str,
        default="10,10,10",
        help="Fan-out of neighbor sampling. It is IMPORTANT to keep len(fanout)"
        " identical with the number of layers in your model. Default: 10,10,10",
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
        help="The dataset we can use for node classification example. Currently"
        " ogbn-products, ogbn-arxiv, ogbn-papers100M and"
        " igb-hom-[tiny|small|medium|large] and igb-hom datasets are supported.",
    )
    parser.add_argument(
        "--mode",
        default="pinned-cuda",
        choices=["cpu-cpu", "cpu-cuda", "pinned-cuda", "cuda-cuda"],
        help="Dataset storage placement and Train device: 'cpu' for CPU and RAM,"
        " 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
    )
    parser.add_argument(
        "--sample-mode",
        default="sample_neighbor",
        choices=["sample_neighbor", "sample_layer_neighbor"],
        help="The sampling function when doing layerwise sampling.",
    )
    return parser.parse_args()


def main(args):
    if not torch.cuda.is_available():
        args.mode = "cpu-cpu"
    print(f"Training in {args.mode} mode.")
    args.storage_device, args.device = args.mode.split("-")
    args.device = torch.device(args.device)

    # Load and preprocess dataset.
    print("Loading data...")
    dataset = gb.BuiltinDataset(args.dataset).load()

    # Move the dataset to the selected storage.
    if args.storage_device == "pinned":
        graph = dataset.graph.pin_memory_()
        features = dataset.feature.pin_memory_()
    else:
        graph = dataset.graph.to(args.storage_device)
        features = dataset.feature.to(args.storage_device)

    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    all_nodes_set = dataset.all_nodes_set
    args.fanout = list(map(int, args.fanout.split(",")))

    num_classes = dataset.tasks[0].metadata["num_classes"]

    in_size = features.size("node", None, "feat")[0]
    hidden_size = 256
    out_size = num_classes

    model = SAGE(in_size, hidden_size, out_size)
    assert len(args.fanout) == len(model.layers)
    model = model.to(args.device)

    # Model training.
    print("Training...")
    train(args, graph, features, train_set, valid_set, num_classes, model)

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
    main(args)
