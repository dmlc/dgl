"""
This script demonstrates node classification with GraphSAGE on large graphs, 
merging GraphBolt (GB) and PyTorch Geometric (PyG). GraphBolt efficiently
manages data loading for large datasets, crucial for mini-batch processing.
Post data loading, PyG's user-friendly framework takes over for training,
showcasing seamless integration with GraphBolt. This combination offers an
efficient alternative to traditional Deep Graph Library (DGL) methods,
highlighting adaptability and scalability in handling large-scale graph data
for diverse real-world applications.

Key Features:
- Implements the GraphSAGE model, a scalable GNN, for node classification on
  large graphs.
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

import dgl.graphbolt as gb
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from torch_geometric.nn import SAGEConv


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

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, args, dataloader, features):
        """Conduct layer-wise inference to get all the node embeddings."""
        for i, layer in enumerate(self.layers):
            xs = []
            for minibatch in dataloader:
                pyg_data = minibatch.to_pyg_data()
                x = layer(pyg_data.x, pyg_data.edge_index)[
                    : 4 * args.batch_size
                ]
                if i != len(self.layers) - 1:
                    x = x.relu()
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
            if i != len(self.layers) - 1:
                features.update("node", None, "feat", x_all)
        return x_all


def create_dataloader(
    dataset_set, graph, feature, batch_size, fanout, device, job
):
    #####################################################################
    # (HIGHLIGHT) Create a data loader for efficiently loading graph data.
    #
    # - 'ItemSampler' samples mini-batches of node IDs from the dataset.
    # - 'sample_neighbor' performs neighbor sampling on the graph.
    # - 'FeatureFetcher' fetches node features based on the sampled subgraph.
    # - 'CopyTo' copies the fetched data to the specified device.

    #####################################################################
    # Create a datapipe for mini-batch sampling with a specific neighbor fanout.
    # Here, [10, 10, 10] specifies the number of neighbors sampled for each
    # node at each layer. We're using `sample_neighbor` for consistency with
    # DGL's sampling API.
    # Note: GraphBolt offers additional sampling methods, such as
    # `sample_layer_neighbor`, which could provide further optimization and
    # efficiency for GNN training. Users are encouraged to explore these
    # advanced features for potentially improved performance.

    # Initialize an ItemSampler to sample mini-batches from the dataset.
    datapipe = gb.ItemSampler(
        dataset_set,
        batch_size=batch_size,
        shuffle=(job == "train"),
        drop_last=(job == "train"),
    )
    # Sample neighbors for each node in the mini-batch.
    datapipe = datapipe.sample_neighbor(
        graph, fanout if job != "infer" else [-1]
    )
    # Fetch node features for the sampled subgraph.
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    # Copy the data to the specified device.
    datapipe = datapipe.copy_to(device=device)
    # Create and return a DataLoader to handle data loading.
    dataloader = gb.DataLoader(datapipe, num_workers=0)

    return dataloader


def train(model, dataloader, optimizer, criterion, device, num_classes):
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

    model.train()  # Set the model to training mode
    total_loss = 0  # Accumulator for the total loss
    total_correct = 0  # Accumulator for the total number of correct predictions
    total_samples = 0  # Accumulator for the total number of samples processed
    num_batches = 0  # Counter for the number of mini-batches processed

    for minibatch in dataloader:
        pyg_data = minibatch.to_pyg_data()

        # print(pyg_data.x.shape, pyg_data.edge_index.shape)
        out = model(pyg_data.x, pyg_data.edge_index)[: pyg_data.y.shape[0]]
        y = pyg_data.y
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += MF.accuracy(
            out, y, task="multiclass", num_classes=num_classes
        ) * y.size(0)
        total_samples += y.size(0)
        num_batches += 1
    avg_loss = total_loss / num_batches
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes):
    model.eval()
    y_hats = []
    ys = []
    for minibatch in dataloader:
        pyg_data = minibatch.to_pyg_data()
        out = model(pyg_data.x, pyg_data.edge_index)[: pyg_data.y.shape[0]]
        y = pyg_data.y
        y_hats.append(out)
        ys.append(y)

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


@torch.no_grad()
def layerwise_infer(
    model, args, infer_dataloader, test_set, features, num_classes
):
    model.eval()
    pred = model.inference(args, infer_dataloader, features)
    pred = pred[test_set._items[0]]
    label = test_set._items[1].to(pred.device)
    print(pred, label)

    return MF.accuracy(
        pred,
        label,
        task="multiclass",
        num_classes=num_classes,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Which dataset are you going to use?"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help='Name of the dataset to use (e.g., "ogbn-products", "ogbn-arxiv")',
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for training."
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset = gb.BuiltinDataset(dataset_name).load()
    graph = dataset.graph
    feature = dataset.feature
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    all_nodes_set = dataset.all_nodes_set
    num_classes = dataset.tasks[0].metadata["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = create_dataloader(
        train_set,
        graph,
        feature,
        args.batch_size,
        [10, 10, 10],
        device,
        job="train",
    )
    valid_dataloader = create_dataloader(
        valid_set,
        graph,
        feature,
        args.batch_size,
        [10, 10, 10],
        device,
        job="evaluate",
    )
    infer_dataloader = create_dataloader(
        all_nodes_set,
        graph,
        feature,
        4 * args.batch_size,
        [-1],
        device,
        job="infer",
    )
    in_channels = feature.size("node", None, "feat")[0]
    hidden_channels = 128
    model = GraphSAGE(in_channels, hidden_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train(
            model, train_dataloader, optimizer, criterion, device, num_classes
        )

        valid_accuracy = evaluate(model, valid_dataloader, device, num_classes)
        print(
            f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, "
            f"Valid Accuracy: {valid_accuracy:.4f}"
        )
    test_accuracy = layerwise_infer(
        model, args, infer_dataloader, test_set, feature, num_classes
    )
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
