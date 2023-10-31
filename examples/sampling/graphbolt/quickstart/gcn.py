"""
[Semi-Supervised Classification with Graph Convolutional Networks]
(https://arxiv.org/abs/1609.02907)
"""
import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF


############################################################################
# (HIGHLIGHT) Create a single process dataloader with dgl graphbolt package.
############################################################################
def create_dataloader(dateset, itemset, device):
    # Sample seed nodes from the itemset.
    datapipe = gb.ItemSampler(itemset, batch_size=8)

    # Sample neighbors for the seed nodes.
    datapipe = datapipe.sample_neighbor(dataset.graph, fanouts=[4, 2])

    # Fetch features for sampled nodes.
    datapipe = datapipe.fetch_feature(
        dataset.feature, node_feature_keys=["feat"]
    )

    # Convert the mini-batch to DGL format to train a DGL model.
    datapipe = datapipe.to_dgl()

    # Copy the mini-batch to the designated device for training.
    datapipe = datapipe.copy_to(device)

    # Initiate the dataloader for the datapipe.
    return gb.SingleProcessDataLoader(datapipe)


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_size, hidden_size))
        self.layers.append(dglnn.GraphConv(hidden_size, out_size))
        self.hidden_size = hidden_size
        self.out_size = out_size

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
        return hidden_x


@torch.no_grad()
def evaluate(model, dataset, itemset, device):
    model.eval()
    y = []
    y_hats = []
    dataloader = create_dataloader(dataset, itemset, device)

    for step, data in enumerate(dataloader):
        x = data.node_features["feat"]
        y.append(data.labels)
        y_hats.append(model(data.blocks, x))

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(y),
        task="multiclass",
        num_classes=dataset.tasks[0].metadata["num_classes"],
    )


def train(model, dataset, device):
    task = dataset.tasks[0]
    dataloader = create_dataloader(dataset, task.train_set, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(20):
        model.train()
        total_loss = 0
        ########################################################################
        # (HIGHLIGHT) Iterate over the dataloader and train the model with all
        # mini-batches.
        ########################################################################
        for step, data in enumerate(dataloader):
            # The features of sampled nodes.
            x = data.node_features["feat"]

            # The ground truth labels of the seed nodes.
            y = data.labels

            # Forward.
            y_hat = model(data.blocks, x)

            # Compute loss.
            loss = F.cross_entropy(y_hat, y)

            # Backward.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate the model.
        val_acc = evaluate(model, dataset, task.validation_set, device)
        test_acc = evaluate(model, dataset, task.test_set, device)
        print(
            f"Epoch {epoch:03d} | Loss {total_loss / (step + 1):.4f} | "
            f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}"
        )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training in {device} mode.")

    # Load and preprocess dataset.
    print("Loading data...")
    dataset = gb.OnDiskDataset(
        "examples/sampling/graphbolt/quickstart/cora/"
    ).load()

    in_size = dataset.feature.size("node", None, "feat")[0]
    out_size = dataset.tasks[0].metadata["num_classes"]
    model = GCN(in_size, out_size).to(device)

    # Model training.
    print("Training...")
    train(model, dataset, device)
