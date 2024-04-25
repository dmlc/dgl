"""
This example shows how to create a GraphBolt dataloader to sample and train a
node classification model with the Cora dataset.
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
def create_dataloader(dataset, itemset, device):
    # Sample seed nodes from the itemset.
    datapipe = gb.ItemSampler(itemset, batch_size=16)

    # Copy the mini-batch to the designated device for sampling and training.
    datapipe = datapipe.copy_to(device)

    # Sample neighbors for the seed nodes.
    datapipe = datapipe.sample_neighbor(dataset.graph, fanouts=[4, 2])

    # Fetch features for sampled nodes.
    datapipe = datapipe.fetch_feature(
        dataset.feature, node_feature_keys=["feat"]
    )

    # Initiate the dataloader for the datapipe.
    return gb.DataLoader(datapipe)


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_size, hidden_size))
        self.layers.append(dglnn.GraphConv(hidden_size, out_size))

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
    # The first of two tasks in the dataset is node classification.
    task = dataset.tasks[0]
    dataloader = create_dataloader(dataset, task.train_set, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(10):
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
            f"Epoch {epoch:03d} | Loss {total_loss / (step + 1):.3f} | "
            f"Val Acc {val_acc.item():.3f} | Test Acc {test_acc.item():.3f}"
        )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training in {device} mode.")

    # Load and preprocess dataset.
    print("Loading data...")
    dataset = gb.BuiltinDataset("cora").load()

    # If a CUDA device is selected, we pin the graph and the features so that
    # the GPU can access them.
    if device == torch.device("cuda:0"):
        dataset.graph.pin_memory_()
        dataset.feature.pin_memory_()

    in_size = dataset.feature.size("node", None, "feat")[0]
    out_size = dataset.tasks[0].metadata["num_classes"]
    model = GCN(in_size, out_size).to(device)

    # Model training.
    print("Training...")
    train(model, dataset, device)
