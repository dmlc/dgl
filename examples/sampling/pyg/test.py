import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.graphbolt import (
    BuiltinDataset,
    CopyTo,
    DataLoader,
    FeatureFetcher,
    ItemSampler,
)
from torch_geometric.nn import GCNConv


dataset = BuiltinDataset("ogbn-arxiv").load()
graph = dataset.graph
feature = dataset.feature
train_set = dataset.tasks[0].train_set
valid_set = dataset.tasks[0].validation_set
test_set = dataset.tasks[0].test_set
num_classes = dataset.tasks[0].metadata["num_classes"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_sampler_and_loader(dataset_set):
    sampler = ItemSampler(dataset_set, batch_size=1024, shuffle=True)
    datapipe = sampler.sample_neighbor(graph, [4, 4])
    datapipe = FeatureFetcher(datapipe, feature, node_feature_keys=["feat"])
    datapipe = CopyTo(datapipe, device=device)
    return DataLoader(datapipe, num_workers=0)


train_dataloader = create_sampler_and_loader(train_set)
valid_dataloader = create_sampler_and_loader(valid_set)
test_dataloader = create_sampler_and_loader(test_set)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


in_channels = feature.size("node", None, "feat")[0]
model = GCN(in_channels, 16, num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train_epoch(model, dataloader):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    for minibatch in dataloader:
        batch_count += 1
        node_features = minibatch.node_features["feat"].to(device)
        block = minibatch.blocks[-1]
        edge_index = block.edges()
        edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0).to(
            device
        )
        out = model(node_features, edge_index)
        out = out[-block.number_of_dst_nodes() :]
        num_dst_nodes = block.number_of_dst_nodes()
        labels = minibatch.labels[-num_dst_nodes:].to(device)
        loss = F.nll_loss(out, labels)
        total_loss += loss.item()
        predictions = out.argmax(dim=1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += num_dst_nodes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / batch_count, total_correct / total_samples


def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for minibatch in dataloader:
            node_features = minibatch.node_features["feat"].to(device)
            block = minibatch.blocks[-1]
            edge_index = block.edges()
            edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0).to(
                device
            )
            out = model(node_features, edge_index)
            out = out[-block.number_of_dst_nodes() :]
            num_dst_nodes = block.number_of_dst_nodes()
            labels = minibatch.labels[-num_dst_nodes:].to(device)
            predictions = out.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += num_dst_nodes
    return total_correct / total_samples


for epoch in range(100):
    train_loss, train_accuracy = train_epoch(model, train_dataloader)
    valid_accuracy = evaluate(model, valid_dataloader)
    print(
        f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}"
    )


test_accuracy = evaluate(model, test_dataloader)
print(f"Test Accuracy: {test_accuracy:.4f}")
