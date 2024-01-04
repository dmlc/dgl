import random

import torch
import torch.nn.functional as F
from dgl.graphbolt import (
    BuiltinDataset,
    CopyTo,
    DataLoader,
    FeatureFetcher,
    ItemSampler,
)
from torch_geometric.nn import GCNConv


def load_data():
    dataset = BuiltinDataset("ogbl-citation2").load()
    graph = dataset.graph
    feature = dataset.feature
    return (
        graph,
        feature,
        dataset.tasks[0].train_set,
        dataset.tasks[0].validation_set,
        dataset.tasks[0].test_set,
    )


# def generate_negative_edges(num_nodes, positive_edges, num_neg_samples):
#     pos_edge_set = set([(u.item(), v.item()) for u, v in zip(*positive_edges)])
#     neg_edges = []
#     while len(neg_edges) < num_neg_samples:
#         u = random.randint(0, num_nodes - 1)
#         v = random.randint(0, num_nodes - 1)
#         if (u, v) not in pos_edge_set and (u, v) not in neg_edges:
#             neg_edges.append((u, v))
#     neg_edges = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
#     return neg_edges.to(positive_edges.device)


#  More efficient
def generate_negative_edges(num_nodes, positive_edges, num_neg_samples):
    all_nodes = torch.arange(num_nodes, device=positive_edges.device)
    neg_edges = []

    for _ in range(num_neg_samples):
        src = positive_edges[0, torch.randint(0, positive_edges.size(1), (1,))]
        dst = all_nodes[torch.randint(0, num_nodes, (1,))]
        neg_edges.append(torch.stack([src, dst], dim=0))

    neg_edges = torch.cat(neg_edges, dim=1)
    return neg_edges


class LinkPredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LinkPredictionModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def predict_edges(self, z, edge_index):
        src, dest = edge_index
        edge_scores = torch.sigmoid((z[src] * z[dest]).sum(dim=1))
        return edge_scores


def train(
    model, feature, edge_index, train_item_set, optimizer, criterion, device
):
    model.train()
    optimizer.zero_grad()

    pos_edge_index = torch.stack(train_item_set[:], dim=0)

    z = model(feature, edge_index)

    neg_edge_index = generate_negative_edges(
        feature.size(0), pos_edge_index, pos_edge_index.size(1)
    )

    pos_pred = model.predict_edges(z, pos_edge_index)
    neg_pred = model.predict_edges(z, neg_edge_index)
    pos_label = torch.ones(pos_pred.size(0), device=device)
    neg_label = torch.zeros(neg_pred.size(0), device=device)
    loss = criterion(pos_pred, pos_label) + criterion(neg_pred, neg_label)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, feature, edge_index, eval_item_set, device):
    model.eval()
    z = model(feature, edge_index)
    pos_edge_index = torch.stack(eval_item_set[:], dim=0)
    neg_edge_index = generate_negative_edges(
        feature.size(0), pos_edge_index, pos_edge_index.size(1)
    )

    pos_pred = model.predict_edges(z, pos_edge_index)
    neg_pred = model.predict_edges(z, neg_edge_index)
    pos_label = torch.ones(pos_pred.size(0), device=device)
    neg_label = torch.zeros(neg_pred.size(0), device=device)
    pos_loss = F.binary_cross_entropy(pos_pred, pos_label)
    neg_loss = F.binary_cross_entropy(neg_pred, neg_label)
    return (pos_loss + neg_loss) / 2


def get_edge_index_from_csc_graph(graph):
    indptr = graph.csc_indptr
    indices = graph.indices
    edge_index = []

    for dst_node in range(indptr.size(0) - 1):
        for edge_idx in range(indptr[dst_node], indptr[dst_node + 1]):
            src_node = indices[edge_idx]
            edge_index.append((src_node, dst_node))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def main():
    graph, feature_store, train_set, valid_set, test_set = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_tensor = feature_store.read("node", None, "feat")
    feature_tensor = feature_tensor.to(device)
    edge_index = get_edge_index_from_csc_graph(graph).to(device)

    model = LinkPredictionModel(
        in_channels=feature_tensor.size(1), hidden_channels=16
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    for epoch in range(100):
        train_loss = train(
            model,
            feature_tensor,
            edge_index,
            train_set,
            optimizer,
            criterion,
            device,
        )
        valid_loss = evaluate(
            model, feature_tensor, edge_index, valid_set, device
        )
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )

    test_loss = evaluate(model, feature_tensor, edge_index, test_set, device)
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
