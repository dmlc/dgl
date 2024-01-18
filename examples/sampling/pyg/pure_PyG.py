import argparse
import time

import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from memory_profiler import memory_usage
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_size, hidden_size)
        self.conv2 = SAGEConv(hidden_size, hidden_size)
        self.conv3 = SAGEConv(hidden_size, out_size)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = self.conv3(h, edge_index)
        return h


def train(model, data, train_loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs[0].edge_index)
        out = out[:batch_size]
        y = data.y[n_id[:batch_size]].to(device).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_accuracy = MF.accuracy(
            out, y, num_classes=num_classes, task="multiclass"
        )
        total_correct += batch_accuracy * batch_size
        total_samples += batch_size
    return total_loss / len(train_loader), total_correct / total_samples


@torch.no_grad()
def evaluate(model, data, loader, device, num_classes):
    model.eval()
    total_accuracy = 0
    total_samples = 0
    for batch_size, n_id, adjs in loader:
        edge_index = adjs[0][0] if len(adjs[0]) == 3 else adjs[0]
        edge_index = edge_index.to(device)
        out = model(data.x[n_id].to(device), edge_index)
        out = out[:batch_size]
        labels = data.y[n_id[:batch_size]].to(device).squeeze()
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        batch_accuracy = MF.accuracy(
            out, labels, num_classes=num_classes, task="multiclass"
        )
        total_accuracy += batch_accuracy * batch_size
        total_samples += batch_size
    return total_accuracy / total_samples


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE with PyG")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        help="Dataset name (e.g., 'ogbn-arxiv')",
    )
    args = parser.parse_args()
    start_time = time.time()
    initial_mem = torch.cuda.memory_allocated()
    mem_usage_start = memory_usage(-1, interval=1, timeout=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("device is: ", device)
    print("Loading data")
    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0].to(device)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)
    num_classes = dataset.num_classes
    start_dataloader_time = time.time()
    train_loader = NeighborSampler(
        data.edge_index,
        node_idx=split_idx["train"],
        sizes=[10, 10, 10],
        batch_size=1024,
        shuffle=True,
        num_workers=12,
    )
    end_dataloader_time = time.time()
    dataloader_creation_time = end_dataloader_time - start_dataloader_time
    print(f"train_loader Creation Time: {dataloader_creation_time} seconds")
    subgraph_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=[-1],
        num_workers=12,
        batch_size=1024,
        shuffle=False,
    )
    model = GraphSAGE(dataset.num_node_features, 128, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        loss, acc = train(
            model, data, train_loader, optimizer, criterion, device, num_classes
        )
        val_acc = evaluate(model, data, subgraph_loader, device, num_classes)
        print(
            f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, Val Acc: {val_acc:.4f}"
        )
    test_acc = evaluate(model, data, subgraph_loader, device, num_classes)
    print(f"Test Accuracy: {test_acc:.4f}")
    end_time = time.time()
    mem_usage_end = memory_usage(-1, interval=1, timeout=1)
    total_training_time = end_time - start_time
    peak_memory_usage = max(mem_usage_end) - mem_usage_start[0]
    print(f"Total Training Time: {total_training_time} seconds")
    print(f"Peak Memory Usage: {peak_memory_usage} MiB")
    final_mem = torch.cuda.memory_allocated()
    peak_mem_during_training = torch.cuda.max_memory_allocated()
    print(f"Initial Memory: {initial_mem / 1024 / 1024} MB")
    print(f"Final Memory: {final_mem / 1024 / 1024} MB")
    print(
        f"Peak Memory during Training: {peak_mem_during_training / 1024 / 1024} MB"
    )


if __name__ == "__main__":
    main()
