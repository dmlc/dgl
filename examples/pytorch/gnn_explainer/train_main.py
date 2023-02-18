import argparse
import os

import torch as th
import torch.nn as nn

from dgl import save_graphs

from dgl.data import (
    BACommunityDataset,
    BAShapeDataset,
    TreeCycleDataset,
    TreeGridDataset,
)
from models import Model


def main(args):
    if args.dataset == "BAShape":
        dataset = BAShapeDataset(seed=0)
    elif args.dataset == "BACommunity":
        dataset = BACommunityDataset(seed=0)
    elif args.dataset == "TreeCycle":
        dataset = TreeCycleDataset(seed=0)
    elif args.dataset == "TreeGrid":
        dataset = TreeGridDataset(seed=0)

    graph = dataset[0]
    labels = graph.ndata["label"]
    n_feats = graph.ndata["feat"]
    num_classes = dataset.num_classes

    model = Model(n_feats.shape[-1], num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optim = th.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):
        model.train()
        # For demo purpose, we train the model on all datapoints
        # In practice, you should train only on the training datapoints
        logits = model(graph, n_feats)
        loss = loss_fn(logits, labels)
        acc = th.sum(logits.argmax(dim=1) == labels).item() / len(labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"In Epoch: {epoch}; Acc: {acc}; Loss: {loss.item()}")

    model_stat_dict = model.state_dict()
    model_path = os.path.join("./", f"model_{args.dataset}.pth")
    th.save(model_stat_dict, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy model training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BAShape",
        choices=["BAShape", "BACommunity", "TreeCycle", "TreeGrid"],
    )
    args = parser.parse_args()
    print(args)

    main(args)
