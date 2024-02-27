import argparse
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from ggcm import GGCM
from utils import Classifier


def evaluate(model, embeds, graph):
    model.eval()
    with torch.no_grad():
        output = model(embeds)
        pred = output.argmax(dim=-1)
        label = graph.ndata["label"]
        val_mask, test_mask = graph.ndata["val_mask"], graph.ndata["test_mask"]
        loss = F.cross_entropy(output[val_mask], label[val_mask])
    accs = []
    for mask in [val_mask, test_mask]:
        accs.append(float((pred[mask] == label[mask]).sum() / mask.sum()))
    return loss.item(), accs[0], accs[1]


def main(args):
    # prepare data
    transform = AddSelfLoop()
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    graph = data[0].to(args.device)
    features = graph.ndata["feat"]
    train_mask = graph.ndata["train_mask"]
    in_feats = features.shape[1]
    n_classes = data.num_classes

    # get node embedding
    ggcm = GGCM()
    embeds = ggcm.get_embedding(graph, args)

    # create classifier model
    classifier = Classifier(in_feats, n_classes)
    optimizer = optim.Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.wd
    )

    # train classifier
    best_acc = -1
    for i in range(args.epochs):
        classifier.train()
        output = classifier(embeds)
        loss = F.cross_entropy(
            output[train_mask], graph.ndata["label"][train_mask]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val, acc_val, acc_test = evaluate(classifier, embeds, graph)
        if acc_val > best_acc:
            best_acc, best_model = acc_val, copy.deepcopy(classifier)

        print(f"{i+1} {loss_val:.4f} {acc_val:.3f} acc_test={acc_test:.3f}")

    _, _, acc_test = evaluate(best_model, embeds, graph)
    print(f"Final test acc: {acc_test:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGCM")
    parser.add_argument(
        "--dataset",
        type=str,
        default="citeseer",
        choices=["citeseer", "cora", "pubmed"],
        help="Dataset to use.",
    )
    parser.add_argument("--decline", type=float, default=1, help="Decline.")
    parser.add_argument("--alpha", type=float, default=0.15, help="Alpha.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--lr", type=float, default=0.13, help="Initial learning rate."
    )
    parser.add_argument(
        "--layer_num", type=int, default=16, help="Degree of the approximation."
    )
    parser.add_argument(
        "--negative_rate",
        type=float,
        default=20.0,
        help="Negative sampling rate for a negative graph.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        nargs="*",
        default=2e-3,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument(
        "--decline_neg", type=float, default=1.0, help="Decline negative."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="device to use",
    )
    args, _ = parser.parse_known_args()

    main(args)
