import argparse
import time
import copy

import torch.nn.functional as F
import torch.optim as optim

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from ggcm import GGCM
from utils import model_test, symmetric_normalize_adjacency


def train(model, embedds, args):
    # Evaluate embedding by classification with the given split setting
    best_acc = -1
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for i in range(args.epochs):
        model.train()
        output = model(embedds)
        loss = F.cross_entropy(output[model.train_mask], model.label[model.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val, acc_val, acc_test = model_test(model, embedds)
        if acc_val > best_acc:
            best_acc, best_model = acc_val, copy.deepcopy(model)

        print(f'{i+1} {loss_val:.4f} {acc_val:.3f} acc_test={acc_test:.3f}')

    loss_val, acc_val, acc_test = model_test(best_model, embedds)
    return acc_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default="citeseer",
        help='Dataset to use.',
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--degree', type=int, default=16)
    parser.add_argument('--decline', type=float, default=1)
    parser.add_argument('--negative_rate', type=float, default=20.0)
    parser.add_argument('--wd', type=float, nargs='*', default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.12)
    parser.add_argument('--decline_neg', type=float, default=1.0)
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='device to use',
    )
    args, _ = parser.parse_known_args()

    transform = (AddSelfLoop())
    if args.dataset == "cora":
        num_edges = CoraGraphDataset()[0].num_edges()
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        num_edges = CiteseerGraphDataset()[0].num_edges()
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        num_edges = PubmedGraphDataset()[0].num_edges()
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    graph = data[0]
    graph = graph.to(args.device)
    features = graph.ndata["feat"]
    adj = symmetric_normalize_adjacency(graph)

    avg_edge_num = int(args.negative_rate * num_edges / features.shape[0])
    avg_edge_num = ((avg_edge_num + 1) // 2) * 2

    model = GGCM(graph, args).to(args.device)
    start_time = time.time()
    embedds = GGCM.update_embedds(features, adj, avg_edge_num, args)
    test_acc = train(model, embedds, args)
    print(f'Final test acc: {test_acc:.4f}')
    print(f'Total Time: {time.time() - start_time:.4f}')
