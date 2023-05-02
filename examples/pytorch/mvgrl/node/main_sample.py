import argparse
import random
import warnings

import dgl

import numpy as np
import torch as th
import torch.nn as nn

warnings.filterwarnings("ignore")

from dataset import process_dataset, process_dataset_appnp
from model import LogReg, MVGRL

parser = argparse.ArgumentParser(description="mvgrl")

parser.add_argument(
    "--dataname", type=str, default="cora", help="Name of dataset."
)
parser.add_argument(
    "--gpu", type=int, default=-1, help="GPU index. Default: -1, using cpu."
)
parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="Patient epochs to wait before early stopping.",
)
parser.add_argument(
    "--lr1", type=float, default=0.001, help="Learning rate of mvgrl."
)
parser.add_argument(
    "--lr2", type=float, default=0.01, help="Learning rate of linear evaluator."
)
parser.add_argument(
    "--wd1", type=float, default=0.0, help="Weight decay of mvgrl."
)
parser.add_argument(
    "--wd2", type=float, default=0.0, help="Weight decay of linear evaluator."
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.01,
    help="Edge mask threshold of diffusion graph.",
)
parser.add_argument(
    "--hid_dim", type=int, default=512, help="Hidden layer dim."
)
parser.add_argument(
    "--sample_size", type=int, default=2000, help="Subgraph size."
)

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

if __name__ == "__main__":
    print(args)

    # Step 1: Prepare data =================================================================== #
    if args.dataname == "pubmed":
        (
            graph,
            diff_graph,
            feat,
            label,
            train_idx,
            val_idx,
            test_idx,
            edge_weight,
        ) = process_dataset_appnp(args.epsilon)
    else:
        (
            graph,
            diff_graph,
            feat,
            label,
            train_idx,
            val_idx,
            test_idx,
            edge_weight,
        ) = process_dataset(args.dataname, args.epsilon)
    edge_weight = th.tensor(edge_weight).float()
    graph.ndata["feat"] = feat
    diff_graph.edata["edge_weight"] = edge_weight

    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]
    edge_weight = th.tensor(edge_weight).float()

    train_idx = train_idx.to(args.device)
    val_idx = val_idx.to(args.device)
    test_idx = test_idx.to(args.device)

    n_node = graph.num_nodes()

    sample_size = args.sample_size

    lbl1 = th.ones(sample_size * 2)
    lbl2 = th.zeros(sample_size * 2)
    lbl = th.cat((lbl1, lbl2))
    lbl = lbl.to(args.device)

    # Step 2: Create model =================================================================== #
    model = MVGRL(n_feat, args.hid_dim)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr1, weight_decay=args.wd1
    )
    loss_fn = nn.BCEWithLogitsLoss()

    node_list = list(range(n_node))

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        sample_idx = random.sample(node_list, sample_size)

        g = dgl.node_subgraph(graph, sample_idx)
        dg = dgl.node_subgraph(diff_graph, sample_idx)

        f = g.ndata.pop("feat")
        ew = dg.edata.pop("edge_weight")

        shuf_idx = np.random.permutation(sample_size)
        sf = f[shuf_idx, :]

        g = g.to(args.device)
        dg = dg.to(args.device)
        f = f.to(args.device)
        ew = ew.to(args.device)

        sf = sf.to(args.device)

        out = model(g, dg, f, sf, ew)
        loss = loss_fn(out, lbl)

        loss.backward()
        optimizer.step()

        print("Epoch: {0}, Loss: {1:0.4f}".format(epoch, loss.item()))

        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), "model.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("Early stopping")
            break

    model.load_state_dict(th.load("model.pkl"))

    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    feat = feat.to(args.device)
    edge_weight = edge_weight.to(args.device)
    embeds = model.get_embedding(graph, diff_graph, feat, edge_weight)

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    label = label.to(args.device)
    train_labels = label[train_idx]
    test_labels = label[test_idx]
    accs = []

    # Step 5:  Linear evaluation ========================================================== #
    for _ in range(5):
        model = LogReg(args.hid_dim, n_classes)
        opt = th.optim.Adam(
            model.parameters(), lr=args.lr2, weight_decay=args.wd2
        )

        model = model.to(args.device)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(300):
            model.train()
            opt.zero_grad()
            logits = model(train_embs)
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

        model.eval()
        logits = model(test_embs)
        preds = th.argmax(logits, dim=1)
        acc = th.sum(preds == test_labels).float() / test_labels.shape[0]
        accs.append(acc * 100)

    accs = th.stack(accs)
    print(accs.mean().item(), accs.std().item())
