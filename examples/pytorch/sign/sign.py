import argparse
import os
import time

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import load_dataset


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class Model(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, R, n_layers, dropout):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        for hop in range(R + 1):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        # self.linear = nn.Linear(hidden * (R + 1), out_feats)
        self.project = FeedForwardNet(
            (R + 1) * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out


def calc_weight(g):
    """
    Compute row_normalized(D^(-1/2)AD^(-1/2))
    """
    with g.local_scope():
        # compute D^(-0.5)*D(-1/2), assuming A is Identity
        g.ndata["in_deg"] = g.in_degrees().float().pow(-0.5)
        g.ndata["out_deg"] = g.out_degrees().float().pow(-0.5)
        g.apply_edges(fn.u_mul_v("out_deg", "in_deg", "weight"))
        # row-normalize weight
        g.update_all(fn.copy_e("weight", "msg"), fn.sum("msg", "norm"))
        g.apply_edges(fn.e_div_v("weight", "norm", "weight"))
        return g.edata["weight"]


def preprocess(g, features, args):
    """
    Pre-compute the average of n-th hop neighbors
    """
    with torch.no_grad():
        g.edata["weight"] = calc_weight(g)
        g.ndata["feat_0"] = features
        for hop in range(1, args.R + 1):
            g.update_all(
                fn.u_mul_e(f"feat_{hop-1}", "weight", "msg"),
                fn.sum("msg", f"feat_{hop}"),
            )
        res = []
        for hop in range(args.R + 1):
            res.append(g.ndata.pop(f"feat_{hop}"))
        return res


def prepare_data(device, args):
    data = load_dataset(args.dataset)
    g, n_classes, train_nid, val_nid, test_nid = data
    g = g.to(device)
    in_feats = g.ndata["feat"].shape[1]
    feats = preprocess(g, g.ndata["feat"], args)
    labels = g.ndata["label"]
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    train_feats = [x[train_nid] for x in feats]
    train_labels = labels[train_nid]
    return (
        feats,
        labels,
        train_feats,
        train_labels,
        in_feats,
        n_classes,
        train_nid,
        val_nid,
        test_nid,
    )


def evaluate(epoch, args, model, feats, labels, train, val, test):
    with torch.no_grad():
        batch_size = args.eval_batch_size
        if batch_size <= 0:
            pred = model(feats)
        else:
            pred = []
            num_nodes = labels.shape[0]
            n_batch = (num_nodes + batch_size - 1) // batch_size
            for i in range(n_batch):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, num_nodes)
                batch_feats = [feat[batch_start:batch_end] for feat in feats]
                pred.append(model(batch_feats))
            pred = torch.cat(pred)

        pred = torch.argmax(pred, dim=1)
        correct = (pred == labels).float()
        train_acc = correct[train].sum() / len(train)
        val_acc = correct[val].sum() / len(val)
        test_acc = correct[test].sum() / len(test)
        return train_acc, val_acc, test_acc


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    data = prepare_data(device, args)
    (
        feats,
        labels,
        train_feats,
        train_labels,
        in_size,
        num_classes,
        train_nid,
        val_nid,
        test_nid,
    ) = data

    model = Model(
        in_size,
        args.num_hidden,
        num_classes,
        args.R,
        args.ff_layer,
        args.dropout,
    )
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_epoch = 0
    best_val = 0
    best_test = 0

    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        model.train()
        loss = loss_fcn(model(train_feats), train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0:
            model.eval()
            acc = evaluate(
                epoch, args, model, feats, labels, train_nid, val_nid, test_nid
            )
            end = time.time()
            log = "Epoch {}, Times(s): {:.4f}".format(epoch, end - start)
            log += ", Accuracy: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(
                *acc
            )
            print(log)
            if acc[1] > best_val:
                best_val = acc[1]
                best_epoch = epoch
                best_test = acc[2]

    print(
        "Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
            best_epoch, best_val, best_test
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGN")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--R", type=int, default=3, help="number of hops")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=250000,
        help="evaluation batch size, -1 for full batch",
    )
    parser.add_argument(
        "--ff-layer", type=int, default=2, help="number of feed-forward layers"
    )
    args = parser.parse_args()

    print(args)
    main(args)
