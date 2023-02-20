import argparse
import time

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
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


class SIGN(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden,
        out_feats,
        num_hops,
        n_layers,
        dropout,
        input_drop,
    ):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.R + 1):
        g.update_all(
            fn.copy_u(f"feat_{hop-1}", "msg"), fn.mean("msg", f"feat_{hop}")
        )
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))

    if args.dataset == "ogbn-mag":
        # For MAG dataset, only return features for target node types (i.e.
        # paper nodes)
        target_mask = g.ndata["target_mask"]
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = []
        for x in res:
            feat = torch.zeros(
                (num_target,) + x.shape[1:], dtype=x.dtype, device=x.device
            )
            feat[target_ids] = x[target_mask]
            new_res.append(feat)
        res = new_res
    return res


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_dataset(args.dataset, device)
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    in_feats = g.ndata["feat"].shape[1]
    feats = neighbor_average_features(g, args)
    labels = labels.to(device)
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    return (
        feats,
        labels,
        in_feats,
        n_classes,
        train_nid,
        val_nid,
        test_nid,
        evaluator,
    )


def train(model, feats, labels, loss_fcn, optimizer, train_loader):
    model.train()
    device = labels.device
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        loss = loss_fcn(model(batch_feats), labels[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(
    model, feats, labels, test_loader, evaluator, train_nid, val_nid, test_nid
):
    model.eval()
    device = labels.device
    preds = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        preds.append(torch.argmax(model(batch_feats), dim=-1))
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    train_res = evaluator(preds[train_nid], labels[train_nid])
    val_res = evaluator(preds[val_nid], labels[val_nid])
    test_res = evaluator(preds[test_nid], labels[test_nid])
    return train_res, val_res, test_res


def run(args, data, device):
    (
        feats,
        labels,
        in_size,
        num_classes,
        train_nid,
        val_nid,
        test_nid,
        evaluator,
    ) = data
    train_loader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.arange(labels.shape[0]),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Initialize model and optimizer for each run
    num_hops = args.R + 1
    model = SIGN(
        in_size,
        args.num_hidden,
        num_classes,
        num_hops,
        args.ff_layer,
        args.dropout,
        args.input_dropout,
    )
    model = model.to(device)
    print("# Params:", get_n_params(model))

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        train(model, feats, labels, loss_fcn, optimizer, train_loader)

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                acc = test(
                    model,
                    feats,
                    labels,
                    test_loader,
                    evaluator,
                    train_nid,
                    val_nid,
                    test_nid,
                )
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Acc: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(*acc)
            print(log)
            if acc[1] > best_val:
                best_epoch = epoch
                best_val = acc[1]
                best_test = acc[2]

    print(
        "Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
            best_epoch, best_val, best_test
        )
    )
    return best_val, best_test


def main(args):
    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    with torch.no_grad():
        data = prepare_data(device, args)
    val_accs = []
    test_accs = []
    for i in range(args.num_runs):
        print(f"Run {i} start training")
        best_val, best_test = run(args, data, device)
        val_accs.append(best_val)
        test_accs.append(best_test)

    print(
        f"Average val accuracy: {np.mean(val_accs):.4f}, "
        f"std: {np.std(val_accs):.4f}"
    )
    print(
        f"Average test accuracy: {np.mean(test_accs):.4f}, "
        f"std: {np.std(test_accs):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGN")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--R", type=int, default=5, help="number of hops")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout on activation"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=100000,
        help="evaluation batch size",
    )
    parser.add_argument(
        "--ff-layer", type=int, default=2, help="number of feed-forward layers"
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0,
        help="dropout on input features",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="number of times to repeat the experiment",
    )
    args = parser.parse_args()

    print(args)
    main(args)
