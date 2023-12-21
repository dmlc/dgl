"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn
GCN with batch processing
"""
import argparse
import time

import dgl
import dgl.function as fn
import mxnet as mx
import numpy as np
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)
from mxnet import gluon


class GCNLayer(gluon.Block):
    def __init__(self, g, out_feats, activation, dropout):
        super(GCNLayer, self).__init__()
        self.g = g
        self.dense = gluon.nn.Dense(out_feats, activation)
        self.dropout = dropout

    def forward(self, h):
        self.g.ndata["h"] = h * self.g.ndata["out_norm"]
        self.g.update_all(
            fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="accum")
        )
        accum = self.g.ndata.pop("accum")
        accum = self.dense(accum * self.g.ndata["in_norm"])
        if self.dropout:
            accum = mx.nd.Dropout(accum, p=self.dropout)
        h = self.g.ndata.pop("h")
        h = mx.nd.concat(h / self.g.ndata["out_norm"], accum, dim=1)
        return h


class GCN(gluon.Block):
    def __init__(self, g, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.inp_layer = gluon.nn.Dense(n_hidden, activation)
        self.dropout = dropout
        self.layers = gluon.nn.Sequential()
        for i in range(n_layers):
            self.layers.add(GCNLayer(g, n_hidden, activation, dropout))
        self.out_layer = gluon.nn.Dense(n_classes)

    def forward(self, features):
        emb_inp = [features, self.inp_layer(features)]
        if self.dropout:
            emb_inp[-1] = mx.nd.Dropout(emb_inp[-1], p=self.dropout)
        h = mx.nd.concat(*emb_inp, dim=1)
        for layer in self.layers:
            h = layer(h)
        h = self.out_layer(h)
        return h


def evaluate(model, features, labels, mask):
    pred = model(features).argmax(axis=1)
    accuracy = ((pred == labels) * mask).sum() / mask.sum().asscalar()
    return accuracy.asscalar()


def main(args):
    # load and preprocess dataset
    if args.dataset == "cora":
        data = CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    g = data[0]
    if args.gpu < 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        ctx = mx.gpu(args.gpu)
        g = g.to(ctx)

    features = g.ndata["feat"]
    labels = mx.nd.array(g.ndata["label"], dtype="float32", ctx=ctx)
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = data.graph.number_of_edges()
    print(
        """----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d"""
        % (
            n_edges,
            n_classes,
            train_mask.sum().asscalar(),
            val_mask.sum().asscalar(),
            test_mask.sum().asscalar(),
        )
    )

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    # normalization
    in_degs = g.in_degrees().astype("float32")
    out_degs = g.out_degrees().astype("float32")
    in_norm = mx.nd.power(in_degs, -0.5)
    out_norm = mx.nd.power(out_degs, -0.5)
    if cuda:
        in_norm = in_norm.as_in_context(ctx)
        out_norm = out_norm.as_in_context(ctx)
    g.ndata["in_norm"] = mx.nd.expand_dims(in_norm, 1)
    g.ndata["out_norm"] = mx.nd.expand_dims(out_norm, 1)

    model = GCN(
        g,
        args.n_hidden,
        n_classes,
        args.n_layers,
        "relu",
        args.dropout,
    )
    model.initialize(ctx=ctx)
    n_train_samples = train_mask.sum().asscalar()
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(
        model.collect_params(),
        "adam",
        {"learning_rate": args.lr, "wd": args.weight_decay},
    )

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            pred = model(features)
            loss = loss_fcn(pred, labels, mx.nd.expand_dims(train_mask, 1))
            loss = loss.sum() / n_train_samples

        loss.backward()
        trainer.step(batch_size=1)

        if epoch >= 3:
            dur.append(time.time() - t0)
            acc = evaluate(model, features, labels, val_mask)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    np.mean(dur),
                    loss.asscalar(),
                    acc,
                    n_edges / np.mean(dur) / 1000,
                )
            )

    # test set accuracy
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    register_data_args(parser)
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--normalization",
        choices=["sym", "left"],
        default=None,
        help="graph normalization types (default=None)",
    )
    parser.add_argument(
        "--self-loop",
        action="store_true",
        help="graph self-loop (default=False)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()

    print(args)

    main(args)
