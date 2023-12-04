import argparse
import time

import dgl

import mxnet as mx
import numpy as np
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)
from dgl.nn.mxnet.conv import APPNPConv
from mxnet import gluon, nd
from mxnet.gluon import nn


class APPNP(nn.Block):
    def __init__(
        self,
        g,
        in_feats,
        hiddens,
        n_classes,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.g = g

        with self.name_scope():
            self.layers = nn.Sequential()
            # input layer
            self.layers.add(nn.Dense(hiddens[0], in_units=in_feats))
            # hidden layers
            for i in range(1, len(hiddens)):
                self.layers.add(nn.Dense(hiddens[i], in_units=hiddens[i - 1]))
            # output layer
            self.layers.add(nn.Dense(n_classes, in_units=hiddens[-1]))
            self.activation = activation
            if feat_drop:
                self.feat_drop = nn.Dropout(feat_drop)
            else:
                self.feat_drop = lambda x: x
            self.propagate = APPNPConv(k, alpha, edge_drop)

    def forward(self, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
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
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create APPNP model
    model = APPNP(
        g,
        in_feats,
        args.hidden_sizes,
        n_classes,
        nd.relu,
        args.in_drop,
        args.edge_drop,
        args.alpha,
        args.k,
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
            loss.asscalar()
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
    parser = argparse.ArgumentParser(description="APPNP")
    register_data_args(parser)
    parser.add_argument(
        "--in-drop", type=float, default=0.5, help="input feature dropout"
    )
    parser.add_argument(
        "--edge-drop", type=float, default=0.5, help="edge propagation dropout"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[64],
        help="hidden unit sizes for appnp",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of propagation steps"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Teleport Probability"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
