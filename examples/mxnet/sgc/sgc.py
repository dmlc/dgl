"""
This code was modified from the GCN implementation in DGL examples.
Simplifying Graph Convolutional Networks
Paper: https://arxiv.org/abs/1902.07153
Code: https://github.com/Tiiiger/SGC
SGC implementation in DGL.
"""
import argparse
import math
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
from dgl.nn.mxnet.conv import SGConv
from mxnet import gluon, nd
from mxnet.gluon import nn


def evaluate(model, g, features, labels, mask):
    pred = model(g, features).argmax(axis=1)
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
        g = g.int().to(ctx)

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

    # create SGC model
    model = SGConv(in_feats, n_classes, k=2, cached=True, bias=args.bias)

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
            pred = model(g, features)
            loss = loss_fcn(pred, labels, mx.nd.expand_dims(train_mask, 1))
            loss = loss.sum() / n_train_samples

        loss.backward()
        trainer.step(batch_size=1)

        if epoch >= 3:
            loss.asscalar()
            dur.append(time.time() - t0)
            acc = evaluate(model, g, features, labels, val_mask)
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
    acc = evaluate(model, g, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGC")
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument(
        "--bias", action="store_true", default=False, help="flag to use bias"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-6, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
