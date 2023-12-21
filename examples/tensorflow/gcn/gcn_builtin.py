import argparse
import math
import time

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import tensorflow as tf
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)
from tensorflow.keras import layers


class GCNLayer(layers.Layer):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.g = g

        w_init = tf.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_out", distribution="uniform"
        )
        self.weight = tf.Variable(
            initial_value=w_init(shape=(in_feats, out_feats), dtype="float32"),
            trainable=True,
        )
        if dropout:
            self.dropout = layers.Dropout(rate=dropout)
        else:
            self.dropout = 0.0
        if bias:
            b_init = tf.zeros_initializer()
            self.bias = tf.Variable(
                initial_value=b_init(shape=(out_feats,), dtype="float32"),
                trainable=True,
            )
        else:
            self.bias = None
        self.activation = activation

    def call(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata["h"] = tf.matmul(h, self.weight)
        self.g.ndata["norm_h"] = self.g.ndata["h"] * self.g.ndata["norm"]
        self.g.update_all(fn.copy_u("norm_h", "m"), fn.sum("m", "h"))
        h = self.g.ndata["h"]
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GCN(layers.Layer):
    def __init__(
        self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.layers = []

        # input layer
        self.layers.append(GCNLayer(g, in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GCNLayer(g, n_hidden, n_hidden, activation, dropout)
            )
        # output layer
        self.layers.append(GCNLayer(g, n_hidden, n_classes, None, dropout))

    def call(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


def evaluate(model, features, labels, mask):
    logits = model(features, training=False)
    logits = logits[mask]
    labels = labels[mask]
    indices = tf.math.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
    return acc.numpy().item()


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
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(args.gpu)
        g = g.to(device)

    with tf.device(device):
        features = g.ndata["feat"]
        labels = g.ndata["label"]
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
                train_mask.numpy().sum(),
                val_mask.numpy().sum(),
                test_mask.numpy().sum(),
            )
        )

        # add self loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        n_edges = g.number_of_edges()
        # # normalization
        degs = tf.cast(tf.identity(g.in_degrees()), dtype=tf.float32)
        norm = tf.math.pow(degs, -0.5)
        norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)

        g.ndata["norm"] = tf.expand_dims(norm, -1)

        # create GCN model
        model = GCN(
            g,
            in_feats,
            args.n_hidden,
            n_classes,
            args.n_layers,
            tf.nn.relu,
            args.dropout,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

        loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        # initialize graph
        dur = []
        for epoch in range(args.n_epochs):
            if epoch >= 3:
                t0 = time.time()
            # forward
            with tf.GradientTape() as tape:
                logits = model(features)
                loss_value = loss_fcn(labels[train_mask], logits[train_mask])
                # Manually Weight Decay
                # We found Tensorflow has a different implementation on weight decay
                # of Adam(W) optimizer with PyTorch. And this results in worse results.
                # Manually adding weights to the loss to do weight decay solves this problem.
                for weight in model.trainable_weights:
                    loss_value = loss_value + args.weight_decay * tf.nn.l2_loss(
                        weight
                    )

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, features, labels, val_mask)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    np.mean(dur),
                    loss_value.numpy().item(),
                    acc,
                    n_edges / np.mean(dur) / 1000,
                )
            )

        acc = evaluate(model, features, labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))


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
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()
    print(args)

    main(args)
