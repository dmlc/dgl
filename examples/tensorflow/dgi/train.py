import argparse
import time

import dgl

import networkx as nx
import numpy as np
import tensorflow as tf
from dgi import Classifier, DGI
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    register_data_args,
)
from tensorflow.keras import layers


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
        n_edges = g.number_of_edges()

        # add self loop
        if args.self_loop:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        n_edges = g.number_of_edges()

        # create DGI model
        dgi = DGI(
            g,
            in_feats,
            args.n_hidden,
            args.n_layers,
            tf.keras.layers.PReLU(
                alpha_initializer=tf.constant_initializer(0.25)
            ),
            args.dropout,
        )

        dgi_optimizer = tf.keras.optimizers.Adam(learning_rate=args.dgi_lr)

        # train deep graph infomax
        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []
        for epoch in range(args.n_dgi_epochs):
            if epoch >= 3:
                t0 = time.time()

            with tf.GradientTape() as tape:
                loss = dgi(features)
                # Manually Weight Decay
                # We found Tensorflow has a different implementation on weight decay
                # of Adam(W) optimizer with PyTorch. And this results in worse results.
                # Manually adding weights to the loss to do weight decay solves this problem.
                for weight in dgi.trainable_weights:
                    loss = loss + args.weight_decay * tf.nn.l2_loss(weight)
                grads = tape.gradient(loss, dgi.trainable_weights)
                dgi_optimizer.apply_gradients(zip(grads, dgi.trainable_weights))

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                dgi.save_weights("best_dgi.pkl")
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping!")
                break

            if epoch >= 3:
                dur.append(time.time() - t0)

            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    np.mean(dur),
                    loss.numpy().item(),
                    n_edges / np.mean(dur) / 1000,
                )
            )

        # create classifier model
        classifier = Classifier(args.n_hidden, n_classes)

        classifier_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.classifier_lr
        )

        # train classifier
        print("Loading {}th epoch".format(best_t))
        dgi.load_weights("best_dgi.pkl")
        embeds = dgi.encoder(features, corrupt=False)
        embeds = tf.stop_gradient(embeds)
        dur = []
        loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        for epoch in range(args.n_classifier_epochs):
            if epoch >= 3:
                t0 = time.time()
            with tf.GradientTape() as tape:
                preds = classifier(embeds)
                loss = loss_fcn(labels[train_mask], preds[train_mask])
                # Manually Weight Decay
                # We found Tensorflow has a different implementation on weight decay
                # of Adam(W) optimizer with PyTorch. And this results in worse results.
                # Manually adding weights to the loss to do weight decay solves this problem.
                # In original code, there's no weight decay applied in this part
                # link: https://github.com/PetarV-/DGI/blob/master/execute.py#L121
                # for weight in classifier.trainable_weights:
                #     loss = loss + \
                #         args.weight_decay * tf.nn.l2_loss(weight)
                grads = tape.gradient(loss, classifier.trainable_weights)
                classifier_optimizer.apply_gradients(
                    zip(grads, classifier.trainable_weights)
                )
            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(classifier, embeds, labels, val_mask)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    np.mean(dur),
                    loss.numpy().item(),
                    acc,
                    n_edges / np.mean(dur) / 1000,
                )
            )

        print()
        acc = evaluate(classifier, embeds, labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGI")
    register_data_args(parser)
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument(
        "--dgi-lr", type=float, default=1e-3, help="dgi learning rate"
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=1e-2,
        help="classifier learning rate",
    )
    parser.add_argument(
        "--n-dgi-epochs",
        type=int,
        default=300,
        help="number of training epochs",
    )
    parser.add_argument(
        "--n-classifier-epochs",
        type=int,
        default=300,
        help="number of training epochs",
    )
    parser.add_argument(
        "--n-hidden", type=int, default=512, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="early stop patience condition"
    )
    parser.add_argument(
        "--self-loop",
        action="store_true",
        help="graph self-loop (default=False)",
    )
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
