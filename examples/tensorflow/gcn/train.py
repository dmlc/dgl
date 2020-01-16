import argparse
import time
import numpy as np
import networkx as nx
import tensorflow as tf
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gcn import GCN


def evaluate(model, features, labels, mask):
    logits = model(features, training=False)
    logits = logits[mask]
    labels = labels[mask]
    indices = tf.math.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
    return acc.numpy().item()


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.gpu < 0:
        device = "/cpu:0"
    else:
        device = "/gpu:{}".format(args.gpu)

    with tf.device(device):
        features = tf.convert_to_tensor(data.features, dtype=tf.float32)
        labels = tf.convert_to_tensor(data.labels, dtype=tf.int64)
        train_mask = tf.convert_to_tensor(data.train_mask, dtype=tf.bool)
        val_mask = tf.convert_to_tensor(data.val_mask, dtype=tf.bool)
        test_mask = tf.convert_to_tensor(data.test_mask, dtype=tf.bool)
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
        print("""----Data statistics------'
        #Edges %d
        #Classes %d
        #Train samples %d
        #Val samples %d
        #Test samples %d""" %
              (n_edges, n_classes,
               train_mask.numpy().sum(),
               val_mask.numpy().sum(),
               test_mask.numpy().sum()))

        # graph preprocess and calculate normalization factor
        g = data.graph
        if args.self_loop:
            g.remove_edges_from(nx.selfloop_edges(g))
            g.add_edges_from(zip(g.nodes(), g.nodes()))
        g = DGLGraph(g)
        n_edges = g.number_of_edges()
        # normalization
        degs = tf.cast(tf.identity(g.in_degrees()), dtype=tf.float32)
        norm = tf.math.pow(degs, -0.5)
        norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)

        g.ndata['norm'] = tf.expand_dims(norm, -1)

        # create GCN model
        model = GCN(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    tf.nn.relu,
                    args.dropout)

        loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # use optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.lr, epsilon=1e-8)

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
                    loss_value = loss_value + \
                        args.weight_decay*tf.nn.l2_loss(weight)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, features, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss_value.numpy().item(),
                                                 acc, n_edges / np.mean(dur) / 1000))

        acc = evaluate(model, features, labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
