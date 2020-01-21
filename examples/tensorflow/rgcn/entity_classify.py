"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers
from dgl import DGLGraph
from dgl.nn.tensorflow import RelGraphConv
from dgl.contrib.data import load_data
from functools import partial

from model import BaseRGCN

class EntityClassify(BaseRGCN):
    def create_features(self):
        features = tf.range(self.num_nodes)
        return features

    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=tf.nn.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=tf.nn.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=partial(tf.nn.softmax, axis=1),
                self_loop=self.use_self_loop)

def acc(logits, labels, mask):
    logits = tf.gather(logits, mask)
    labels = tf.gather(labels, mask)
    indices = tf.math.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(indices == labels, dtype=tf.float32))
    return acc

def main(args):
    # load graph data
    data = load_data(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes
    labels = data.labels
    train_idx = data.train_idx
    test_idx = data.test_idx

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # since the nodes are featureless, the input feature is then the node id.
    feats = tf.range(num_nodes, dtype=tf.int64)

    # edge type and normalization factor
    edge_type = tf.convert_to_tensor(data.edge_type)
    edge_norm = tf.expand_dims(tf.convert_to_tensor(data.edge_norm), 1)
    labels = tf.reshape(tf.convert_to_tensor(labels), (-1, ))

    # check cuda
    if args.gpu < 0:
        device = "/cpu:0"
        use_cuda = False
    else:
        device = "/gpu:{}".format(args.gpu)
        use_cuda = True
    
    with tf.device(device):

        # create graph
        g = DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(data.edge_src, data.edge_dst)

        # create model
        model = EntityClassify(len(g),
                            args.n_hidden,
                            num_classes,
                            num_rels,
                            num_bases=args.n_bases,
                            num_hidden_layers=args.n_layers - 2,
                            dropout=args.dropout,
                            use_self_loop=args.use_self_loop,
                            use_cuda=use_cuda)

        # optimizer
        optimizer = tf.keras.optimizers.Adam(
                    learning_rate=args.lr)
        # training loop
        print("start training...")
        forward_time = []
        backward_time = []
        loss_fcn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)
        for epoch in range(args.n_epochs):
            t0 = time.time()
            with tf.GradientTape() as tape:
                logits = model(g, feats, edge_type, edge_norm)
                loss = loss_fcn(tf.gather(labels, train_idx), tf.gather(logits, train_idx))
                # Manually Weight Decay
                # We found Tensorflow has a different implementation on weight decay 
                # of Adam(W) optimizer with PyTorch. And this results in worse results.
                # Manually adding weights to the loss to do weight decay solves this problem.
                for weight in model.trainable_weights:
                    loss = loss + \
                        args.l2norm * tf.nn.l2_loss(weight)
                t1 = time.time()
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                format(epoch, forward_time[-1], backward_time[-1]))
            train_acc = acc(logits, labels, train_idx)
            val_loss = loss_fcn(tf.gather(labels, val_idx), tf.gather(logits, val_idx))
            val_acc = acc(logits, labels, val_idx)
            print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                format(train_acc, loss.numpy().item(), val_acc, val_loss.numpy().item()))
        print()

        logits = model(g, feats, edge_type, edge_norm)
        test_loss = loss_fcn(tf.gather(labels, test_idx), tf.gather(logits, test_idx))
        test_acc = acc(logits, labels, test_idx)
        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.numpy().item()))
        print()

        print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
