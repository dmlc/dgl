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
import mxnet as mx
from mxnet import gluon
import mxnet.ndarray as F
from dgl import DGLGraph
from dgl.nn.mxnet import RelGraphConv
from dgl.contrib.data import load_data
from functools import partial

from model import BaseRGCN

class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=partial(F.softmax, axis=1),
                self_loop=self.use_self_loop)

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

    train_idx = mx.nd.array(train_idx)
    # since the nodes are featureless, the input feature is then the node id.
    feats = mx.nd.arange(num_nodes, dtype='int32')
    # edge type and normalization factor
    edge_type = mx.nd.array(data.edge_type, dtype='int32')
    edge_norm = mx.nd.array(data.edge_norm).expand_dims(1)
    labels = mx.nd.array(labels).reshape((-1))

    # check cuda
    use_cuda = args.gpu >= 0
    if use_cuda:
        ctx = mx.gpu(args.gpu)
        feats = feats.as_in_context(ctx)
        edge_type = edge_type.as_in_context(ctx)
        edge_norm = edge_norm.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        train_idx = train_idx.as_in_context(ctx)
    else:
        ctx = mx.cpu(0)

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
                           gpu_id=args.gpu)
    model.initialize(ctx=ctx)

    # optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr, 'wd': args.l2norm})
    loss_fcn = gluon.loss.SoftmaxCELoss(from_logits=False)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    for epoch in range(args.n_epochs):
        t0 = time.time()
        with mx.autograd.record():
            pred = model(g, feats, edge_type, edge_norm)
            loss = loss_fcn(pred[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        trainer.step(len(train_idx))
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = F.sum(pred[train_idx].argmax(axis=1) == labels[train_idx]).asscalar() / train_idx.shape[0]
        val_acc = F.sum(pred[val_idx].argmax(axis=1) == labels[val_idx]).asscalar() / len(val_idx)
        print("Train Accuracy: {:.4f} | Validation Accuracy: {:.4f}".format(train_acc, val_acc))
    print()

    logits = model.forward(g, feats, edge_type, edge_norm)
    test_acc = F.sum(logits[test_idx].argmax(axis=1) == labels[test_idx]).asscalar() / len(test_idx)
    print("Test Accuracy: {:.4f}".format(test_acc))
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
