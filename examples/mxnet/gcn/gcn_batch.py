"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with batch processing
"""
import argparse
import numpy as np
import time
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def gcn_msg(src, edge):
    return src

def gcn_reduce(node, msgs):
    return mx.nd.sum(msgs, 1)

class NodeUpdateModule(gluon.Block):
    def __init__(self, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear = gluon.nn.Dense(out_feats, activation=activation)

    def forward(self, node):
        return self.linear(node)

class GCN(gluon.Block):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.dropout = dropout
        # input layer
        self.layers = gluon.nn.Sequential()
        self.layers.add(NodeUpdateModule(n_hidden, activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(NodeUpdateModule(n_hidden, activation))
        # output layer
        self.layers.add(NodeUpdateModule(n_classes))

    def forward(self, features):
        self.g.set_n_repr(features)
        for layer in self.layers:
            # apply dropout
            if self.dropout:
                val = F.dropout(self.g.get_n_repr(), p=self.dropout)
                self.g.set_n_repr(val)
            self.g.update_all(gcn_msg, gcn_reduce, layer, batchable=True)
        return self.g.pop_n_repr()

def main(args):
    # load and preprocess dataset
    data = load_data(args)

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    mask = mx.nd.array(data.train_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu <= 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        features = features.as_in_context(mx.gpu(0))
        labels = labels.as_in_context(mx.gpu(0))
        mask = mask.as_in_context(mx.gpu(0))
        ctx = mx.gpu(0)

    # create GCN model
    g = DGLGraph(data.graph)
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                'relu',
                args.dropout)
    model.initialize(ctx=ctx)

    # use optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            logits = model(features)
            loss = mx.nd.softmax_cross_entropy(logits, labels)

        #optimizer.zero_grad()
        loss.backward()
        trainer.step(features.shape[0])

        if epoch >= 3:
            dur.append(time.time() - t0)
            print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
                epoch, loss.asnumpy()[0], np.mean(dur), n_edges / np.mean(dur) / 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    args = parser.parse_args()

    main(args)
