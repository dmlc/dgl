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
from functools import partial

def gcn_msg(edge, normalization=None):
    # print('h', edge.src['h'].shape, edge.src['out_degree'])
    msg = edge.src['h']
    if normalization == 'sym':
        msg = msg / edge.src['out_degree'].sqrt().reshape((-1,1))
    return {'m': msg}


def gcn_reduce(node, normalization=None):
    # print('m', node.mailbox['m'].shape, node.data['in_degree'])
    accum = mx.nd.sum(node.mailbox['m'], 1)
    if normalization == 'sym':
        accum = accum / node.data['in_degree'].sqrt().reshape((-1,1))
    elif normalization == 'asym':
        accum = accum / node.data['in_degree'].reshape((-1,1))
    return {'accum': accum}


class NodeUpdateModule(gluon.Block):
    def __init__(self, out_feats, activation=None, dropout=0):
        super(NodeUpdateModule, self).__init__()
        self.linear = gluon.nn.Dense(out_feats, activation=activation)
        self.dropout = dropout

    def forward(self, node):
        accum = self.linear(node.data['accum'])
        if self.dropout:
            accum = mx.nd.Dropout(accum, p=self.dropout)
        return {'h': mx.nd.concat(node.data['h'], accum, dim=1)}


class GCN(gluon.Block):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization,
                 ):
        super(GCN, self).__init__()
        self.g = g
        self.dropout = dropout

        self.inp_layer = gluon.nn.Dense(n_hidden, activation)

        self.conv_layers = gluon.nn.Sequential()
        for i in range(n_layers):
            self.conv_layers.add(NodeUpdateModule(n_hidden, activation, dropout))

        self.out_layer = gluon.nn.Dense(n_classes)

        self.gcn_msg = partial(gcn_msg, normalization=normalization)
        self.gcn_reduce = partial(gcn_reduce, normalization=normalization)


    def forward(self, features):
        emb_inp = [features, self.inp_layer(features)]
        if self.dropout:
            emb_inp[-1] = mx.nd.Dropout(emb_inp[-1], p=self.dropout)

        self.g.ndata['h'] = mx.nd.concat(*emb_inp, dim=1)
        for layer in self.conv_layers:
            self.g.update_all(self.gcn_msg, self.gcn_reduce, layer)

        emb_out = self.g.ndata.pop('h')
        return self.out_layer(emb_out)


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.self_loop:
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    mask = mx.nd.array(data.train_mask)
    in_degree  = mx.nd.array([data.graph.in_degree(i)
                              for i in range(len(data.graph))])
    out_degree = mx.nd.array([data.graph.out_degree(i)
                              for i in range(len(data.graph))])

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
        in_degree = in_degree.as_in_context(mx.gpu(0))
        out_degree = out_degree.as_in_context(mx.gpu(0))
        ctx = mx.gpu(0)

    # create GCN model
    g = DGLGraph(data.graph)
    g.ndata['in_degree']  = in_degree
    g.ndata['out_degree'] = out_degree

    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                'relu',
                args.dropout,
                args.normalization,
                )
    model.initialize(ctx=ctx)
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            pred = model(features)
            loss = loss_fcn(pred, labels, mask)

        #optimizer.zero_grad()
        loss.backward()
        trainer.step(features.shape[0])

        if epoch >= 3:
            dur.append(time.time() - t0)
            print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
                epoch, loss.asnumpy()[0], np.mean(dur), n_edges / np.mean(dur) / 1000))

    # test set accuracy
    pred = model(features)
    accuracy = (pred*100).softmax().pick(labels).mean()
    print("Final accuracy {:.2%}".format(accuracy.mean().asscalar()))
    return accuracy.mean().asscalar()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--wd", type=float, default=5e-4,
            help="weight decay")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--normalization",
            choices=['sym','asym'], default=None,
            help="graph normalization types (default=None)")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    args = parser.parse_args()

    print(args)

    main(args)
