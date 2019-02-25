"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with SPMV optimization
"""
import argparse, time
import numpy as np
import mxnet as mx
from mxnet import gluon

from dgl import DGLGraph
from dgl.nn.mxnet import GraphConv
from dgl.data import register_data_args, load_data

class GCNLayer(gluon.Block):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout=0.):
        super(GCNLayer, self).__init__()
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats, activation=activation)
        self.dropout = gluon.nn.Dropout(rate=dropout)

    def forward(self, h, g):
        if self.dropout:
            h = self.dropout(h)
        return self.graph_conv(h, g)

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
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GCNLayer(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GCNLayer(n_hidden, n_hidden, activation=activation, dropout=dropout))
        # output layer
        self.layers.add(GCNLayer(n_hidden, n_classes, activation=None, dropout=dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h, self.g)
        return h

def evaluate(model, features, labels, mask):
    pred = model(features).argmax(axis=1)
    accuracy = ((pred == labels) * mask).sum() / mask.sum().asscalar()
    return accuracy.asscalar()

def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.self_loop:
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_mask = mx.nd.array(data.train_mask)
    val_mask = mx.nd.array(data.val_mask)
    test_mask = mx.nd.array(data.test_mask)
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
              train_mask.sum().asscalar(),
              val_mask.sum().asscalar(),
              test_mask.sum().asscalar()))

    if args.gpu < 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        ctx = mx.gpu(args.gpu)

    features = features.as_in_context(ctx)
    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    val_mask = val_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    # create GCN model
    g = DGLGraph(data.graph)
    # normalization
    degs = g.in_degrees().astype('float32')
    norm = mx.nd.power(degs, -0.5)
    if cuda:
        norm = norm.as_in_context(ctx)
    g.ndata['norm'] = mx.nd.expand_dims(norm, 1)

    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                mx.nd.relu,
                args.dropout)
    model.initialize(ctx=ctx)
    n_train_samples = train_mask.sum().asscalar()
    loss_fcn = gluon.loss.SoftmaxCELoss()

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
            {'learning_rate': args.lr, 'wd': args.weight_decay})

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
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(
                epoch, np.mean(dur), loss.asscalar(), acc, n_edges / np.mean(dur) / 1000))

    # test set accuracy
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)
