import mxnet as mx
import numpy as np
from dgl import DGLGraph
from mxnet import gluon
import functools
import argparse
import time, datetime
from build_graph import load_data


gmttime = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

parser = argparse.ArgumentParser(description='AUTOREC-inspired models')
parser.add_argument("--dataset", choices=['ml-100k','ml-1m'],
        help="choose which dataset to use")
parser.add_argument("--dropout", type=float, default=0.2,
        help="dropout probability")
parser.add_argument("--gpu", type=int, default=-1,
        help="gpu")
parser.add_argument("--save", default="model.params" + gmttime,
        help="save location for model params")
parser.add_argument("--lr", type=float, default=0.005,
        help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4,
        help="weight decay")
parser.add_argument("--n-epochs", type=int, default=200,
        help="number of training epochs")
parser.add_argument("--n-hidden", type=int, default=32,
        help="number of hidden units")
parser.add_argument("--n-layers", type=int, default=1,
        help="number of gcn layers")
parser.add_argument("--activation", default=None,
        help="activation type")
parser.add_argument("--share", action='store_true',
        help="share weights")
parser.add_argument("--label-rate", type=float, default=0,
        help="split user history to data and label")


def gcn_msg(edge):
    mask = edge.data.get('is_data', edge.data['is_train'])
    mask = mask * edge.data['is_train']

    rmat = mx.nd.one_hot(edge.data['r'], depth=edge.src['W'].shape[1])

    W = (edge.src['W'] * rmat.reshape((0,0,1))).sum(axis=1)

    return { 'W' : W, 'mask' : mask }


def gcn_reduce(node):
    accum  = mx.nd.sum(node.mailbox['W'], axis=1) # nodes, hidden
    degree = mx.nd.sum(node.mailbox['mask'], axis=1) # nodes
    return {'accum' : accum / (degree + 1e-10).sqrt().reshape((0,1))}


class NodeUpdateModule(gluon.Block):
    def __init__(self, out_feats, activation=None, dropout=0):
        super(NodeUpdateModule, self).__init__()
        self.linear = gluon.nn.Dense(out_feats, activation=activation)
        self.dropout = gluon.nn.Dropout(dropout)


    def forward(self, node):
        h = self.linear(node.data['h'] + node.data['accum']) # hidden
        h = self.dropout(h)
        return {'h': h}


def _cumsum(data, axis=1):
    _data   = data if axis==0 else data.swapaxes(0,1)
    _out, _ = mx.nd.contrib.foreach(
        lambda arr, psum: (arr+psum, arr+psum),
        _data, mx.nd.zeros_like(_data[0]))
    out     = _out if axis==0 else _out.swapaxes(0,1)
    return out


def quad_fcn(edge):
    user2item = edge.data['user2item']
    item_b = mx.nd.where(user2item, edge.dst['b'], edge.src['b']) # edges, classes
    user_h = mx.nd.where(user2item, edge.src['h'], edge.dst['h']) # edges, hidden
    item_W = mx.nd.where(user2item, edge.dst['W'], edge.src['W']) # edges, classes, hidden

    score = item_b + (item_W * user_h.reshape((0,1,-1))).sum(axis=2) # edges, classes

    return {'pred' : _cumsum(score, axis=1).sum(axis=1)}


class CF_NADE(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 share,
                 ):
        super(CF_NADE, self).__init__()
        self.n_hidden   = n_hidden
        self.n_classes  = n_classes
        self.dropout    = dropout
        self.share      = share

        with self.name_scope():
            self.W = self.params.get('W', shape=(in_feats, n_classes, n_hidden))
            self.b = self.params.get('b', shape=(in_feats, n_classes))
            self.c = self.params.get('c', shape=(in_feats, n_hidden))
            self.dropout = gluon.nn.Dropout(dropout)

            self.conv_layers = gluon.nn.Sequential()
            for i in range(n_layers):
                self.conv_layers.add(
                    NodeUpdateModule(n_hidden, activation, dropout))


    def forward(self, g, is_data=None):
        g.ndata['W'] = self.W.data() # item embedding
        g.ndata['b'] = self.b.data() # average rating
        g.ndata['h'] = self.c.data() # user embedding for layer=0

        if self.share:
            g.ndata['W'] = _cumsum(g.ndata['W'], axis=1)
            g.ndata['b'] = _cumsum(g.ndata['b'], axis=1)
            g.ndata['h'] = _cumsum(g.ndata['h'], axis=1)

        if is_data is not None:
            g.edata['is_data'] = is_data

        for layer in self.conv_layers:
            g.update_all(gcn_msg, gcn_reduce, layer)

        if is_data is not None:
            g.edata.pop('is_data', None)

        g.apply_edges(quad_fcn)

        return g.edata.pop('pred')


def train(model, l2loss, args):
    # use optimizer
    trainer = gluon.Trainer(
        model.collect_params(),
        'adam',
        {'learning_rate': args.lr, 'wd': args.wd,})

    # initialize graph
    dur = []
    val_best = np.inf
    for epoch in range(args.n_epochs):
        t0 = time.time()
        # forward
        if args.label_rate > 0:
            is_data = mx.nd.random.uniform(0,1,len(G.edata['r'])//2, ctx=ctx) < (1-args.label_rate)
            is_data = mx.nd.concat(is_data, is_data, dim=0)
            is_label = 1-is_data
        else:
            is_data  = mx.nd.ones_like(G.edata['r'])
            is_label = mx.nd.ones_like(G.edata['r'])

        with mx.autograd.record():
            pred = model(G, is_data)
            loss = l2loss(pred, G.edata['r'], G.edata['is_train'] * is_label)

        loss.backward()
        avg_loss = (loss.sum() / (G.edata['is_train'] * is_label).sum()).asscalar()

        if args.label_rate > 0:
            pred = model(G)    

        val_loss = l2loss(pred, G.edata['r'], G.edata['is_val'])
        val_loss = (val_loss.sum() / G.edata['is_val'].sum()).asscalar()
        
        trainer.step(len(G))

        if val_loss < val_best:
            val_best = val_loss
            model.collect_params().save(args.save)

        dur.append(time.time() - t0)
        print("Epoch {:05d} | Loss {:.4f} | Val {:.4f} | lr {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, avg_loss, val_loss, trainer.learning_rate, np.mean(dur), len(G.edges) / np.mean(dur) / 1000))

    model.collect_params().load(args.save, ctx=ctx)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    G, n_classes = load_data(args.dataset, mx.gpu())
    if args.gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()

    model = CF_NADE(len(G), args.n_hidden, n_classes, args.n_layers,
                args.activation, args.dropout, args.share)
    model.collect_params().initialize(ctx=ctx)
    l2loss = gluon.loss.L2Loss()
    train(model, l2loss, args)

    pred = model(G)
    for split in ['is_train','is_val', 'is_test']:
        print('split=', split, 'final RMSE=', (2. * l2loss(pred, G.edata['r'], G.edata[split]).sum() / G.edata[split].sum()).sqrt().asscalar())

    