import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
import dgl.function as fn
from dgl.data import register_data_args, load_data
import scipy as sp
from dgl import utils

class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, dropout=0, **kwargs):
        super(NodeUpdate, self).__init__(**kwargs)
        self.linear = gluon.nn.Dense(out_feats, activation=activation)
        self.dropout = dropout

    def forward(self, node):
        accum = node.data['h']
        if self.dropout:
            accum = mx.nd.Dropout(accum, p=self.dropout)
        accum = self.linear(accum)
        #accum = mx.nd.concat(self.linear(accum), node.data['h'], dim=1)
        return {'h': accum}

class GCNLayer(gluon.Block):
    def __init__(self,
                 ind,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.ind = ind
        self.in_feats = in_feats
        self.node_update = NodeUpdate(out_feats, activation, dropout)

    def forward(self, subg, h):
        subg.layers[self.ind].data['h'] = h
        subg.flow_compute(self.ind, fn.copy_src(src='h', out='m')
                          fn.sum(msg='m', out='h'),
                          self.node_update)
        return subg.layers[self.ind + 1].data['h']


class GCN(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.add(GCNLayer(n_hidden, n_classes, None, dropout))


    def forward(self, subg, subg_edges_per_hop):
        h = subg.ndata['in']
        for i, layer in enumerate(self.layers):
            h = layer(subg, h)
        return h


def evaluate(model, g, num_hops, labels, mask):
    pred = model(g, [None for i in range(num_hops)]).argmax(axis=1)
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
    n_nodes = data.graph.number_of_nodes()
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
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

    num_neighbors = args.num_neighbors
    # create GCN model
    g = DGLGraph(data.graph, readonly=True)
    # normalization
    degs = g.in_degrees().astype('float32')
    degs[degs > num_neighbors] = num_neighbors
    norm = mx.nd.power(degs, -0.5)
    if cuda:
        norm = norm.as_in_context(ctx)
    g.ndata['norm'] = mx.nd.expand_dims(norm, 1)
    g.ndata['in'] = features

    num_data = len(train_mask.asnumpy())
    full_idx = np.arange(0, num_data)
    train_idx = full_idx[train_mask.asnumpy() == 1]

    seed_nodes = list(train_idx)
    num_hops = args.n_layers + 1

    model = GCN(in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                mx.nd.relu,
                args.dropout,
                args.normalization)
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

        subg, subg_edges_per_hop = sample_subgraph(g, seed_nodes, num_hops, num_neighbors)
        subg_train_mask = subg.map_to_subgraph_nid(train_idx)
        subg.copy_from_parent()

        # forward
        smask = np.zeros((len(subg.parent_nid),))
        smask[subg_train_mask.asnumpy()] = 1

        with mx.autograd.record():
            pred = model(subg, subg_edges_per_hop)
            loss = loss_fcn(pred, labels[subg.parent_nid], mx.nd.expand_dims(mx.nd.array(smask), 1))
            loss = loss.sum() / n_train_samples

        print(loss.asnumpy())
        loss.backward()
        trainer.step(batch_size=1)

        if epoch >= 3:
            dur.append(time.time() - t0)
            acc = evaluate(model, g, num_hops, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(
                epoch, np.mean(dur), loss.asscalar(), acc, n_edges / np.mean(dur) / 1000))

    acc = evaluate(model, g, num_hops, labels, test_mask)
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
    parser.add_argument("--num-neighbors", type=int, default=5,
            help="number of neighbors to be sampled")
    parser.add_argument("--normalization",
            choices=['sym','left'], default=None,
            help="graph normalization types (default=None)")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)
