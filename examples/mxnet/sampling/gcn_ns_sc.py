from multiprocessing import Process
import argparse, time, math
import numpy as np
from scipy import sparse as spsp
import os
os.environ['OMP_NUM_THREADS'] = '16'
import mxnet as mx
from mxnet import gluon
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

class NodeUpdate(gluon.Block):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']
        h = h * node.data['norm']
        h = self.dense(h)
        # skip connection
        if self.concat:
            h = mx.nd.concat(h, self.activation(h))
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}


class GCNSampling(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.dropout = dropout
        self.n_layers = n_layers
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            skip_start = (0 == n_layers-1)
            self.layers.add(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
            # hidden layers
            for i in range(1, n_layers):
                skip_start = (i == n_layers-1)
                self.layers.add(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
            # output layer
            self.layers.add(NodeUpdate(2*n_hidden, n_classes))


    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = mx.nd.Dropout(h, p=self.dropout)
            nf.layers[i].data['h'] = h
            degs = nf.layer_in_degree(i + 1).astype('float32').as_in_context(h.context)
            nf.layers[i + 1].data['norm'] = mx.nd.expand_dims(1./degs, 1)
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h


class GCNInfer(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 **kwargs):
        super(GCNInfer, self).__init__(**kwargs)
        self.n_layers = n_layers
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            skip_start = (0 == n_layers-1)
            self.layers.add(NodeUpdate(in_feats, n_hidden, activation, test=True, concat=skip_start))
            # hidden layers
            for i in range(1, n_layers):
                skip_start = (i == n_layers-1)
                self.layers.add(NodeUpdate(n_hidden, n_hidden, activation, test=True, concat=skip_start))
            # output layer
            self.layers.add(NodeUpdate(2*n_hidden, n_classes, test=True))


    def forward(self, g):
        g.ndata['activation'] = g.ndata['features']

        for i, layer in enumerate(self.layers):
            h = g.ndata.pop('activation')
            g.ndata['h'] = h
            g.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'),
                         layer)

        return g.ndata.pop('activation')

def worker_func(args, g, features, labels, train_mask, val_mask, test_mask,
                in_feats, n_classes, train_nid, test_nid, n_test_samples, ctx, nworkers):
    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        mx.nd.relu,
                        args.dropout,
                        prefix='GCN')

    model.initialize(ctx=ctx)
    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           args.n_layers,
                           mx.nd.relu,
                           prefix='GCN')

    infer_model.initialize(ctx=ctx)

    # use optimizer
    print(model.collect_params())
    kv_type = 'local' if nworkers == 1 else 'dist_sync'
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create(kv_type))

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        train_start = start = time.time()
        train_sample_time = 0
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
            train_sample_time += time.time() - start
            nf.copy_from_parent(ctx=ctx)
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1).astype('int64')
                batch_labels = labels[batch_nids].as_in_context(ctx)
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)
            start = time.time()
        print("train time: %.3f, sample: %.3f" % (time.time() - train_start, train_sample_time))

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.

        infer_start = start = time.time()
        pred = infer_model(g)
        num_acc = (pred.argmax(axis=1) == labels)[test_nid].sum().asscalar()
        #for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
        #                                               g.number_of_nodes(),
        #                                               neighbor_type='in',
        #                                               num_workers=32,
        #                                               num_hops=args.n_layers+1,
        #                                               seed_nodes=test_nid):
        #    infer_sample_time += time.time() - start
        #    nf.copy_from_parent()
        #    pred = infer_model(nf)
        #    batch_nids = nf.layer_parent_nid(-1).astype('int64').as_in_context(ctx)
        #    batch_labels = labels[batch_nids]
        #    num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()
        #    start = time.time()
        print("infer time: %.3f" % (time.time() - infer_start))

        print("Test Accuracy {:.4f}". format(num_acc/n_test_samples))


def main(args):
    g = dgl.contrib.graph_store.create_graph_store_client(args.graph_name, "shared_mem")
    features = g.ndata['features']
    labels = g.ndata['labels']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    if args.gpu >= 0:
        runtime_ctx = mx.gpu(args.gpu)
    else:
        runtime_ctx = mx.cpu()

    train_nid = mx.nd.array(np.nonzero(train_mask.asnumpy())[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(test_mask.asnumpy())[0]).astype(np.int64)

    in_feats = features.shape[1]
    n_classes = len(np.unique(labels.asnumpy()))
    n_train_samples = train_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()

    num_neighbors = args.num_neighbors
    degs = g.in_degrees().astype('float32')
    norm = mx.nd.expand_dims(1./degs, 1)
    g.ndata['norm'] = norm

    worker_func(args, g, features, labels, train_mask, val_mask, test_mask,
                in_feats, n_classes, train_nid, test_nid, n_test_samples,
                runtime_ctx, args.nworkers)
    print("parent ends")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--graph-name", type=str, default="",
            help="graph name")
    parser.add_argument("--num-feats", type=int, default=100,
            help="the number of features")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="the gpu index")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--nworkers", type=int, default=1,
            help="number of workers")
    args = parser.parse_args()

    print(args)

    main(args)

