import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class NodeUpdate(gluon.Block):
    def __init__(self, in_feats, out_feats, activation=None, concat=False):
        super(NodeUpdate, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
        self.activation = activation
        self.concat = concat

    def forward(self, node):
        h = node.data['h']
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
        nf.layers[0].data['activation'] = nf.layers[0].data['feature']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = mx.nd.Dropout(h, p=self.dropout)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.mean(msg='m', out='h'),
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
            self.layers.add(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
            # hidden layers
            for i in range(1, n_layers):
                skip_start = (i == n_layers-1)
                self.layers.add(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
            # output layer
            self.layers.add(NodeUpdate(2*n_hidden, n_classes))


    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['feature']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.mean(msg='m', out='h'),
                             layer)

        return nf.layers[-1].data.pop('activation')

def copy_from_kvstore(nf, kv, ctx, graph_name, ndata_names):
    for i in range(nf.num_layers):
        for name in ndata_names:
            full_name = graph_name + "_" + name
            global_id = nf._parent.ndata['global_id']
            data = kv.pull(name=full_name, id_tensor=global_id[nf.layer_parent_nid(i)])
            data = data.as_in_context(ctx)
            nf._node_frames[i][name] = data

def gcn_ns_train(g, kv, ctx, args, n_classes, train_nid, test_nid):
    in_feats = args.n_features
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
    # TODO I need to run distributed kvstore.
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create('local'))

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        print(epoch)
        start = time.time()
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
            #TODO copy from kvstore
            copy_from_kvstore(nf, kv, ctx, args.graph_name, ['feature', 'label'])
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = nf.layers[-1].data['label']
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        train_time = time.time() - start

        num_acc = 0.
        num_tests = 0

        start = time.time()
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid):
            copy_from_kvstore(nf, kv, ctx, args.graph_name, ['feature', 'label'])
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1)
            batch_labels = nf.layers[-1].data['label']
            num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()
            num_tests += nf.layer_size(-1)
            break
        eval_time = time.time() - start

        print("Trainer {}: Test Accuracy {:.4f}, Train Time {:.4f}, Eval time {:.4f}". format(
            args.id, num_acc/num_tests, train_time, eval_time))
