import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


class NodeUpdate(gluon.Block):
    def __init__(self, layer_id, in_feats, out_feats, dropout, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.layer_id = layer_id
        self.dropout = dropout
        self.test = test
        self.concat = concat
        with self.name_scope():
            self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
            self.activation = activation

    def forward(self, node):
        h = node.data['h']
        norm = node.data['norm']
        if self.test:
            h = h * norm
        else:
            agg_history_str = 'agg_h_{}'.format(self.layer_id-1)
            agg_history = node.data[agg_history_str]
            subg_norm = node.data['subg_norm']
            # control variate
            h = h * subg_norm + agg_history * norm
            if self.dropout:
                h = mx.nd.Dropout(h, p=self.dropout)
        h = self.dense(h)
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
            self.dense = gluon.nn.Dense(n_hidden, in_units=in_feats)
            self.activation = activation
            # hidden layers
            for i in range(1, n_layers):
                skip_start = (i == self.n_layers-1)
                self.layers.add(NodeUpdate(i, n_hidden, n_hidden, dropout, activation, concat=skip_start))
            # output layer
            self.layers.add(NodeUpdate(n_layers, 2*n_hidden, n_classes, dropout))

    def forward(self, nf):
        h = nf.layers[0].data['preprocess']
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = self.dense(h)

        skip_start = (0 == self.n_layers-1)
        if skip_start:
            h = mx.nd.concat(h, self.activation(h))
        else:
            h = self.activation(h)

        for i, layer in enumerate(self.layers):
            new_history = h.copy().detach()
            history_str = 'h_{}'.format(i)
            history = nf.layers[i].data[history_str]
            h = h - history

            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
            h = nf.layers[i+1].data.pop('activation')
            # update history
            if i < nf.num_layers-1:
                nf.layers[i].data[history_str] = new_history

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
            self.dense = gluon.nn.Dense(n_hidden, in_units=in_feats)
            self.activation = activation
            # hidden layers
            for i in range(1, n_layers):
                skip_start = (i == self.n_layers-1)
                self.layers.add(NodeUpdate(i, n_hidden, n_hidden, 0, activation, True, concat=skip_start))
            # output layer
            self.layers.add(NodeUpdate(n_layers, 2*n_hidden, n_classes, 0, None, True))


    def forward(self, nf):
        h = nf.layers[0].data['preprocess']
        h = self.dense(h)

        skip_start = (0 == self.n_layers-1)
        if skip_start:
            h = mx.nd.concat(h, self.activation(h))
        else:
            h = self.activation(h)

        for i, layer in enumerate(self.layers):
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
            h = nf.layers[i+1].data.pop('activation')

        return h


def gcn_cv_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples, distributed):
    n0_feats = g.nodes[0].data['features']
    num_nodes = g.number_of_nodes()
    in_feats = n0_feats.shape[1]
    g_ctx = n0_feats.context

    norm = mx.nd.expand_dims(1./g.in_degrees().astype('float32'), 1)
    g.set_n_repr({'norm': norm.as_in_context(g_ctx)})
    degs = g.in_degrees().astype('float32').asnumpy()
    degs[degs > args.num_neighbors] = args.num_neighbors
    g.set_n_repr({'subg_norm': mx.nd.expand_dims(mx.nd.array(1./degs, ctx=g_ctx), 1)})
    n_layers = args.n_layers

    g.update_all(fn.copy_src(src='features', out='m'),
                 fn.sum(msg='m', out='preprocess'),
                 lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})
    for i in range(n_layers - 1):
        g.init_ndata('h_{}'.format(i), (num_nodes, args.n_hidden), 'float32')
        g.init_ndata('agg_h_{}'.format(i), (num_nodes, args.n_hidden), 'float32')
    g.init_ndata('h_{}'.format(n_layers-1), (num_nodes, 2*args.n_hidden), 'float32')
    g.init_ndata('agg_h_{}'.format(n_layers-1), (num_nodes, 2*args.n_hidden), 'float32')

    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        n_layers,
                        mx.nd.relu,
                        args.dropout,
                        prefix='GCN')

    model.initialize(ctx=ctx)

    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GCNInfer(in_feats,
                           args.n_hidden,
                           n_classes,
                           n_layers,
                           mx.nd.relu,
                           prefix='GCN')

    infer_model.initialize(ctx=ctx)

    # use optimizer
    print(model.collect_params())
    kv_type = 'dist_sync' if distributed else 'local'
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create(kv_type))

    # initialize graph
    dur = []
    adj = g.adjacency_matrix().as_in_context(g_ctx)
    for epoch in range(args.n_epochs):
        start = time.time()
        if distributed:
            msg_head = "Worker {:d}, epoch {:d}".format(g.worker_id, epoch)
        else:
            msg_head = "epoch {:d}".format(epoch)
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       num_workers=32,
                                                       shuffle=True,
                                                       num_hops=n_layers,
                                                       seed_nodes=train_nid):
            for i in range(n_layers):
                agg_history_str = 'agg_h_{}'.format(i)
                dests = nf.layer_parent_nid(i+1).as_in_context(g_ctx)
                # TODO we could use DGLGraph.pull to implement this, but the current
                # implementation of pull is very slow. Let's manually do it for now.
                agg = mx.nd.dot(mx.nd.take(adj, dests), g.nodes[:].data['h_{}'.format(i)])
                g.set_n_repr({agg_history_str: agg}, dests)

            node_embed_names = [['preprocess', 'h_0']]
            for i in range(1, n_layers):
                node_embed_names.append(['h_{}'.format(i), 'agg_h_{}'.format(i-1), 'subg_norm', 'norm'])
            node_embed_names.append(['agg_h_{}'.format(n_layers-1), 'subg_norm', 'norm'])

            nf.copy_from_parent(node_embed_names=node_embed_names, ctx=ctx)
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)

            node_embed_names = [['h_{}'.format(i)] for i in range(n_layers)]
            node_embed_names.append([])

            nf.copy_to_parent(node_embed_names=node_embed_names)
        mx.nd.waitall()
        print(msg_head + ': training takes ' + str(time.time() - start))

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.
        num_tests = 0

        if not distributed or g.worker_id == 0:
            for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                           g.number_of_nodes(),
                                                           neighbor_type='in',
                                                           num_hops=n_layers,
                                                           seed_nodes=test_nid):
                node_embed_names = [['preprocess']]
                for i in range(n_layers):
                    node_embed_names.append(['norm'])

                nf.copy_from_parent(node_embed_names=node_embed_names, ctx=ctx)
                pred = infer_model(nf)
                batch_nids = nf.layer_parent_nid(-1)
                batch_labels = g.nodes[batch_nids].data['labels'].as_in_context(ctx)
                num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()
                num_tests += nf.layer_size(-1)
                if distributed:
                    g._sync_barrier()
                print("Test Accuracy {:.4f}". format(num_acc/num_tests))
                break
        elif distributed:
                g._sync_barrier()
