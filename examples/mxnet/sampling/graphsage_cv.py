import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

class GraphSAGELayer(gluon.Block):
    def __init__(self,
                 in_feats,
                 hidden,
                 out_feats,
                 dropout,
                 last=False,
                 **kwargs):
        super(GraphSAGELayer, self).__init__(**kwargs)
        self.last = last
        self.dropout = dropout
        with self.name_scope():
            self.dense1 = gluon.nn.Dense(hidden, in_units=in_feats)
            self.layer_norm1 = gluon.nn.LayerNorm(in_channels=hidden)
            self.dense2 = gluon.nn.Dense(out_feats, in_units=hidden)
            if not self.last:
                self.layer_norm2 = gluon.nn.LayerNorm(in_channels=out_feats)

    def forward(self, h):
        h = self.dense1(h)
        h = self.layer_norm1(h)
        h = mx.nd.relu(h)
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = self.dense2(h)
        if not self.last:
            h = self.layer_norm2(h)
            h = mx.nd.relu(h)
        return h


class NodeUpdate(gluon.Block):
    def __init__(self, layer_id, in_feats, out_feats, hidden, dropout,
                 test=False, last=False):
        super(NodeUpdate, self).__init__()
        self.layer_id = layer_id
        self.dropout = dropout
        self.test = test
        self.last = last
        with self.name_scope():
            self.layer = GraphSAGELayer(in_feats, hidden, out_feats, dropout, last)

    def forward(self, node):
        h = node.data['h']
        norm = node.data['norm']
        # activation from previous layer of myself
        self_h = node.data['self_h']

        if self.test:
            h = (h - self_h) * norm
            # graphsage
            h = mx.nd.concat(h, self_h)
        else:
            agg_history_str = 'agg_h_{}'.format(self.layer_id-1)
            agg_history = node.data[agg_history_str]
            # normalization constant
            subg_norm = node.data['subg_norm']
            # delta_h (h - history) from previous layer of myself
            self_delta_h = node.data['self_delta_h']
            # control variate
            h = (h - self_delta_h) * subg_norm + agg_history * norm
            # graphsage
            h = mx.nd.concat(h, self_h)
            if self.dropout:
                h = mx.nd.Dropout(h, p=self.dropout)

        h = self.layer(h)

        return {'activation': h}



class GraphSAGETrain(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout,
                 **kwargs):
        super(GraphSAGETrain, self).__init__(**kwargs)
        self.dropout = dropout
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            self.input_layer = GraphSAGELayer(2*in_feats, n_hidden, n_hidden, dropout)
            # hidden layers
            for i in range(1, n_layers):
                self.layers.add(NodeUpdate(i, 2*n_hidden, n_hidden, n_hidden, dropout))
            # output layer
            self.layers.add(NodeUpdate(n_layers, 2*n_hidden, n_classes, n_hidden, dropout, last=True))

    def forward(self, nf):
        h = nf.layers[0].data['preprocess']
        features = nf.layers[0].data['features']
        h = mx.nd.concat(h, features)
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)

        h = self.input_layer(h)

        for i, layer in enumerate(self.layers):
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid)
            self_h = h[layer_nid]
            # activation from previous layer of myself, used in graphSAGE
            nf.layers[i+1].data['self_h'] = self_h

            new_history = h.copy().detach()
            history_str = 'h_{}'.format(i)
            history = nf.layers[i].data[history_str]
            # delta_h used in control variate
            delta_h = h - history
            # delta_h from previous layer of the nodes in (i+1)-th layer, used in control variate
            nf.layers[i+1].data['self_delta_h'] = delta_h[layer_nid]

            nf.layers[i].data['h'] = delta_h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
            h = nf.layers[i+1].data.pop('activation')
            # update history
            if i < nf.num_layers-1:
                nf.layers[i].data[history_str] = new_history

        return h


class GraphSAGEInfer(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 **kwargs):
        super(GraphSAGEInfer, self).__init__(**kwargs)
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            self.input_layer = GraphSAGELayer(2*in_feats, n_hidden, n_hidden, 0)
            # hidden layers
            for i in range(1, n_layers):
                self.layers.add(NodeUpdate(i, 2*n_hidden, n_hidden, n_hidden, 0, True))
            # output layer
            self.layers.add(NodeUpdate(n_layers, 2*n_hidden, n_classes, n_hidden, 0, True, last=True))


    def forward(self, nf):
        h = nf.layers[0].data['preprocess']
        features = nf.layers[0].data['features']
        h = mx.nd.concat(h, features)
        h = self.input_layer(h)

        for i, layer in enumerate(self.layers):
            nf.layers[i].data['h'] = h
            parent_nid = dgl.utils.toindex(nf.layer_parent_nid(i+1))
            layer_nid = nf.map_from_parent_nid(i, parent_nid)
            # activation from previous layer of the nodes in (i+1)-th layer, used in graphSAGE
            self_h = h[layer_nid]
            nf.layers[i+1].data['self_h'] = self_h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)
            h = nf.layers[i+1].data.pop('activation')

        return h


def main(args):
    # load and preprocess dataset
    data = load_data(args)

    if args.gpu >= 0:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.cpu()

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = mx.nd.array(np.nonzero(data.train_mask)[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(data.test_mask)[0]).astype(np.int64)

    num_neighbors = args.num_neighbors
    n_layers = args.n_layers

    features = mx.nd.array(data.features).as_in_context(ctx)
    labels = mx.nd.array(data.labels).as_in_context(ctx)
    train_mask = mx.nd.array(data.train_mask).as_in_context(ctx)
    val_mask = mx.nd.array(data.val_mask).as_in_context(ctx)
    test_mask = mx.nd.array(data.test_mask).as_in_context(ctx)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    n_train_samples = train_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))

    g = DGLGraph(data.graph, readonly=True)

    g.ndata['features'] = features

    norm = mx.nd.expand_dims(1./g.in_degrees().astype('float32'), 1)
    g.ndata['norm'] = norm.as_in_context(ctx)

    degs = g.in_degrees().astype('float32').asnumpy()
    degs[degs > num_neighbors] = num_neighbors
    g.ndata['subg_norm'] = mx.nd.expand_dims(mx.nd.array(1./degs, ctx=ctx), 1)


    g.update_all(fn.copy_src(src='features', out='m'),
                 fn.sum(msg='m', out='preprocess'),
                 lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})

    for i in range(n_layers):
        g.ndata['h_{}'.format(i)] = mx.nd.zeros((features.shape[0], args.n_hidden), ctx=ctx)

    model = GraphSAGETrain(in_feats,
                           args.n_hidden,
                           n_classes,
                           n_layers,
                           args.dropout,
                           prefix='GraphSAGE')

    model.initialize(ctx=ctx)

    loss_fcn = gluon.loss.SoftmaxCELoss()

    infer_model = GraphSAGEInfer(in_feats,
                                 args.n_hidden,
                                 n_classes,
                                 n_layers,
                                 prefix='GraphSAGE')

    infer_model.initialize(ctx=ctx)

    # use optimizer
    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay},
                            kvstore=mx.kv.create('local'))

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_hops=n_layers,
                                                       add_self_loop=True,
                                                       seed_nodes=train_nid):
            for i in range(n_layers):
                agg_history_str = 'agg_h_{}'.format(i)
                g.pull(nf.layer_parent_nid(i+1), fn.copy_src(src='h_{}'.format(i), out='m'),
                       fn.sum(msg='m', out=agg_history_str))

            node_embed_names = [['preprocess', 'features', 'h_0']]
            for i in range(1, n_layers):
                node_embed_names.append(['h_{}'.format(i), 'agg_h_{}'.format(i-1), 'subg_norm', 'norm'])
            node_embed_names.append(['agg_h_{}'.format(n_layers-1), 'subg_norm', 'norm'])

            nf.copy_from_parent(node_embed_names=node_embed_names)
            # forward
            with mx.autograd.record():
                pred = model(nf)
                batch_nids = nf.layer_parent_nid(-1).as_in_context(ctx)
                batch_labels = labels[batch_nids]
                loss = loss_fcn(pred, batch_labels)
                loss = loss.sum() / len(batch_nids)

            loss.backward()
            trainer.step(batch_size=1)

            node_embed_names = [['h_{}'.format(i)] for i in range(n_layers)]
            node_embed_names.append([])

            nf.copy_to_parent(node_embed_names=node_embed_names)

        infer_params = infer_model.collect_params()

        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        num_acc = 0.

        for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_hops=n_layers,
                                                       seed_nodes=test_nid,
                                                       add_self_loop=True):
            node_embed_names = [['preprocess', 'features']]
            for i in range(n_layers):
                node_embed_names.append(['norm', 'subg_norm'])
            nf.copy_from_parent(node_embed_names=node_embed_names)

            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1).as_in_context(ctx)
            batch_labels = labels[batch_nids]
            num_acc += (pred.argmax(axis=1) == batch_labels).sum().asscalar()

        print("Test Accuracy {:.4f}". format(num_acc/n_test_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE with Control Variate')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden GraphSAGE units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden GraphSAGE layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)

