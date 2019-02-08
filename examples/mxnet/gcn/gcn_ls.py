import argparse, time
import numpy as np
import numpy.random as npr
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.contrib.sampling import LayerSampler
from dgl.data import register_data_args, load_data
import dgl.function as fn
from dgl.subgraph import DGLSubGraph

def check(g, sub_g, n_layers, layer_size, train_nid):
    n = sub_g.number_of_nodes()
    m = sub_g.number_of_edges()
#   print('# nodes: %d, # edges: %d, # seeds: %d' % (n, m, len(train_nid)))
    nid = np.arange(n)
    src, dst = sub_g.edges()
    src = src.asnumpy()
    dst = dst.asnumpy()
    lid = sub_g.layer_ids.asnumpy()
#   print('np.unique(lid)', np.unique(lid))
    for i in range(n_layers + 1):
        nmask = lid == i
        src_mask = np.isin(src, nid[nmask])
        dst_mask = np.isin(dst, nid[nmask])
        nn = np.sum(nmask)
        mm_src = np.sum(src_mask)
        mm_dst = np.sum(dst_mask)
        assert nn <= layer_size
        if i == 0:
            assert mm_dst == 0
        if i == n_layers + 1:
            assert mm_src == 0
#       print('[layer %d]# nodes: %d, # src: %d, # dst: %d' % (i, nn, mm_src, mm_dst))
    src_lid = lid[src]
    dst_lid = lid[dst]
    assert np.all(dst_lid - src_lid == 1)

class GCNLayer(gluon.Block):
    def __init__(self, in_feats, out_feats, activation, dropout=0):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        with self.name_scope():
            self.dense = mx.gluon.nn.Dense(out_feats, activation)

    def forward(self, sub_g, src, dst):
        if self.dropout > 0:
            dropout = lambda nodes: {'h' : nd.Dropout(nodes.data['h'], p=self.dropout)}
            sub_g.apply_nodes(dropout)
        # normalize = lambda nodes : {'h' : nodes.data['h'] * nodes.data['normalizer']}
        # sub_g.apply_nodes(normalize)
        if src is None:
            sub_g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        else:
            sub_g.send_and_recv((src, dst),
                                fn.copy_src(src='h', out='m'),
                                fn.sum(msg='m', out='h'))
        # sub_g.apply_nodes(normalize)
        sub_g.apply_nodes(lambda nodes : {'h' : self.dense(nodes.data['h'])}) 

class GCN(gluon.Block):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout, normalization):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.dense = mx.gluon.nn.Dense(n_classes)

    def forward(self, sub_g):
        sub_g.ndata['h'] = sub_g.ndata['x']
        if isinstance(sub_g, DGLSubGraph):
            n = sub_g.number_of_nodes()
            nid = np.arange(n)
            src, dst = sub_g.edges()
            src = src.asnumpy()
            dst = dst.asnumpy()
            layer_ids = sub_g.layer_ids.asnumpy()
            sample_prob = sub_g.sample_prob.asnumpy()
            for i, layer in enumerate(self.layers):
                nmask = layer_ids == i
                emask = np.isin(src, nid[nmask])
                src = src[emask]
                dst = dst[emask]
                h = sub_g.ndata['h']
                p = np.expand_dims(np.where(nmask, sample_prob, np.ones(n)), axis=1)
                sub_g.ndata['h'] = h
#               sub_g.ndata['h'] = h * nd.array(p).as_in_context(h.context)
                layer(sub_g, src, dst)
        else:
            for layer in self.layers:
                layer(sub_g, None, None)
        return self.dense(sub_g.pop_n_repr('h'))

def evaluate(model, g, val=False):
    y = g.ndata['y']
    y_bar = nd.argmax(model(g), axis=1)
    mask = g.ndata['val_mask'] if val else g.ndata['test_mask']
    accuracy = nd.sum(mask * (y == y_bar)) / nd.sum(mask)
    return accuracy.asscalar()

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    if args.self_loop:
        data.graph.add_edges_from([(i, i) for i in range(len(data.graph))])
    n_nodes = data.graph.number_of_nodes()
    n_edges = data.graph.number_of_edges()
    features = nd.array(data.features)
    in_feats = features.shape[1]
    labels = nd.array(data.labels)
    n_classes = data.num_labels
    train_mask = nd.array(data.train_mask)
    val_mask = nd.array(data.val_mask)
    test_mask = nd.array(data.test_mask)
    print("""-----Data statistics-----
      # Nodes %d
      # Edges %d
      # Features %d
      # Classes %d
      # Train samples %d
      # Val samples %d
      # Test samples %d""" % (n_nodes, n_edges, in_feats, n_classes,
                              train_mask.sum().asscalar(),
                              val_mask.sum().asscalar(),
                              test_mask.sum().asscalar()))

    train_nid = np.arange(n_nodes)[data.train_mask.astype(bool)].tolist()

    ctx = mx.cpu(0) if args.gpu < 0 else mx.gpu(args.gpu)
    features = features.as_in_context(ctx)
    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    val_mask = val_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    g = DGLGraph(data.graph, readonly=True)
    g.ndata['x'] = features
    g.ndata['y'] = labels
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    deg = g.in_degrees().astype('float32').as_in_context(ctx)
    g.ndata['normalizer'] = nd.expand_dims(nd.power(deg, -0.5), 1)
    assert not g.is_multigraph

    model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers,
                'relu', args.dropout, args.normalization)
    model.initialize(ctx=ctx)
    print(model.collect_params())

    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': args.lr, 'wd': args.weight_decay})

    def sampler():
        seed_nodes = npr.choice(train_nid, 32, replace=False)
        for x in LayerSampler(g, 1000000, args.layer_size, args.n_layers,
                              neighbor_type='in', seed_nodes=seed_nodes,
                              return_prob=True):
            yield x

    dur = []
    for epoch in range(args.n_epochs):
        t0 = time.time()

        sub_g, _ = next(sampler())
        sub_g.copy_from_parent()
        # check(g, sub_g, args.n_layers, args.layer_size, train_nid)

        with mx.autograd.record():
            y = sub_g.ndata['y']
            y_bar = model(sub_g)
            mask = sub_g.layer_ids.as_in_context(y) == args.n_layers
            loss = nd.sum(mask * gluon.loss.SoftmaxCELoss()(y_bar, y)) / nd.sum(mask)

        loss.backward()
        trainer.step(batch_size=1)

        dur.append(time.time() - t0)
        acc = evaluate(model, g, val=True)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(
              epoch, np.mean(dur), loss.asscalar(), acc, n_edges / np.mean(dur) / 1000))

    acc = evaluate(model, g)
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
    parser.add_argument("--layer-size", type=int, default=128,
            help="number of neighbors to be sampled")
    parser.add_argument("--normalization",
            choices=['sym'], default=None,
            help="graph normalization types (default=None)")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)
