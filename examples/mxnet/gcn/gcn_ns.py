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


def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    return dgl.DGLGraph(arr, readonly=True)


def sample_subgraph(g, seed_nodes, num_hops, num_neighbors):
    induced_nodes = []
    seeds = seed_nodes
    parent_uv_edges_per_hop = []
    for _ in range(num_hops):
        for subg, aux in dgl.contrib.sampling.NeighborSampler(g, 1000000, num_neighbors,
                                                              neighbor_type='in',
                                                              seed_nodes=np.array(seeds),
                                                              return_seed_id=True):
            #seed_ids = aux['seeds']
            #print(g.in_edges(seeds, form='all'))
            subg_src, subg_dst = subg.edges()
            parent_nid = subg.parent_nid
            src = parent_nid[subg_src]
            dst = parent_nid[subg_dst]
            parent_uv_edges_per_hop.append((src.asnumpy(), dst.asnumpy()))
            #print((src, dst))
            seeds = list(np.unique(src.asnumpy()))
            induced_nodes.extend(list(parent_nid.asnumpy()))

    subgraph = g.subgraph(list(np.unique(np.array(induced_nodes))))
    #print(subgraph.parent_nid)
    #print(parent_uv_edges_per_hop)
    subg_uv_edges_per_hop = [(subgraph.map_to_subgraph_nid(src).asnumpy(),
                              subgraph.map_to_subgraph_nid(dst).asnumpy())
                             for src, dst in parent_uv_edges_per_hop]
    #print(subg_uv_edges_per_hop)
    return subgraph, subg_uv_edges_per_hop


def test_sample():
    g = generate_rand_graph(100)
    num_hops = 3
    seeds = [10, 20]
    num_neighbors = 2
    subgraph, subg_uv_edges_per_hop = sample_subgraph(g, seeds, num_hops, num_neighbors)


class GCNLayer(gluon.Block):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        with self.name_scope():
            stdv = 1. / math.sqrt(out_feats)
            self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                    init=mx.init.Uniform(stdv))
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                    init=mx.init.Uniform(stdv))
            else:
                self.bias = None
        self.activation = activation
        self.dropout = dropout

    def forward(self, h, subg, subg_edges):
        if self.dropout:
            h = mx.nd.Dropout(h, p=self.dropout)
        h = mx.nd.dot(h, self.weight.data(h.context))
        # normalization by square root of src degree
        h = h * subg.ndata['norm']
        subg.ndata['h'] = h
        subg.send_and_recv(subg_edges, fn.copy_src(src='h', out='m'),
                           fn.sum(msg='m', out='h'))
        h = subg.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * subg.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias.data(h.context)
        if self.activation:
            h = self.activation(h)
        return h


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
            h = layer(h, subg, subg_edges_per_hop[self.n_layers-i])

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
    g = DGLGraph(data.graph, readonly=True)
    # normalization
    degs = g.in_degrees().astype('float32')
    norm = mx.nd.power(degs, -0.5)
    if cuda:
        norm = norm.as_in_context(ctx)
    g.ndata['norm'] = mx.nd.expand_dims(norm, 1)
    g.ndata['in'] = features

    num_data = len(train_mask.asnumpy())
    full_idx = np.arange(0, num_data)
    train_idx = full_idx[train_mask.asnumpy() == 1]

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

        seed_nodes = list(train_idx)
        num_hops = args.n_layers + 1
        num_neighbors = 3
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
