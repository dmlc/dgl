"""
Learning Steady-States of Iterative Algorithms over Graphs
Paper: http://proceedings.mlr.press/v80/dai18a.html

"""
import argparse
import random
import numpy as np
import time
import math
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def gcn_msg(edges):
    return {'m': mx.nd.concat(edges.src['in'], edges.src['h'], dim=1)}

def gcn_reduce(nodes):
    return {'accum': mx.nd.sum(nodes.mailbox['m'], 1) / nodes.mailbox['m'].shape[1]}

class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, alpha=0.1, **kwargs):
        super(NodeUpdate, self).__init__(**kwargs)
        self.linear1 = gluon.nn.Dense(out_feats, activation=activation)
        self.linear2 = gluon.nn.Dense(out_feats)
        self.alpha = alpha

    def forward(self, in_data, hidden_data, accum):
        tmp = mx.nd.concat(in_data, accum, dim=1)
        hidden = self.linear2(self.linear1(tmp))
        return hidden_data * (1 - self.alpha) + self.alpha * hidden

class DGLNodeUpdate(gluon.Block):
    def __init__(self, update):
        super(DGLNodeUpdate, self).__init__()
        self.update = update

    def forward(self, node):
        return {'h1': self.update(node.data['in'], node.data['h'], node.data['accum'])}

class DGLSSEUpdateHiddenInfer(gluon.Block):
    def __init__(self,
                 n_hidden,
                 activation,
                 dropout,
                 use_spmv,
                 inference,
                 **kwargs):
        super(DGLSSEUpdateHiddenInfer, self).__init__(**kwargs)
        with self.name_scope():
            self.layer = DGLNodeUpdate(NodeUpdate(n_hidden, activation))
        self.dropout = dropout
        self.use_spmv = use_spmv
        self.inference = inference

    def forward(self, g, vertices):
        if self.use_spmv:
            feat = g.ndata['in']
            g.ndata['cat'] = mx.nd.concat(feat, g.ndata['h'], dim=1)

            msg_func = fn.copy_src(src='cat', out='m')
            reduce_func = fn.sum(msg='m', out='accum')
        else:
            msg_func = gcn_msg
            reduce_func = gcn_reduce
        deg = mx.nd.expand_dims(g.in_degrees(), 1).astype(np.float32)
        if vertices is None:
            g.update_all(msg_func, reduce_func, None)
            if self.use_spmv:
                g.ndata.pop('cat')
                g.ndata['accum'] = g.ndata['accum'] / deg
            batch_size = 100000
            num_batches = int(math.ceil(g.number_of_nodes() / batch_size))
            for i in range(num_batches):
                vs = mx.nd.arange(i * batch_size, min((i + 1) * batch_size, g.number_of_nodes()), dtype=np.int64)
                g.apply_nodes(self.layer, vs, inplace=self.inference)
            g.ndata.pop('accum')
            return g.get_n_repr()['h1']
        else:
            # We don't need dropout for inference.
            if self.dropout:
                g.ndata['h'] = mx.nd.Dropout(g.ndata['h'], p=self.dropout)
            g.update_all(msg_func, reduce_func, None)
            ctx = g.ndata['accum'].context
            if self.use_spmv:
                g.ndata.pop('cat')
                deg = deg.as_in_context(ctx)
                g.ndata['accum'] = g.ndata['accum'] / deg
            g.apply_nodes(self.layer, vertices, inplace=self.inference)
            g.ndata.pop('accum')
            return mx.nd.take(g.ndata['h1'], vertices.as_in_context(ctx))

class DGLSSEUpdateHiddenTrain(gluon.Block):
    def __init__(self,
                 n_hidden,
                 activation,
                 dropout,
                 use_spmv,
                 inference,
                 **kwargs):
        super(DGLSSEUpdateHiddenTrain, self).__init__(**kwargs)
        with self.name_scope():
            self.update = DGLNodeUpdate(NodeUpdate(n_hidden, activation))
        self.dropout = dropout
        self.use_spmv = use_spmv
        self.inference = inference

    def forward(self, subg, vertices):
        assert vertices is not None
        if self.use_spmv:
            feat = subg.layers[0].data['in']
            subg.layers[0].data['cat'] = mx.nd.concat(feat, subg.layers[0].data['h'],
                                                      dim=1)

            msg_func = fn.copy_src(src='cat', out='m')
            reduce_func = fn.sum(msg='m', out='accum')
        else:
            msg_func = gcn_msg
            reduce_func = gcn_reduce
        deg = mx.nd.expand_dims(subg.layer_in_degree(1), 1).astype(np.float32)
        # We don't need dropout for inference.
        if self.dropout:
            subg.layers[0].data['h'] = mx.nd.Dropout(subg.layers[0].data['h'], p=self.dropout)
        subg.block_compute(0, msg_func, reduce_func, None)
        ctx = subg.layers[1].data['accum'].context
        if self.use_spmv:
            subg.layers[0].data.pop('cat')
            deg = deg.as_in_context(ctx)
            subg.layers[1].data['accum'] = subg.layers[1].data['accum'] / deg
        subg.apply_layer(1, self.update, inplace=self.inference)
        subg.layers[1].data.pop('accum')
        return subg.layers[1].data['h1']

class SSEPredict(gluon.Block):
    def __init__(self, update_hidden, out_feats, dropout, **kwargs):
        super(SSEPredict, self).__init__(**kwargs)
        with self.name_scope():
            self.linear1 = gluon.nn.Dense(out_feats, activation='relu')
            self.linear2 = gluon.nn.Dense(out_feats)
        self.update_hidden = update_hidden
        self.dropout = dropout

    def forward(self, g, vertices):
        hidden = self.update_hidden(g, vertices)
        if self.dropout:
            hidden = mx.nd.Dropout(hidden, p=self.dropout)
        return self.linear2(self.linear1(hidden))

def copy_to_gpu(subg, ctx):
    for i in range(subg.num_layers):
        frame = subg.layers[i].data
        for key in frame:
            subg.layers[i].data[key] = frame[key].as_in_context(ctx)

class CachedSubgraphLoader(object):
    def __init__(self, loader, shuffle):
        self._loader = loader
        self._cached = []
        self._shuffle = shuffle

    def restart(self):
        self._subgraphs = self._cached
        self._gen_subgraph = len(self._subgraphs) == 0
        random.shuffle(self._subgraphs)
        self._cached = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._subgraphs) > 0:
            subg = self._subgraphs.pop(0)
        elif self._gen_subgraph:
            subg = self._loader.__next__()
        else:
            raise StopIteration

        # We can't cache the input subgraph because it contains node frames
        # and data frames.
        subg = dgl.NodeFlow(subg._parent, subg._graph)
        self._cached.append(subg)
        return subg

def main(args, data):
    if isinstance(data.features, mx.nd.NDArray):
        features = data.features
    else:
        features = mx.nd.array(data.features)
    if isinstance(data.labels, mx.nd.NDArray):
        labels = data.labels
    else:
        labels = mx.nd.array(data.labels)
    if data.train_mask is not None:
        train_vs = mx.nd.array(np.nonzero(data.train_mask)[0], dtype='int64')
        eval_vs = mx.nd.array(np.nonzero(data.train_mask == 0)[0], dtype='int64')
    else:
        train_size = len(labels) * args.train_percent
        train_vs = mx.nd.arange(0, train_size, dtype='int64')
        eval_vs = mx.nd.arange(train_size, len(labels), dtype='int64')

    print("train size: " + str(len(train_vs)))
    print("eval size: " + str(len(eval_vs)))
    eval_labels = mx.nd.take(labels, eval_vs)
    in_feats = features.shape[1]
    n_edges = data.graph.number_of_edges()

    # create the SSE model
    try:
        graph = data.graph.get_graph()
    except AttributeError:
        graph = data.graph
    g = DGLGraph(graph, readonly=True)
    g.ndata['in'] = features
    g.ndata['h'] = mx.nd.random.normal(shape=(g.number_of_nodes(), args.n_hidden),
            ctx=mx.cpu(0))

    update_hidden_infer = DGLSSEUpdateHiddenInfer(args.n_hidden, 'relu',
                                                  args.update_dropout, args.use_spmv,
                                                  inference=True, prefix='sse')
    update_hidden_train = DGLSSEUpdateHiddenTrain(args.n_hidden, 'relu',
                                                  args.update_dropout, args.use_spmv,
                                                  inference=False, prefix='sse')

    model_train = SSEPredict(update_hidden_train, args.n_hidden, args.predict_dropout, prefix='app')
    model_infer = SSEPredict(update_hidden_infer, args.n_hidden, args.predict_dropout, prefix='app')
    model_infer.initialize(ctx=mx.cpu(0))
    if args.gpu <= 0:
        model_train.initialize(ctx=mx.cpu(0))
    else:
        train_ctxs = []
        for i in range(args.gpu):
            train_ctxs.append(mx.gpu(i))
        model_train.initialize(ctx=train_ctxs)

    # use optimizer
    num_batches = int(g.number_of_nodes() / args.batch_size)
    scheduler = mx.lr_scheduler.CosineScheduler(args.n_epochs * num_batches,
            args.lr * 10, 0, 0, args.lr/5)
    trainer = gluon.Trainer(model_train.collect_params(), 'adam', {'learning_rate': args.lr,
        'lr_scheduler': scheduler}, kvstore=mx.kv.create('device'))

    # compute vertex embedding.
    all_hidden = update_hidden_infer(g, None)
    g.ndata['h'] = all_hidden
    rets = []
    rets.append(all_hidden)

    if args.neigh_expand <= 0:
        neigh_expand = g.number_of_nodes()
    else:
        neigh_expand = args.neigh_expand
    # initialize graph
    dur = []
    sampler = dgl.contrib.sampling.NeighborSampler(g, args.batch_size, neigh_expand,
            neighbor_type='in', num_workers=args.num_parallel_subgraphs, seed_nodes=train_vs,
            shuffle=True)
    if args.cache_subgraph:
        sampler = CachedSubgraphLoader(sampler, shuffle=True)
    for epoch in range(args.n_epochs):
        t0 = time.time()
        train_loss = 0
        i = 0
        num_batches = len(train_vs) / args.batch_size
        start1 = time.time()
        for subg in sampler:
            seeds = subg.layer_parent_nid(-1)
            subg_seeds = subg.layer_nid(-1)
            subg.copy_from_parent()

            losses = []
            if args.gpu > 0:
                ctx = mx.gpu(i % args.gpu)
                copy_to_gpu(subg, ctx)

            with mx.autograd.record():
                logits = model_train(subg, subg_seeds)
                batch_labels = mx.nd.take(labels, seeds).as_in_context(logits.context)
                loss = mx.nd.softmax_cross_entropy(logits, batch_labels)
            loss.backward()
            losses.append(loss)
            i += 1
            if args.gpu <= 0:
                trainer.step(seeds.shape[0])
                train_loss += loss.asnumpy()[0]
                losses = []
            elif i % args.gpu == 0:
                trainer.step(len(seeds) * len(losses))
                for loss in losses:
                    train_loss += loss.asnumpy()[0]
                losses = []

            if i % args.num_parallel_subgraphs == 0:
                end1 = time.time()
                print("process " + str(args.num_parallel_subgraphs)
                        + " subgraphs takes " + str(end1 - start1))
                start1 = end1

        if args.cache_subgraph:
            sampler.restart()
        else:
            sampler = dgl.contrib.sampling.NeighborSampler(g, args.batch_size, neigh_expand,
                                                           neighbor_type='in',
                                                           num_workers=args.num_parallel_subgraphs,
                                                           seed_nodes=train_vs, shuffle=True)

        # test set accuracy
        logits = model_infer(g, eval_vs)
        y_bar = mx.nd.argmax(logits, axis=1)
        y = eval_labels
        accuracy = mx.nd.sum(y_bar == y) / len(y)
        accuracy = accuracy.asnumpy()[0]

        # update the inference model.
        infer_params = model_infer.collect_params()
        for key in infer_params:
            idx = trainer._param2idx[key]
            trainer._kvstore.pull(idx, out=infer_params[key].data())

        # Update node embeddings.
        all_hidden = update_hidden_infer(g, None)
        g.ndata['h'] = all_hidden
        rets.append(all_hidden)

        dur.append(time.time() - t0)
        print("Epoch {:05d} | Train Loss {:.4f} | Test Accuracy {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, train_loss, accuracy, np.mean(dur), n_edges / np.mean(dur) / 1000))

    return rets

class MXNetGraph(object):
    """A simple graph object that uses scipy matrix."""
    def __init__(self, mat):
        self._mat = mat

    def get_graph(self):
        return self._mat

    def number_of_nodes(self):
        return self._mat.shape[0]

    def number_of_edges(self):
        return mx.nd.contrib.getnnz(self._mat).asnumpy()[0]

class GraphData:
    def __init__(self, csr, num_feats):
        num_edges = mx.nd.contrib.getnnz(csr).asnumpy()[0]
        edge_ids = mx.nd.arange(0, num_edges, step=1, repeat=1, dtype=np.int64)
        csr = mx.nd.sparse.csr_matrix((edge_ids, csr.indices, csr.indptr), shape=csr.shape, dtype=np.int64)
        self.graph = MXNetGraph(csr)
        self.features = mx.nd.random.normal(shape=(csr.shape[0], num_feats))
        self.labels = mx.nd.floor(mx.nd.random.uniform(low=0, high=10, shape=(csr.shape[0])))
        self.train_mask = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--graph-file", type=str, default="",
            help="graph file")
    parser.add_argument("--num-feats", type=int, default=10,
            help="the number of features")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--batch-size", type=int, default=128,
            help="number of vertices in a batch")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--warmup", type=int, default=10,
            help="number of iterations to warm up with large learning rate")
    parser.add_argument("--update-dropout", type=float, default=0,
            help="the dropout rate for updating vertex embedding")
    parser.add_argument("--predict-dropout", type=float, default=0,
            help="the dropout rate for prediction")
    parser.add_argument("--train_percent", type=float, default=0.5,
            help="the percentage of data used for training")
    parser.add_argument("--use-spmv", action="store_true",
            help="use SpMV for faster speed.")
    parser.add_argument("--cache-subgraph", default=False, action="store_false")
    parser.add_argument("--num-parallel-subgraphs", type=int, default=1,
            help="the number of subgraphs to construct in parallel.")
    parser.add_argument("--neigh-expand", type=int, default=16,
            help="the number of neighbors to sample.")
    args = parser.parse_args()
    print("cache: " + str(args.cache_subgraph))

    # load and preprocess dataset
    if args.graph_file != '':
        csr = mx.nd.load(args.graph_file)[0]
        data = GraphData(csr, args.num_feats)
        csr = None
    else:
        data = load_data(args)
    rets1 = main(args, data)
    rets2 = main(args, data)
    for hidden1, hidden2 in zip(rets1, rets2):
        print("hidden: " + str(mx.nd.sum(mx.nd.abs(hidden1 - hidden2)).asnumpy()))
