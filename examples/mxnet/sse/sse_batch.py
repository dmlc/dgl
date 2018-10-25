"""
Learning Steady-States of Iterative Algorithms over Graphs
Paper: http://proceedings.mlr.press/v80/dai18a.html

"""
import argparse
import numpy as np
import time
import mxnet as mx
from mxnet import gluon
import dgl
import dgl.function as fn
from dgl import DGLGraph, utils
from dgl.data import register_data_args, load_data

def gcn_msg(src, edge):
    # TODO should we use concat?
    return {'m': mx.nd.concat(src['in'], src['h'], dim=1)}

def gcn_reduce(node, msgs):
    return {'accum': mx.nd.sum(msgs['m'], 1)}

class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, alpha=0.9):
        super(NodeUpdate, self).__init__()
        self.linear1 = gluon.nn.Dense(out_feats, activation=activation)
        # TODO what is the dimension here?
        self.linear2 = gluon.nn.Dense(out_feats)
        self.alpha = alpha

    def forward(self, node):
        tmp = mx.nd.concat(node['in'], node['accum'], dim=1)
        hidden = self.linear2(self.linear1(tmp))
        return {'h': node['h'] * (1 - self.alpha) + self.alpha * hidden}

class SSEUpdateHidden(gluon.Block):
    def __init__(self,
                 n_hidden,
                 activation,
                 dropout,
                 use_spmv):
        super(SSEUpdateHidden, self).__init__()
        self.layer = NodeUpdate(n_hidden, activation)
        self.dropout = dropout
        self.use_spmv = use_spmv

    def forward(self, g, vertices):
        if self.use_spmv:
            feat = g.get_n_repr()['in']
            h = g.get_n_repr()['h']
            g.set_n_repr({'cat': mx.nd.concat(feat, h, dim=1)})

            msg_func = fn.copy_src(src='cat', out='tmp')
            reduce_func = fn.sum(msg='tmp', out='accum')
        else:
            msg_func = gcn_msg
            reduce_func = gcn_reduce
        if vertices is None:
            g.update_all(msg_func, reduce_func, self.layer)
            ret = g.get_n_repr()['h']
        else:
            # We don't need dropout for inference.
            if self.dropout:
                # TODO here we apply dropout on all vertex representation.
                val = mx.nd.Dropout(g.get_n_repr()['h'], p=self.dropout)
                g.set_n_repr({'h': val})
            g.pull(vertices, msg_func, reduce_func, self.layer)
            ctx = g.get_n_repr()['h'].context
            ret = mx.nd.take(g.get_n_repr()['h'], vertices.tousertensor().as_in_context(ctx))
        return ret

class SSEPredict(gluon.Block):
    def __init__(self, update_hidden, out_feats, dropout):
        super(SSEPredict, self).__init__()
        self.linear1 = gluon.nn.Dense(out_feats, activation='relu')
        self.linear2 = gluon.nn.Dense(out_feats)
        self.update_hidden = update_hidden
        self.dropout = dropout

    def forward(self, g, vertices):
        hidden = self.update_hidden(g, vertices)
        if self.dropout:
            hidden = mx.nd.Dropout(hidden, p=self.dropout)
        return self.linear2(self.linear1(hidden))

def subgraph_gen(g, seed_vertices):
    vertices = []
    for seed in seed_vertices:
        src, _, _ = g.in_edges(seed)
        vs = np.concatenate((src.asnumpy(), seed.asnumpy()), axis=0)
        vs = mx.nd.array(np.unique(vs), dtype=np.int64)
        vertices.append(vs)
    subgs = g.subgraphs(vertices)
    nids = []
    for i, subg in enumerate(subgs):
        subg.copy_from_parent()
        nids.append(subg.map_to_subgraph_nid(utils.toindex(seed_vertices[i])))
    return subgs, nids

def main(args, data):
    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    train_size = len(labels) * args.train_percent
    train_vs = np.arange(train_size, dtype='int64')
    eval_vs = np.arange(train_size, len(labels), dtype='int64')
    print("train size: " + str(len(train_vs)))
    print("eval size: " + str(len(eval_vs)))
    train_labels = mx.nd.array(data.labels[train_vs])
    eval_labels = mx.nd.array(data.labels[eval_vs])
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu <= 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        features = features.as_in_context(mx.gpu(0))
        train_labels = train_labels.as_in_context(mx.gpu(0))
        eval_labels = eval_labels.as_in_context(mx.gpu(0))
        ctx = mx.gpu(0)

    # create the SSE model
    try:
        graph = data.graph.get_graph()
    except AttributeError:
        graph = data.graph
    g = DGLGraph(graph, readonly=True)
    g.set_n_repr({'in': features, 'h': mx.nd.random.normal(shape=(g.number_of_nodes(), args.n_hidden),
        ctx=ctx)})

    update_hidden = SSEUpdateHidden(args.n_hidden, 'relu', args.update_dropout, args.use_spmv)
    model = SSEPredict(update_hidden, args.n_hidden, args.predict_dropout)
    model.initialize(ctx=ctx)

    # use optimizer
    num_batches = int(g.number_of_nodes() / args.batch_size)
    scheduler = mx.lr_scheduler.CosineScheduler(args.n_epochs * num_batches,
            args.lr * 10, 0, 0, args.lr/5)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr,
        'lr_scheduler': scheduler})

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        # compute vertex embedding.
        update_hidden(g, None)

        t0 = time.time()
        permute = np.random.permutation(len(train_vs))
        randv = train_vs[permute]
        rand_labels = train_labels[permute]
        data_iter = mx.io.NDArrayIter(data=mx.nd.array(randv, dtype='int64'), label=rand_labels,
                                      batch_size=args.batch_size)
        train_loss = 0
        data = []
        labels = []
        for batch in data_iter:
            data.append(batch.data[0])
            labels.append(batch.label[0])
            if len(data) < args.num_parallel_subgraphs:
                continue

            subgs, seed_ids = subgraph_gen(g, data)
            for subg, seed_id, label, d in zip(subgs, seed_ids, labels, data):
                with mx.autograd.record():
                    logits = model(subg, seed_id)
                    loss = mx.nd.softmax_cross_entropy(logits, label)
                loss.backward()
                trainer.step(d.shape[0])
                train_loss += loss.asnumpy()[0]
            data = []
            labels = []

        #logits = model(eval_vs)
        #eval_loss = mx.nd.softmax_cross_entropy(logits, eval_labels)
        #eval_loss = eval_loss.asnumpy()[0]
        eval_loss = 0

        dur.append(time.time() - t0)
        print("Epoch {:05d} | Train Loss {:.4f} | Eval Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, train_loss, eval_loss, np.mean(dur), n_edges / np.mean(dur) / 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
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
    parser.add_argument("--update-dropout", type=float, default=0.5,
            help="the dropout rate for updating vertex embedding")
    parser.add_argument("--predict-dropout", type=float, default=0.5,
            help="the dropout rate for prediction")
    parser.add_argument("--train_percent", type=float, default=0.5,
            help="the percentage of data used for training")
    parser.add_argument("--use-spmv", type=bool, default=False,
            help="use SpMV for faster speed.")
    parser.add_argument("--num-parallel-subgraphs", type=int, default=1,
            help="the number of subgraphs to construct in parallel.")
    args = parser.parse_args()

    # load and preprocess dataset
    data = load_data(args)
    main(args, data)
