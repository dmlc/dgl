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
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def gcn_msg(edge):
    # TODO should we use concat?
    return {'m': mx.nd.concat(edge.src['in'], edge.src['h'], dim=1)}

def gcn_reduce(node):
    return {'accum': mx.nd.sum(node.mailbox['m'], 1) / node.mailbox['m'].shape[1]}

class NodeUpdate(gluon.Block):
    def __init__(self, out_feats, activation=None, alpha=0.1):
        super(NodeUpdate, self).__init__()
        self.linear1 = gluon.nn.Dense(out_feats, activation=activation)
        # TODO what is the dimension here?
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

class SSEUpdateHidden(gluon.Block):
    def __init__(self, g, features,
                 n_hidden,
                 dropout,
                 activation):
        super(SSEUpdateHidden, self).__init__()
        self.g = g
        self.deg = mx.nd.expand_dims(g.in_degrees(np.arange(0, g.number_of_nodes())), 1).astype(np.float32)
        self.layer = NodeUpdate(n_hidden, activation)
        self.dropout = dropout

    def forward(self, vertices):
        if vertices is None:
            feat = self.g.get_n_repr()['in']
            cat = mx.nd.concat(feat, self.g.ndata['h'], dim=1)
            accum = mx.nd.dot(self.g.adjacency_matrix(), cat) / self.deg
            return self.layer(feat, self.g.ndata['h'], accum)
        else:
            # We don't need dropout for inference.
            if self.dropout:
                # TODO here we apply dropout on all vertex representation.
                self.g.ndata['h'] = mx.nd.Dropout(self.g.ndata['h'], p=self.dropout)
            feat = self.g.get_n_repr()['in']
            cat = mx.nd.concat(feat, self.g.ndata['h'], dim=1)
            slices = mx.nd.take(self.g.adjacency_matrix(), vertices)
            accum = mx.nd.dot(slices, cat) / mx.nd.take(self.deg, vertices)
            return self.layer(mx.nd.take(feat, vertices),
                              mx.nd.take(self.g.ndata['h'], vertices), accum)

class DGLSSEUpdateHidden(gluon.Block):
    def __init__(self,
                 g,
                 features,
                 n_hidden,
                 activation,
                 dropout,
                 use_spmv):
        super(DGLSSEUpdateHidden, self).__init__()
        self.g = g
        self.g.set_n_repr({'in': features,
                           'h': mx.nd.random.normal(shape=(g.number_of_nodes(), n_hidden), ctx=features.context)})
        self.layer = DGLNodeUpdate(NodeUpdate(n_hidden, activation))
        self.dropout = dropout
        self.use_spmv = use_spmv
        self.deg = mx.nd.expand_dims(g.in_degrees(np.arange(0, g.number_of_nodes())), 1).astype(np.float32)

    def forward(self, vertices):
        if self.use_spmv:
            feat = self.g.get_n_repr()['in']
            self.g.set_n_repr({'cat': mx.nd.concat(feat, self.g.ndata['h'], dim=1)})
            msg_func = fn.copy_src(src='cat', out='m')
            reduce_func = fn.sum(msg='m', out='accum')
        else:
            msg_func = gcn_msg
            reduce_func = gcn_reduce
        if vertices is None:
            self.g.update_all(msg_func, reduce_func, None)
            if self.use_spmv:
                self.g.ndata['accum'] = self.g.ndata['accum'] / self.deg
            self.g.apply_nodes(self.layer)
            return self.g.get_n_repr()['h1']
        else:
            # We don't need dropout for inference.
            if self.dropout:
                # TODO here we apply dropout on all vertex representation.
                self.g.ndata['h'] = mx.nd.Dropout(self.g.ndata['h'], p=self.dropout)
            self.g.pull(vertices, msg_func, reduce_func, None)
            if self.use_spmv:
                self.g.ndata['accum'] = self.g.ndata['accum'] / self.deg
            self.g.apply_nodes(self.layer, vertices)
            return self.g.get_n_repr()['h1'][vertices]

class SSEPredict(gluon.Block):
    def __init__(self, update_hidden, out_feats, dropout):
        super(SSEPredict, self).__init__()
        self.linear1 = gluon.nn.Dense(out_feats, activation='relu')
        self.linear2 = gluon.nn.Dense(out_feats)
        self.update_hidden = update_hidden
        self.dropout = dropout

    def forward(self, vertices):
        hidden = self.update_hidden(vertices)
        if self.dropout:
            hidden = mx.nd.Dropout(hidden, p=self.dropout)
        return self.linear2(self.linear1(hidden)), hidden

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
    g = DGLGraph(data.graph, readonly=True)
    update_hidden = DGLSSEUpdateHidden(g, features, args.n_hidden, 'relu', args.update_dropout, args.use_spmv)
    if not args.dgl:
        update_hidden = SSEUpdateHidden(g, features, args.n_hidden, args.update_dropout, 'relu')
    model = SSEPredict(update_hidden, args.n_hidden, args.predict_dropout)
    model.initialize(ctx=ctx)

    # use optimizer
    num_batches = int(g.number_of_nodes() / args.batch_size)
    scheduler = mx.lr_scheduler.CosineScheduler(args.n_epochs * num_batches,
            args.lr * 10, 0, 0, args.lr/5)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr,
        'lr_scheduler': scheduler})

    rets = []
    rets.append(g.get_n_repr()['h'])

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        # compute vertex embedding.
        all_hidden = update_hidden(None)
        g.ndata['h'] = all_hidden
        rets.append(all_hidden)

        t0 = time.time()
        permute = np.random.permutation(len(train_vs))
        permute = permute[0:int(len(permute) / 3)]
        randv = train_vs[permute]
        rand_labels = train_labels[permute]
        data_iter = mx.io.NDArrayIter(data=mx.nd.array(randv, dtype='int64'), label=rand_labels,
                                      batch_size=args.batch_size)
        train_loss = 0
        for batch in data_iter:
            # TODO this isn't exactly how the model is trained.
            # We should enable the semi-supervised training.
            with mx.autograd.record():
                logits, hidden = model(mx.nd.array(batch.data[0], dtype=np.int64))
                loss = mx.nd.softmax_cross_entropy(logits, batch.label[0])
            loss.backward()
            trainer.step(batch.data[0].shape[0])
            train_loss += loss.asnumpy()[0]

        logits, hidden = model(mx.nd.array(eval_vs, dtype=np.int64))
        eval_loss = mx.nd.softmax_cross_entropy(logits, eval_labels)
        eval_loss = eval_loss.asnumpy()[0]

        dur.append(time.time() - t0)
        print("Epoch {:05d} | Train Loss {:.4f} | Eval Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, train_loss, eval_loss, np.mean(dur), n_edges / np.mean(dur) / 1000))

    return rets

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
    parser.add_argument("--update-dropout", type=float, default=0,
            help="the dropout rate for updating vertex embedding")
    parser.add_argument("--predict-dropout", type=float, default=0,
            help="the dropout rate for prediction")
    parser.add_argument("--train_percent", type=float, default=0.5,
            help="the percentage of data used for training")
    parser.add_argument("--use-spmv", action="store_true",
            help="use SpMV for faster speed.")
    parser.add_argument("--dgl", action="store_true")
    args = parser.parse_args()

    # load and preprocess dataset
    data = load_data(args)
    rets1 = main(args, data)
    rets2 = main(args, data)
    for hidden1, hidden2 in zip(rets1, rets2):
        print("hidden: " + str(mx.nd.sum(mx.nd.abs(hidden1 - hidden2)).asnumpy()))
