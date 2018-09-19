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
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

def gcn_msg(src, edge):
    # TODO should we use concat?
    return mx.nd.concat(src['in'], src['h'], dim=1)

def gcn_reduce(node, msgs):
    return {'accum': mx.nd.sum(msgs, 1)}

class NodeUpdateModule(gluon.Block):
    def __init__(self, out_feats, activation=None):
        super(NodeUpdateModule, self).__init__()
        self.linear1 = gluon.nn.Dense(out_feats, activation=activation)
        # TODO what is the dimension here?
        self.linear2 = gluon.nn.Dense(out_feats)

    def forward(self, node):
        node = mx.nd.concat(node['in'], node['accum'], dim=1)
        return self.linear2(self.linear1(node))

class SSE(gluon.Block):
    def __init__(self,
                 g,
                 features,
                 n_hidden,
                 activation):
        super(SSE, self).__init__()
        self.g = g
        self.g.set_n_repr({'in': features,
                           'h': mx.nd.random.normal(shape=(g.number_of_nodes(), n_hidden))})
        self.layer = NodeUpdateModule(n_hidden, activation)

    def forward(self, vertices):
        # TODO we should support NDArray for vertex IDs.
        vs = vertices.asnumpy()
        return self.g.pull(vs, gcn_msg, gcn_reduce, self.layer,
                           batchable=True, writeback=False)

def main(args):
    # load and preprocess dataset
    data = load_data(args)

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    mask = mx.nd.array(data.train_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu <= 0:
        cuda = False
        ctx = mx.cpu(0)
    else:
        cuda = True
        features = features.as_in_context(mx.gpu(0))
        labels = labels.as_in_context(mx.gpu(0))
        mask = mask.as_in_context(mx.gpu(0))
        ctx = mx.gpu(0)

    # create the SSE model
    g = DGLGraph(data.graph)
    model = SSE(g,
                features,
                args.n_hidden,
                'relu')
    model.initialize(ctx=ctx)

    # use optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        t0 = time.time()
        randv = np.random.permutation(g.number_of_nodes())
        rand_labels = labels[randv]
        data_iter = mx.io.NDArrayIter(data=mx.nd.array(randv, dtype='int32'), label=rand_labels,
                                      batch_size=args.batch_size)
        for batch in data_iter:
            # TODO this isn't exactly how the model is trained.
            # We should enable the semi-supervised training.
            with mx.autograd.record():
                logits = model(batch.data[0])
                loss = mx.nd.softmax_cross_entropy(logits, batch.label[0])
            loss.backward()
            trainer.step(batch.data[0].shape[0])

        dur.append(time.time() - t0)
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, loss.asnumpy()[0], np.mean(dur), n_edges / np.mean(dur) / 1000))

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
    args = parser.parse_args()

    main(args)
