import argparse
import time
import numpy as np
import networkx as nx
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.mxnet.conv import GMMConv


class MoNet(nn.Block):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 out_feats,
                 n_layers,
                 dim,
                 n_kernels,
                 dropout):
        super(MoNet, self).__init__()
        self.g = g
        with self.name_scope():
            self.layers = nn.Sequential()
            self.pseudo_proj = nn.Sequential()

            # Input layer
            self.layers.add(
                GMMConv(in_feats, n_hidden, dim, n_kernels))
            self.pseudo_proj.add(nn.Dense(dim, in_units=2, activation='tanh'))

            # Hidden layer
            for _ in range(n_layers - 1):
                self.layers.add(GMMConv(n_hidden, n_hidden, dim, n_kernels))
                self.pseudo_proj.add(nn.Dense(dim, in_units=2, activation='tanh'))

            # Output layer
            self.layers.add(GMMConv(n_hidden, out_feats, dim, n_kernels))
            self.pseudo_proj.add(nn.Dense(dim, in_units=2, activation='tanh'))

            self.dropout = nn.Dropout(dropout)

    def forward(self, feat, pseudo):
        h = feat
        for i in range(len(self.layers)):
            if i > 0:
                h = self.dropout(h)
            h = self.layers[i](
                self.g, h, self.pseudo_proj[i](pseudo))
        return h


def evaluate(model, features, pseudo, labels, mask):
    pred = model(features, pseudo).argmax(axis=1)
    accuracy = ((pred == labels) * mask).sum() / mask.sum().asscalar()
    return accuracy.asscalar()

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = nd.array(data.features)
    labels = nd.array(data.labels)
    train_mask = nd.array(data.train_mask)
    val_mask = nd.array(data.val_mask)
    test_mask = nd.array(data.test_mask)

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
        ctx = mx.cpu(0)
    else:
        ctx = mx.gpu(args.gpu)
        print("use cuda:", args.gpu)

    features = features.as_in_context(ctx)
    labels = labels.as_in_context(ctx)
    train_mask = train_mask.as_in_context(ctx)
    val_mask = val_mask.as_in_context(ctx)
    test_mask = test_mask.as_in_context(ctx)

    # graph preprocess and calculate normalization factor
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    us, vs = g.edges()
    pseudo = []
    for i in range(g.number_of_edges()):
        pseudo.append([
            1 / np.sqrt(g.in_degree(us[i].asscalar())),
            1 / np.sqrt(g.in_degree(vs[i].asscalar()))
        ])
    pseudo = nd.array(pseudo, ctx=ctx)

    # create GraphSAGE model
    model = MoNet(g,
                  in_feats,
                  args.n_hidden,
                  n_classes,
                  args.n_layers,
                  args.pseudo_dim,
                  args.n_kernels,
                  args.dropout
                  )
    model.initialize(ctx=ctx)
    n_train_samples = train_mask.sum().asscalar()
    loss_fcn = gluon.loss.SoftmaxCELoss()

    print(model.collect_params())
    trainer = gluon.Trainer(model.collect_params(), 'adam',
            {'learning_rate': args.lr, 'wd': args.weight_decay})


    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            pred = model(features, pseudo)
            loss = loss_fcn(pred, labels, mx.nd.expand_dims(train_mask, 1))
            loss = loss.sum() / n_train_samples

        loss.backward()
        trainer.step(batch_size=1)

        if epoch >= 3:
            loss.asscalar()
            dur.append(time.time() - t0)
            acc = evaluate(model, features, pseudo, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(
                epoch, np.mean(dur), loss.asscalar(), acc, n_edges / np.mean(dur) / 1000))

    # test set accuracy
    acc = evaluate(model, features, pseudo, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoNet on citation network')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--pseudo-dim", type=int, default=2,
                        help="Pseudo coordinate dimensions in GMMConv, 2 for cora and 3 for pubmed")
    parser.add_argument("--n-kernels", type=int, default=3,
                        help="Number of kernels in GMMConv layer")
    parser.add_argument("--weight-decay", type=float, default=5e-5,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)