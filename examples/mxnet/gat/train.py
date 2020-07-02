"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import networkx as nx
import time
import mxnet as mx
from mxnet import gluon
import numpy as np
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gat import GAT
from utils import EarlyStopping

def elu(data):
    return mx.nd.LeakyReLU(data, act_type='elu')


def evaluate(model, features, labels, mask):
    logits = model(features)
    logits = logits[mask].asnumpy().squeeze()
    val_labels = labels[mask].asnumpy().squeeze()
    max_index = np.argmax(logits, axis=1)
    accuracy = np.sum(np.where(max_index == val_labels, 1, 0)) / len(val_labels)
    return accuracy


def main(args):
    # load and preprocess dataset
    data = load_data(args)
    g = data.g

    if args.gpu < 0:
        cuda = False
        ctx = mx.cpu()
    else:
        cuda = True
        ctx = mx.gpu(args.gpu)
        g = g.to(ctx)
    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()
    train_mask = mx.nd.array(np.where(g.ndata['train_mask'].asnumpy() == 1))
    val_mask = mx.nd.array(np.where(g.ndata['val_mask'].asnumpy() == 1))
    test_mask = mx.nd.array(np.where(g.ndata['test_mask'].asnumpy() == 1))
    labels = mx.nd.array(g.ndata['label'], dtype='float32')
    features = g.ndata['feat']

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

    # add self-loop
    _, _, self_e = g.edge_ids(g.nodes(), g.nodes(), return_uv=True)
    g.remove_edges(self_e)
    g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                in_feats,
                args.num_hidden,
                n_classes,
                heads,
                elu,
                args.in_drop,
                args.attn_drop,
                args.alpha,
                args.residual)

    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    model.initialize(ctx=ctx)

    # use optimizer
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})

    dur = []
    for epoch in range(args.epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        with mx.autograd.record():
            logits = model(features)
            loss = mx.nd.softmax_cross_entropy(logits[train_mask].squeeze(), labels[train_mask].squeeze())
            loss.backward()
        trainer.step(train_mask.shape[0])

        if epoch >= 3:
            dur.append(time.time() - t0)
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, loss.asnumpy()[0], np.mean(dur), n_edges / np.mean(dur) / 1000))
        val_accuracy = evaluate(model, features, labels, val_mask)
        print("Validation Accuracy {:.4f}".format(val_accuracy))
        if args.early_stop:
            if stopper.step(val_accuracy, model): 
                break
    print()

    if args.early_stop:
        model.load_parameters('model.param')
    test_accuracy = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(test_accuracy))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    args = parser.parse_args()
    print(args)

    main(args)
