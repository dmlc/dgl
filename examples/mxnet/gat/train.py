"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
Compared with the original paper, this code does not implement
early stopping.
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

    features = mx.nd.array(data.features)
    labels = mx.nd.array(data.labels)
    mask = mx.nd.array(np.where(data.train_mask == 1))
    test_mask = mx.nd.array(np.where(data.test_mask == 1))
    val_mask = mx.nd.array(np.where(data.val_mask == 1))
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu < 0:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)
        features = features.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        mask = mask.as_in_context(ctx)
        test_mask = test_mask.as_in_context(ctx)
        val_mask = val_mask.as_in_context(ctx)
    # create graph
    g = data.graph
    # add self-loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
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
            loss = mx.nd.softmax_cross_entropy(logits[mask].squeeze(), labels[mask].squeeze())
            loss.backward()
        trainer.step(mask.shape[0])

        if epoch >= 3:
            dur.append(time.time() - t0)
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, loss.asnumpy()[0], np.mean(dur), n_edges / np.mean(dur) / 1000))
        val_accuracy = evaluate(model, features, labels, val_mask)
        print("Validation Accuracy {:.4f}".format(val_accuracy))
        if stopper.step(val_accuracy, model): 
            break
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
    args = parser.parse_args()
    print(args)

    main(args)
