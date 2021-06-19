from multiprocessing import Process
import argparse, time, math
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '16'
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gcn_ns_sc import gcn_ns_train
from gcn_cv_sc import gcn_cv_train
from graphsage_cv import graphsage_cv_train

def main(args):
    g = dgl.contrib.graph_store.create_graph_from_store(args.graph_name, "shared_mem")
    # We need to set random seed here. Otherwise, all processes have the same mini-batches.
    mx.random.seed(g.worker_id)
    features = g.nodes[:].data['features']
    labels = g.nodes[:].data['labels']
    train_mask = g.nodes[:].data['train_mask']
    val_mask = g.nodes[:].data['val_mask']
    test_mask = g.nodes[:].data['test_mask']

    if args.num_gpus > 0:
        ctx = mx.gpu(g.worker_id % args.num_gpus)
    else:
        ctx = mx.cpu()

    train_nid = mx.nd.array(np.nonzero(train_mask.asnumpy())[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(test_mask.asnumpy())[0]).astype(np.int64)

    n_classes = len(np.unique(labels.asnumpy()))
    n_train_samples = train_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()

    if args.model == "gcn_ns":
        gcn_ns_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples)
    elif args.model == "gcn_cv":
        gcn_cv_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples, True)
    elif args.model == "graphsage_cv":
        graphsage_cv_train(g, ctx, args, n_classes, train_nid, test_nid, n_test_samples, True)
    else:
        print("unknown model. Please choose from gcn_ns, gcn_cv, graphsage_cv")
    print("parent ends")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--model", type=str,
                        help="select a model. Valid models: gcn_ns, gcn_cv, graphsage_cv")
    parser.add_argument("--graph-name", type=str, default="",
            help="graph name")
    parser.add_argument("--num-feats", type=int, default=100,
            help="the number of features")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--num-gpus", type=int, default=0,
            help="the number of GPUs to train")
    parser.add_argument("--lr", type=float, default=3e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=3,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    print(args)

    main(args)
