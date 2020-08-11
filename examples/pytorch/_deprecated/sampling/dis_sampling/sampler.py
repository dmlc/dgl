import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.contrib.sampling import SamplerPool

class MySamplerPool(SamplerPool):
    def worker(self, args):

        number_hops = 1
        if args.model == "gcn_ns":
            number_hops = args.n_layers + 1
        elif args.model == "gcn_cv":
            number_hops = args.n_layers
        else:
            print("unknown model. Please choose from gcn_ns and gcn_cv")

        # Start sender
        namebook = { 0:args.ip }
        sender = dgl.contrib.sampling.SamplerSender(namebook)

        # load and preprocess dataset
        data = load_data(args)

        if args.self_loop and not args.dataset.startswith('reddit'):
            data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

        train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
        test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)

        # create GCN model
        g = DGLGraph(data.graph, readonly=True)

        while True:
            idx = 0
            for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                           args.num_neighbors,
                                                           neighbor_type='in',
                                                           shuffle=True,
                                                           num_workers=32,
                                                           num_hops=number_hops,
                                                           seed_nodes=train_nid):
                print("send train nodeflow: %d" % (idx))
                sender.send(nf, 0)
                idx += 1
            sender.signal(0)

def main(args):
    pool = MySamplerPool()
    pool.start(args.num_sampler, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--model", type=str,
                        help="select a model. Valid models: gcn_ns, gcn_cv")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
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
    parser.add_argument("--ip", type=str, default='127.0.0.1:50051',
            help="IP address")
    parser.add_argument("--num-sampler", type=int, default=1,
            help="number of sampler")
    args = parser.parse_args()

    print(args)

    main(args)


