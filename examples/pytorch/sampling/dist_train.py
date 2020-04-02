import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
import torch as th
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.contrib import DistGraphStoreServer, DistGraphStore
from dgl.data.utils import load_graphs
import socket

from gcn_ns_dist import gcn_ns_train

def start_server(args):
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)
    serv = DistGraphStoreServer(server_namebook, args.id, args.graph_name,
                                args.server_data, args.client_data, args.num_client)
    serv.start()

def main(args):
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)
    g = DistGraphStore(server_namebook, args.graph_name)

    # We need to set random seed here. Otherwise, all processes have the same mini-batches.
    th.manual_seed(g.get_id())
    train_mask = g.get_ndata('train_mask').numpy()
    val_mask = g.get_ndata('val_mask').numpy()
    test_mask = g.get_ndata('test_mask').numpy()
    print('part {}, train: {}, val: {}, test: {}'.format(g.get_id(),
        np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)), flush=True)

    train_nid = th.tensor(g.get_local_nids()[train_mask == 1], dtype=th.int64)
    test_nid = th.tensor(g.get_local_nids()[test_mask == 1], dtype=th.int64)

    if args.model == "gcn_ns":
        gcn_ns_train(g, args, args.num_gpus > 0, args.n_classes, train_nid, test_nid)
    else:
        print("unknown model. Please choose from gcn_ns, gcn_cv, graphsage_cv")
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--model", type=str,
                        help="select a model. Valid models: gcn_ns, gcn_cv, graphsage_cv")
    parser.add_argument('--server', action='store_true',
            help='whether this is a server.')
    parser.add_argument('--graph-name', type=str, help='graph name')
    parser.add_argument('--id', type=int,
            help='the partition id')
    parser.add_argument('--n-features', type=int, help='the input feature size')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--server_data', type=str, help='The file with the server data')
    parser.add_argument('--client_data', type=str, help='The file with data exposed to the client.')
    parser.add_argument('--num-client', type=int, help='The number of clients')
    parser.add_argument('--n-classes', type=int, help='the number of classes')
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

    if args.server:
        start_server(args)
    else:
        main(args)
