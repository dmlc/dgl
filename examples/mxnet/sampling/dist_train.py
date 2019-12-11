import os
from multiprocessing import Process
import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.contrib import KVServer

from gcn_ns_dist import gcn_ns_train

server_namebook, client_namebook = dgl.contrib.ReadNetworkConfigure('config.txt')

def load_node_data(args):
    if args.num_parts > 1:
        import pickle
        ndata = pickle.load(open('Reddit/reddit_ndata.pkl', 'rb'))
        print('load reddit ndata')
        return ndata
    else:
        data = load_data(args)
        features = mx.nd.array(data.features)
        labels = mx.nd.array(data.labels)
        train_mask = mx.nd.array(data.train_mask)
        val_mask = mx.nd.array(data.val_mask)
        test_mask = mx.nd.array(data.test_mask)
        return {'feature': features,
                'label': labels,
                'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask}

def start_server(args):
    server = KVServer(
            server_id=args.id,
            client_namebook=client_namebook,
            server_addr=server_namebook[args.id])

    ndata = load_node_data(args)
    graph_name = args.graph_name

    # Initialize data on kvstore, the data_tensor is shared-memory data
    for key, val in ndata.items():
        data_name = graph_name + '_' + key
        print('server init {} from ndata[{}]'.format(data_name, key))
        server.init_data(name=data_name, data_tensor=mx.nd.array(ndata[key]))

    server.start()

    exit()  # exit program directly when finishing training

def connect_to_kvstore(args, partition_book):
    client = dgl.contrib.KVClient(
        client_id=args.id,
        local_server_id=args.id,
        server_namebook=server_namebook,
        client_addr=client_namebook[args.id],
        partition_book=partition_book)

    ndata = load_node_data(args)
    graph_name = args.graph_name
    # Initialize data on kvstore, the data_tensor is shared-memory data
    for key, val in ndata.items():
        data_name = graph_name + '_' + key
        print('client init {} from ndata[{}]'.format(data_name, key))
        client.init_local_data(name=data_name, data_tensor=mx.nd.array(ndata[key]))

    client.connect()

    return client

def load_local_part(args):
    # We need to know:
    # * local nodes and local edges.
    # * mapping to global nodes and global edges.
    # * mapping to the right machine.
    # TODO for now, I use pickle to store partitioned graph.
    if args.num_parts > 1:
        import pickle
        part, part_nodes, part_loc = pickle.load(open('Reddit/reddit_part_{}.pkl'.format(args.id), 'rb'))
        all_locs = np.loadtxt('Reddit/reddit.adj.part.{}'.format(args.num_parts))
        g = dgl.DGLGraph(part, readonly=True)
        g.ndata['global_id'] = mx.nd.array(part_nodes, dtype=np.int64)
        g.ndata['node_loc'] = mx.nd.array(part_loc, dtype=np.int64)
        g.ndata['local'] = mx.nd.array(part_loc == args.id, dtype=np.int64)
        assert np.all(all_locs[part_nodes] == part_loc)
        return g, mx.nd.array(all_locs, dtype=np.int64)
    else:
        data = load_data(args)
        g = dgl.DGLGraph(data.graph, readonly=True)
        g.ndata['global_id'] = mx.nd.arange(g.number_of_nodes(), dtype=np.int64)
        g.ndata['node_loc'] = mx.nd.zeros(g.number_of_nodes(), dtype=np.int64)
        g.ndata['local'] = mx.nd.ones(g.number_of_nodes(), dtype=np.int64)
        return g, g.ndata['node_loc']

def get_from_kvstore(args, kv, g, name):
    name = args.graph_name + "_" + name
    print('client pull ' + name)
    return kv.pull(name=name, id_tensor=g.ndata['global_id'])

def main(args):
    args.id = int(os.environ['DMLC_TASK_ID'])
    g, all_locs = load_local_part(args)
    print('graph size:', g.number_of_nodes())
    print('#inner nodes:', mx.nd.sum(g.ndata['local']).asnumpy())
    kv = connect_to_kvstore(args, all_locs)
    # We need to set random seed here. Otherwise, all processes have the same mini-batches.
    mx.random.seed(args.id)
    local_mask = g.ndata['local'].astype(np.float32)
    train_mask = get_from_kvstore(args, kv, g, 'train_mask').astype(np.float32) * local_mask
    val_mask = get_from_kvstore(args, kv, g, 'val_mask').astype(np.float32) * local_mask
    test_mask = get_from_kvstore(args, kv, g, 'test_mask').astype(np.float32) * local_mask
    print('train: {}, val: {}, test: {}'.format(mx.nd.sum(train_mask).asnumpy(),
        mx.nd.sum(val_mask).asnumpy(),
        mx.nd.sum(test_mask).asnumpy()))

    if args.num_gpus > 0:
        ctx = mx.gpu(g.worker_id % args.num_gpus)
    else:
        ctx = mx.cpu()

    train_nid = mx.nd.array(np.nonzero(train_mask.asnumpy())[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(test_mask.asnumpy())[0]).astype(np.int64)

    if args.model == "gcn_ns":
        gcn_ns_train(g, kv, ctx, args, args.n_classes, train_nid, test_nid)
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
    parser.add_argument('--id', type=int,
            help='the partition id')
    parser.add_argument('--num-parts', type=int,
            help='the number of partitions')
    parser.add_argument('--n-classes', type=int,
            help='the number of classes')
    parser.add_argument('--n-features', type=int,
            help='the number of input features')
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

    if args.server:
        start_server(args)
    else:
        main(args)
