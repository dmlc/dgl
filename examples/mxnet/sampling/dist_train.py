import os
os.environ['DGLBACKEND']='mxnet'
if 'DMLC_ROLE' in os.environ and os.environ['DMLC_ROLE'] == 'worker':
    os.environ['OMP_NUM_THREADS']='8'
else:
    os.environ['OMP_NUM_THREADS']='4'
from multiprocessing import Process
import argparse, time, math
import numpy as np
import mxnet as mx
from mxnet import gluon
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.contrib import KVServer, KVClient
from dgl.data.utils import load_graphs
import socket

from gcn_ns_dist import gcn_ns_train

'''
def load_node_data(args):
    if args.num_parts > 1:
        import pickle
        ndata = pickle.load(open('Reddit8/reddit_ndata.pkl', 'rb'))
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
'''

class DistGraphStoreServer:
    def __init__(self, ip_config, server_id, server_data, num_client):
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print('Server {}: host name: {}, ip: {}'.format(server_id, host_name, host_ip))

        server_namebook = dgl.contrib.read_ip_config(filename=ip_config)
        self._server = KVServer(server_id=server_id, server_namebook=server_namebook, num_client=num_client)

        part_g = load_graphs(server_data)[0][0]
        num_nodes = np.max(part_g.ndata[dgl.NID].asnumpy()) + 1
        g2l = mx.nd.zeros((num_nodes), dtype=np.int64)
        g2l[part_g.ndata[dgl.NID]] = mx.nd.arange(part_g.number_of_nodes())
        if self._server.get_id() % self._server.get_group_count() == 0: # master server
            for ndata_name in part_g.ndata.keys():
                print(ndata_name)
                self._server.set_global2local(name=ndata_name, global2local=g2l)
                self._server.init_data(name=ndata_name, data_tensor=part_g.ndata[ndata_name])
        else:
            for ndata_name in part_g.ndata.keys():
                self._server.set_global2local(name=ndata_name)
                self._server.init_data(name=ndata_name)
        # TODO Do I need synchronization?

    def start(self):
        self._server.print()
        self._server.start()

def start_server(args):
    serv = DistGraphStoreServer(args.ip_config, args.id, args.server_data, args.num_client)
    serv.start()

'''
def load_local_part(args):
    # We need to know:
    # * local nodes and local edges.
    # * mapping to global nodes and global edges.
    # * mapping to the right machine.
    # TODO for now, I use pickle to store partitioned graph.
    if args.num_parts > 1:
        import pickle
        part, part_nodes, part_loc = pickle.load(open('Reddit8/reddit_part_{}.pkl'.format(args.id), 'rb'))
        all_locs = np.loadtxt('Reddit8/reddit.adj.part.{}'.format(args.num_parts))
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
'''

class DistGraphStore:
    def __init__(self, ip_config, graph_path):
        server_namebook = dgl.contrib.read_ip_config(filename=ip_config)
        self._client = KVClient(server_namebook=server_namebook)
        self._client.connect()

        # TODO this cannot guarantee data locality.
        self.part_id = self._client.get_id()
        self.g = load_graphs(graph_path + '/client-' + str(self.part_id) + '.dgl')[0][0]
        # TODO If we don't have HALO nodes, how do we set partition?
        num_nodes = np.max(self.g.ndata[dgl.NID].asnumpy()) + 1
        partition = mx.nd.ones(shape=(num_nodes,), dtype=np.int64)
        partition[self.g.ndata[dgl.NID]] = mx.nd.arange(self.g.number_of_nodes())
        # TODO what is the node data name?
        self._client.set_partition_book(name='feats', partition_book=partition)
        self._client.set_partition_book(name='labels', partition_book=partition)
        self._client.set_partition_book(name='test_mask', partition_book=partition)
        self._client.set_partition_book(name='val_mask', partition_book=partition)
        self._client.set_partition_book(name='train_mask', partition_book=partition)

        self._client.print()
        self._client.barrier()

        local_nids = np.nonzero((self.g.ndata['part_id'] == self.part_id).asnumpy())[0]
        self.local_gnid = self.g.ndata[dgl.NID][local_nids]

    def get_id(self):
        return self.part_id

    def get_ndata(self, name, nids=None):
        if nids is None:
            gnid = self.local_gnid
        else:
            gnid = self.local_gnid[nids]
        return self._client.pull(name=name, id_tensor=gnid)

def main(args):
    g = DistGraphStore(args.ip_config, args.graph_path)

    # We need to set random seed here. Otherwise, all processes have the same mini-batches.
    mx.random.seed(g.get_id())
    train_mask = g.get_ndata('train_mask').astype(np.float32)
    val_mask = g.get_ndata('val_mask').astype(np.float32)
    test_mask = g.get_ndata('test_mask').astype(np.float32)
    print('train: {}, val: {}, test: {}'.format(mx.nd.sum(train_mask).asnumpy(),
        mx.nd.sum(val_mask).asnumpy(),
        mx.nd.sum(test_mask).asnumpy()), flush=True)

    if args.num_gpus > 0:
        ctx = mx.gpu(g.worker_id % args.num_gpus)
    else:
        ctx = mx.cpu()

    train_nid = mx.nd.array(np.nonzero(train_mask.asnumpy())[0]).astype(np.int64)
    test_nid = mx.nd.array(np.nonzero(test_mask.asnumpy())[0]).astype(np.int64)
    print('test5')

    if args.model == "gcn_ns":
        gcn_ns_train(g, ctx, args, args.n_classes, train_nid, test_nid)
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
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--server_data', type=str, help='The file with the server data')
    parser.add_argument('--num-client', type=int, help='The number of clients')
    parser.add_argument('--n-classes', type=int, help='the number of classes')
    parser.add_argument('--graph-path', type=str, help='the directory that stores graph structure')
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
