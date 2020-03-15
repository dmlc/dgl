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

class DistGraphStoreServer(KVServer):
    def __init__(self, ip_config, server_id, server_data, num_client):
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print('Server {}: host name: {}, ip: {}'.format(server_id, host_name, host_ip))

        server_namebook = dgl.contrib.read_ip_config(filename=ip_config)
        super(DistGraphStoreServer, self).__init__(server_id=server_id, server_namebook=server_namebook, num_client=num_client)

        self.part_g = load_graphs(server_data)[0][0]
        num_nodes = np.max(self.part_g.ndata[dgl.NID].asnumpy()) + 1
        self.g2l = mx.nd.zeros((num_nodes), dtype=np.int64)
        self.g2l[:] = -1
        self.g2l[self.part_g.ndata[dgl.NID]] = mx.nd.arange(self.part_g.number_of_nodes())
        if self.get_id() % self.get_group_count() == 0: # master server
            for ndata_name in self.part_g.ndata.keys():
                if '_mask' in ndata_name:
                    print(ndata_name, ':', np.nonzero(self.part_g.ndata[ndata_name].asnumpy())[0])
                self.set_global2local(name=ndata_name, global2local=self.g2l)
                self.init_data(name=ndata_name, data_tensor=self.part_g.ndata[ndata_name])
        else:
            for ndata_name in self.part_g.ndata.keys():
                self.set_global2local(name=ndata_name)
                self.init_data(name=ndata_name)
        # TODO Do I need synchronization?

    def _pull_handler(self, name, lID, target):
        #lID = self.g2l[gID].asnumpy()
        #gID = gID.asnumpy()
        #print(gID[lID == -1])
        assert np.sum(lID == -1) == 0
        return target[name][lID]

def start_server(args):
    serv = DistGraphStoreServer(args.ip_config, args.id, args.server_data, args.num_client)
    serv.start()

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
        partition = mx.nd.zeros(shape=(num_nodes,), dtype=np.int64)
        partition[self.g.ndata[dgl.NID]] = self.g.ndata['part_id']
        # TODO what is the node data name?
        self._client.set_partition_book(name='feats', partition_book=partition)
        self._client.set_partition_book(name='labels', partition_book=partition)
        self._client.set_partition_book(name='test_mask', partition_book=partition)
        self._client.set_partition_book(name='val_mask', partition_book=partition)
        self._client.set_partition_book(name='train_mask', partition_book=partition)

        self._client.barrier()

        self.local_nids = np.nonzero((self.g.ndata['part_id'] == self.part_id).asnumpy())[0]
        self.local_gnid = self.g.ndata[dgl.NID][self.local_nids]

    def number_of_nodes(self):
        return len(self.local_gnid)

    def get_local_nids(self):
        return self.local_nids

    def get_id(self):
        return self.part_id

    def get_ndata(self, name, nids=None):
        if nids is None:
            gnid = self.local_gnid
        else:
            gnid = self.g.ndata[dgl.NID][nids]
        return self._client.pull(name=name, id_tensor=gnid)

def main(args):
    g = DistGraphStore(args.ip_config, args.graph_path)

    # We need to set random seed here. Otherwise, all processes have the same mini-batches.
    mx.random.seed(g.get_id())
    train_mask = g.get_ndata('train_mask').asnumpy()
    val_mask = g.get_ndata('val_mask').asnumpy()
    test_mask = g.get_ndata('test_mask').asnumpy()
    print('part {}, train: {}, val: {}, test: {}'.format(g.get_id(),
        np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)), flush=True)

    if args.num_gpus > 0:
        ctx = mx.gpu(g.worker_id % args.num_gpus)
    else:
        ctx = mx.cpu()

    train_nid = g.get_local_nids()[train_mask == 1]
    test_nid = g.get_local_nids()[test_mask == 1]

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
    parser.add_argument('--n-features', type=int, help='the input feature size')
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
