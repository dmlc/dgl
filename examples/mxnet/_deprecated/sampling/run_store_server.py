import os
import argparse, time, math
import numpy as np
from scipy import sparse as spsp
import mxnet as mx
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

class GraphData:
    def __init__(self, csr, num_feats, graph_name):
        num_nodes = csr.shape[0]
        self.graph = dgl.graph_index.from_csr(csr.indptr, csr.indices, False, 'in')
        self.graph = self.graph.copyto_shared_mem(dgl.contrib.graph_store._get_graph_path(graph_name))
        self.features = mx.nd.random.normal(shape=(csr.shape[0], num_feats))
        self.num_labels = 10
        self.labels = mx.nd.floor(mx.nd.random.uniform(low=0, high=self.num_labels,
                                                       shape=(csr.shape[0])))
        self.train_mask = np.zeros((num_nodes,))
        self.train_mask[np.arange(0, int(num_nodes/2), dtype=np.int64)] = 1
        self.val_mask = np.zeros((num_nodes,))
        self.val_mask[np.arange(int(num_nodes/2), int(num_nodes/4*3), dtype=np.int64)] = 1
        self.test_mask = np.zeros((num_nodes,))
        self.test_mask[np.arange(int(num_nodes/4*3), int(num_nodes), dtype=np.int64)] = 1

def main(args):
    # load and preprocess dataset
    if args.graph_file != '':
        csr = mx.nd.load(args.graph_file)[0]
        n_edges = csr.shape[0]
        graph_name = os.path.basename(args.graph_file)
        data = GraphData(csr, args.num_feats, graph_name)
        csr = None
    else:
        data = load_data(args)
        n_edges = data.graph.number_of_edges()
        graph_name = args.dataset

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    mem_ctx = mx.cpu()

    features = mx.nd.array(data.features, ctx=mem_ctx)
    labels = mx.nd.array(data.labels, ctx=mem_ctx)
    train_mask = mx.nd.array(data.train_mask, ctx=mem_ctx)
    val_mask = mx.nd.array(data.val_mask, ctx=mem_ctx)
    test_mask = mx.nd.array(data.test_mask, ctx=mem_ctx)
    n_classes = data.num_labels

    n_train_samples = train_mask.sum().asscalar()
    n_val_samples = val_mask.sum().asscalar()
    n_test_samples = test_mask.sum().asscalar()

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))

    # create GCN model
    print('graph name: ' + graph_name)
    g = dgl.contrib.graph_store.create_graph_store_server(data.graph, graph_name, "shared_mem",
                                                          args.num_workers, False, edge_dir='in')
    g.ndata['features'] = features
    g.ndata['labels'] = labels
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    g.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--graph-file", type=str, default="",
            help="graph file")
    parser.add_argument("--num-feats", type=int, default=100,
            help="the number of features")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--num-workers", type=int, default=1,
            help="the number of workers")
    args = parser.parse_args()

    main(args)
