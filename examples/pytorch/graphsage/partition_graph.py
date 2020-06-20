
import dgl
import numpy as np
import torch as th
import argparse
import time

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    train_mask = data.train_mask
    val_mask = data.val_mask
    features = th.Tensor(data.features)
    labels = th.LongTensor(data.labels)

    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features
    g.ndata['labels'] = labels
    g.ndata['train_mask'] = th.LongTensor(data.train_mask)
    g.ndata['val_mask'] = th.LongTensor(data.val_mask)
    g.ndata['test_mask'] = th.LongTensor(data.test_mask)
    return g

def load_ogb(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    data = DglNodePropPredDataset(name=name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.int64)
    train_mask[train_nid] = 1
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.int64)
    val_mask[val_nid] = 1
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.int64)
    test_mask[test_nid] = 1
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    return graph

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogb-product, ogb-paper100M')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        g = load_reddit()
    elif args.dataset == 'ogb-product':
        g = load_ogb('ogbn-products')
    elif args.dataset == 'ogb-paper100M':
        g = load_ogb('ogbn-papers100M')
    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(th.sum(g.ndata['train_mask']),
                                                  th.sum(g.ndata['val_mask']),
                                                  th.sum(g.ndata['test_mask'])))
    dgl.distributed.partition_graph(g, args.dataset, args.num_parts, 'data')
