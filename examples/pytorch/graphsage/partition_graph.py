import dgl
import numpy as np
import torch as th
import argparse
import time

from load_graph import load_reddit, load_ogb

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogb-product, ogb-paper100M')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        g, _ = load_reddit()
    elif args.dataset == 'ogb-product':
        g, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogb-paper100M':
        g, _ = load_ogb('ogbn-papers100M')
    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(th.sum(g.ndata['train_mask']),
                                                  th.sum(g.ndata['val_mask']),
                                                  th.sum(g.ndata['test_mask'])))
    dgl.distributed.partition_graph(g, args.dataset, args.num_parts, 'data')
