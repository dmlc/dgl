import dgl
import numpy as np
import torch as th
import argparse
import time

from ogb.nodeproppred import DglNodePropPredDataset

def load_ogb(dataset, global_norm):
    if dataset == 'ogbn-mag':
        dataset = DglNodePropPredDataset(name=dataset)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]['paper']
        val_idx = split_idx["valid"]['paper']
        test_idx = split_idx["test"]['paper']
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
        paper_labels = labels['paper'].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_classes))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))

        # currently we do not support node feature in mag dataset.
        # calculate norm for each edge type and store in edge
        if global_norm is False:
            for canonical_etype in hg.canonical_etypes:
                u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
                _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
                degrees = count[inverse_index]
                norm = th.ones(eid.shape[0]) / degrees
                norm = norm.unsqueeze(1)
                hg.edges[canonical_etype].data['norm'] = norm

        # get target category id
        category_id = len(hg.ntypes)
        for i, ntype in enumerate(hg.ntypes):
            if ntype == category:
                category_id = i

        g = dgl.to_homo(hg)
        if global_norm:
            u, v, eid = g.all_edges(form='all')
            _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = th.ones(eid.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            g.edata['norm'] = norm

        node_ids = th.arange(g.number_of_nodes())
        # find out the target node ids
        node_tids = g.ndata[dgl.NTYPE]
        loc = (node_tids == category_id)
        target_idx = node_ids[loc]
        train_idx = target_idx[train_idx]
        val_idx = target_idx[val_idx]
        test_idx = target_idx[test_idx]
        train_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
        train_mask[train_idx] = True
        val_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
        val_mask[val_idx] = True
        test_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
        test_mask[test_idx] = True
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        labels = th.full((g.number_of_nodes(),), -1, dtype=paper_labels.dtype)
        labels[target_idx] = paper_labels
        g.ndata['labels'] = labels
        return g
    else:
        raise("Do not support other ogbn datasets.")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='ogbn-mag',
                           help='datasets: ogbn-mag')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--global-norm', default=False, action='store_true',
                           help='User global norm instead of per node type norm')
    args = argparser.parse_args()

    start = time.time()
    g = load_ogb(args.dataset, args.global_norm)

    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(th.sum(g.ndata['train_mask']),
                                                  th.sum(g.ndata['val_mask']),
                                                  th.sum(g.ndata['test_mask'])))

    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    dgl.distributed.partition_graph(g, args.dataset, args.num_parts, 'data',
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges)
