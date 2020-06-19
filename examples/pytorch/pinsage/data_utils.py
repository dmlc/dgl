import torch
import dgl
import numpy as np
import scipy.sparse as ssp

# This is the train-test split method most of the recommender system papers running on MovieLens
# takes.  It essentially follows the intuition of "training on the past and predict the future".
# One can also change the threshold to make validation and test set take larger proportions.
def train_test_split_by_time(g, column, etype, itype):
    n_edges = g.number_of_edges(etype)
    with g.local_scope():
        def splits(edges):
            num_edges, count = edges.data['train_mask'].shape

            # sort by timestamp
            _, sorted_idx = edges.data[column].sort(1)

            train_mask = edges.data['train_mask']
            val_mask = edges.data['val_mask']
            test_mask = edges.data['test_mask']

            x = torch.arange(num_edges)

            # If one user has more than one interactions, select the latest one for test.
            if count > 1:
                train_mask[x, sorted_idx[:, -1]] = False
                test_mask[x, sorted_idx[:, -1]] = True
            # If one user has more than two interactions, select the second latest one for validation.
            if count > 2:
                train_mask[x, sorted_idx[:, -2]] = False
                val_mask[x, sorted_idx[:, -2]] = True
            return {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}

        g.edges[etype].data['train_mask'] = torch.ones(n_edges, dtype=torch.bool)
        g.edges[etype].data['val_mask'] = torch.zeros(n_edges, dtype=torch.bool)
        g.edges[etype].data['test_mask'] = torch.zeros(n_edges, dtype=torch.bool)
        g.nodes[itype].data['count'] = g.in_degrees(etype=etype)
        g.group_apply_edges('src', splits, etype=etype)

        train_indices = g.filter_edges(lambda edges: edges.data['train_mask'], etype=etype)
        val_indices = g.filter_edges(lambda edges: edges.data['val_mask'], etype=etype)
        test_indices = g.filter_edges(lambda edges: edges.data['test_mask'], etype=etype)

    return train_indices, val_indices, test_indices

def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        preserve_nodes=True)
    # remove the induced node IDs - should be assigned by model instead
    del train_g.nodes[utype].data[dgl.NID]
    del train_g.nodes[itype].data[dgl.NID]

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g

def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix

def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / \
        (values.max(0, keepdims=True) - values.min(0, keepdims=True))
