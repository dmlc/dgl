import dgl
import torch as th

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    train_mask = data.train_mask
    val_mask = data.val_mask
    features = th.Tensor(data.features)
    labels = th.LongTensor(data.labels)

    # Construct graph
    g = data.graph
    g.ndata['features'] = features
    g.ndata['labels'] = labels
    g.ndata['train_mask'] = th.LongTensor(data.train_mask)
    g.ndata['val_mask'] = th.LongTensor(data.val_mask)
    g.ndata['test_mask'] = th.LongTensor(data.test_mask)
    return g, data.num_labels

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
    return graph, len(th.unique(graph.ndata['labels']))
