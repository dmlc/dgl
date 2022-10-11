import torch as th

def load_data(data):
    g = data[0]
    g.ndata['features'] = g.ndata.pop('feat')
    g.ndata['labels'] = g.ndata.pop('label')
    return g, data.num_classes

def load_dgl(name):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, YelpDataset, FlickrDataset

    d = {
        'cora': CoraGraphDataset,
        'citeseer': CiteseerGraphDataset,
        'pubmed': PubmedGraphDataset,
        'reddit': RedditDataset,
        'yelp': YelpDataset,
        'flickr': FlickrDataset
    }

    return load_data(d[name]())

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=self_loop)
    return load_data(data)


def load_ogb(name, root="dataset"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata.pop('feat')
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    graph.ndata['labels'] = labels.type(th.LongTensor)
    in_feats = graph.ndata['features'].shape[1]

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print("finish constructing", name)
    return graph, num_labels

def load_dataset(dataset_name):
    multilabel = False
    if dataset_name in ['reddit', 'cora', 'citeseer', 'pubmed', 'yelp', 'flickr']:
        g, n_classes = load_dgl(dataset_name)
        multilabel = dataset_name in ['yelp']
        if multilabel:
            g.ndata['labels'] = g.ndata['labels'].to(dtype=th.float32)
    elif dataset_name in ['ogbn-products', 'ogbn-arxiv', 'ogbn-papers100M']:
        g, n_classes = load_ogb(dataset_name)
    else:
        raise ValueError('unknown dataset')
    
    return g, n_classes, multilabel

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata["train_mask"])
    val_g = g.subgraph(g.ndata["train_mask"] | g.ndata["val_mask"])
    test_g = g
    return train_g, val_g, test_g
