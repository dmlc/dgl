from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()

    graph = dataset[0]

    train_mask = graph.ndata.pop('train_mask')
    test_mask = graph.ndata.pop('test_mask')

    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, train_mask, test_mask