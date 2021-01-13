import dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset


def get_graph(dataset):
    if dataset == 'reddit':
        reddit = dgl.data.RedditDataset()
        return reddit[0]
    elif dataset == 'arxiv':
        arxiv = DglNodePropPredDataset(name='ogbn-arxiv')
        return arxiv[0][0]
    elif dataset == 'proteins':
        protein = DglNodePropPredDataset(name='ogbn-proteins')
        return protein[0][0]
    else:
        raise KeyError("Unrecognized dataset name: {}".format(dataset))


if __name__ == '__main__':
    for dataset in ['reddit', 'arxiv', 'proteins']:
        g = get_graph(dataset)
        g = g.int()
        u, v = g.edges()
        with open('dataset/{}_coo.npy'.format(dataset), 'wb') as f:
            np.save(f, u.numpy())
            np.save(f, v.numpy())
        
        adj = g.adj(scipy_fmt='csr', transpose=True)

        with open('dataset/{}_csr.npy'.format(dataset), 'wb') as f:
            np.save(f, adj.indptr)
            np.save(f, adj.indices)
