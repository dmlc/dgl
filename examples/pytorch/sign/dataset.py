import torch
import numpy as np
import dgl


def load_dataset(name):
    dataset = name.lower()
    if dataset == "amazon":
        from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name="ogbn-products")
        splitted_idx = dataset.get_idx_split()
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        labels = labels.squeeze()
        n_classes = int(labels.max() - labels.min() + 1)
        features = g.ndata.pop("feat").float()
    elif dataset in ["reddit", "cora"]:
        if dataset == "reddit":
            from dgl.data import RedditDataset
            data = RedditDataset(self_loop=True)
            g = data.graph
        else:
            from dgl.data import CitationGraphDataset
            data = CitationGraphDataset('cora')
            g = dgl.DGLGraph(data.graph)
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        features = torch.Tensor(data.features)
        labels = torch.LongTensor(data.labels)
        n_classes = data.num_labels
        train_nid = torch.LongTensor(np.nonzero(train_mask)[0])
        val_nid = torch.LongTensor(np.nonzero(val_mask)[0])
        test_nid = torch.LongTensor(np.nonzero(test_mask)[0])
    else:
        print("Dataset {} is not supported".format(name))
        assert(0)

    return g, features, labels, n_classes, train_nid, val_nid, test_nid
