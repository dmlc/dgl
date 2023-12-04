import dgl
import numpy as np
import torch


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
        n_classes = int(labels.max() - labels.min() + 1)
        g.ndata["label"] = labels.squeeze()
        g.ndata["feat"] = g.ndata["feat"].float()
    elif dataset in ["reddit", "cora"]:
        if dataset == "reddit":
            from dgl.data import RedditDataset

            data = RedditDataset(self_loop=True)
            g = data[0]
        else:
            from dgl.data import CitationGraphDataset

            data = CitationGraphDataset("cora")
            g = data[0]
        n_classes = data.num_classes
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]
        train_nid = torch.LongTensor(train_mask.nonzero().squeeze())
        val_nid = torch.LongTensor(val_mask.nonzero().squeeze())
        test_nid = torch.LongTensor(test_mask.nonzero().squeeze())
    else:
        print("Dataset {} is not supported".format(name))
        assert 0

    return g, n_classes, train_nid, val_nid, test_nid
