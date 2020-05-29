import torch
import numpy as np
import dgl


def load_dataset(name, print_stats=True):
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
        g = dgl.graph(g.all_edges())

    elif dataset in ["reddit", "cora"]:
        print_stats = False
        if dataset == "reddit":
            from dgl.data import RedditDataset
            data = RedditDataset(self_loop=True)
            graph = data.graph
        else:
            from dgl.data import CoraDataset
            data = CoraDataset()
            graph = dgl.DGLGraph(data.graph)
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        features = torch.Tensor(data.features)
        labels = torch.LongTensor(data.labels)
        n_classes = data.num_labels
        # Construct graph
        g = dgl.graph(graph.all_edges())
        train_nid = torch.LongTensor(np.nonzero(train_mask)[0])
        val_nid = torch.LongTensor(np.nonzero(val_mask)[0])
        test_nid = torch.LongTensor(np.nonzero(test_mask)[0])
    else:
        print("Dataset {} is not supported".format(name))

    if print_stats:
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        n_train_samples = train_nid.shape[0]
        n_val_samples = val_nid.shape[0]
        n_test_samples = test_nid.shape[0]

        print("""----Data statistics------'
          #Nodes %d
          #Edges %d
          #Classes %d
          #Train samples %d
          #Val samples %d
          #Test samples %d""" %
              (n_nodes, n_edges, n_classes,
                  n_train_samples,
                  n_val_samples,
                  n_test_samples))

    return g, features, labels, n_classes, train_nid, val_nid, test_nid
