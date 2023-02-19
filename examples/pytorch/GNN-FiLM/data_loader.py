import collections

import dgl
from dgl.data import PPIDataset

from torch.utils.data import DataLoader, Dataset

# implement the collate_fn for dgl graph data class
PPIBatch = collections.namedtuple("PPIBatch", ["graph", "label"])


def batcher(device):
    def batcher_dev(batch):
        batch_graphs = dgl.batch(batch)
        return PPIBatch(
            graph=batch_graphs, label=batch_graphs.ndata["label"].to(device)
        )

    return batcher_dev


# add a fresh "self-loop" edge type to the untyped PPI dataset and prepare train, val, test loaders
def load_PPI(batch_size=1, device="cpu"):
    train_set = PPIDataset(mode="train")
    valid_set = PPIDataset(mode="valid")
    test_set = PPIDataset(mode="test")
    # for each graph, add self-loops as a new relation type
    # here we reconstruct the graph since the schema of a heterograph cannot be changed once constructed
    for i in range(len(train_set)):
        g = dgl.heterograph(
            {
                ("_N", "_E", "_N"): train_set[i].edges(),
                ("_N", "self", "_N"): (
                    train_set[i].nodes(),
                    train_set[i].nodes(),
                ),
            }
        )
        g.ndata["label"] = train_set[i].ndata["label"]
        g.ndata["feat"] = train_set[i].ndata["feat"]
        g.ndata["_ID"] = train_set[i].ndata["_ID"]
        g.edges["_E"].data["_ID"] = train_set[i].edata["_ID"]
        train_set.graphs[i] = g
    for i in range(len(valid_set)):
        g = dgl.heterograph(
            {
                ("_N", "_E", "_N"): valid_set[i].edges(),
                ("_N", "self", "_N"): (
                    valid_set[i].nodes(),
                    valid_set[i].nodes(),
                ),
            }
        )
        g.ndata["label"] = valid_set[i].ndata["label"]
        g.ndata["feat"] = valid_set[i].ndata["feat"]
        g.ndata["_ID"] = valid_set[i].ndata["_ID"]
        g.edges["_E"].data["_ID"] = valid_set[i].edata["_ID"]
        valid_set.graphs[i] = g
    for i in range(len(test_set)):
        g = dgl.heterograph(
            {
                ("_N", "_E", "_N"): test_set[i].edges(),
                ("_N", "self", "_N"): (
                    test_set[i].nodes(),
                    test_set[i].nodes(),
                ),
            }
        )
        g.ndata["label"] = test_set[i].ndata["label"]
        g.ndata["feat"] = test_set[i].ndata["feat"]
        g.ndata["_ID"] = test_set[i].ndata["_ID"]
        g.edges["_E"].data["_ID"] = test_set[i].edata["_ID"]
        test_set.graphs[i] = g

    etypes = train_set[0].etypes
    in_size = train_set[0].ndata["feat"].shape[1]
    out_size = train_set[0].ndata["label"].shape[1]

    # prepare train, valid, and test dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=batcher(device),
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        collate_fn=batcher(device),
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=batcher(device),
        shuffle=True,
    )
    return train_loader, valid_loader, test_loader, etypes, in_size, out_size
