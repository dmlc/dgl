"""`UniGNN <https://arxiv.org/abs/2105.00956>`_

This script reproduces the result from the
`official repository <https://github.com/OneForward/UniGNN>`_
with the following command:

.. code::

   python train.py --data=cocitation --dataset=cora --model-name=UniGAT
"""
import pickle
import os
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import dgl
import dgl.data.utils
import dgl.nn
import dgl.function as fn
import numpy as np


@contextlib.contextmanager
def changedir(path):
    try:
        cwd = os.getcwd()
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


class HyperGCNCocitationCora(dgl.data.DGLDataset):
    """Cora cocitation dataset from
    `HyperGCN paper <https://github.com/malllabiisc/HyperGCN>`_.

    It contains a single bipartite graph, which is the star-expansion of
    the hypergraph.  The node type ``node`` represents the nodes in the
    original hypergraph, and the node type ``hyperedge`` represents the
    hyperedges in the original hypergraph.  The edges between ``node``
    and ``hyperedge`` describes whether a hyperedge contains a node
    in the original hypergraph.
    """

    dataset_name = "HyperGCNCocitationCora"

    def __init__(self):
        super().__init__(
            self.dataset_name,
            url="https://raw.githubusercontent.com/malllabiisc/HyperGCN/master/data/cocitation/cora/",
            raw_dir=os.path.join(
                dgl.data.utils.get_download_dir(), self.dataset_name
            ),
        )

    def has_cache(self):
        return True

    def download(self):
        for fname in [
            "hypergraph.pickle",
            "features.pickle",
            "labels.pickle",
        ] + [f"splits/{i}.pickle" for i in range(1, 11)]:
            dgl.data.utils.download(
                self.url + fname, os.path.join(self.raw_dir, fname)
            )

    def process(self):
        splits = []
        with changedir(self.raw_dir):
            with open("hypergraph.pickle", "rb") as f:
                hypergraph_dict = pickle.load(f)
            with open("features.pickle", "rb") as f:
                features = pickle.load(f)
            with open("labels.pickle", "rb") as f:
                labels = pickle.load(f)
            for i in range(1, 11):
                with open(os.path.join("splits", f"{i}.pickle"), "rb") as f:
                    splits.append(pickle.load(f))

        srcs = []
        dsts = []
        for hyperedge_id, nodes in hypergraph_dict.items():
            srcs.append(torch.full((len(nodes),), hyperedge_id))
            dsts.append(torch.LongTensor(list(nodes)))
        srcs = torch.cat(srcs)
        dsts = torch.cat(dsts)
        g = dgl.heterograph(
            {
                ("hyperedge", "contains", "node"): (srcs, dsts),
                ("node", "belongs-to", "hyperedge"): (dsts, srcs),
            }
        )

        self._graph = g
        features = torch.FloatTensor(features.todense())
        labels = torch.LongTensor(labels)
        self._splits = splits
        self._num_classes = max(labels) + 1
        self._num_features = features.shape[1]

        self._graph.nodes["node"].data["feat"] = features
        self._graph.nodes["node"].data["label"] = labels

        for split in self._splits:
            # HyperGCN data does not come with validation sets, but UniGNN
            # takes 20% of the test indices as validation indices.  Here
            # we follow the same practice.
            num_test_idx = len(split["test"])
            num_val_idx = int(num_test_idx * 0.2)

            # We use the same generator to avoid having different splits
            # on different runs and different machines.
            generator = np.random.Generator(np.random.MT19937(1))
            generator.shuffle(split["test"])
            split["val"] = split["test"][:num_val_idx]
            split["test"] = split["test"][num_val_idx:]

    def save(self):
        with changedir(self.save_dir):
            dgl.save_graphs(self.dataset_name + "_graph.bin", self._graph)
            dgl.data.utils.save_info(
                self.dataset_name + "_info.pkl",
                {
                    "num_classes": self._num_classes,
                    "num_features": self._num_features,
                    "splits": self._splits,
                },
            )

    def load(self):
        with changedir(self.save_dir):
            graphs, _ = dgl.load_graphs(self.dataset_name + "_graph.bin")
            info = dgl.data.utils.load_info(self.dataset_name + "_info.pkl")
            graph = graphs[0]

            self._graph = graph
            self._num_classes = info["num_classes"]
            self._num_features = info["num_features"]
            self._splits = info["splits"]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_features(self):
        return self._num_features

    # UniGNN uses all 10 splits.
    # Question: how do we represent multiple splits in DGL dataset?
    def get_split(self, i):
        """Returns the i-th split.

        It returns a triplet of training, validation, and test indices.
        """
        split = self._splits[i]
        train_idx = torch.LongTensor(split["train"])
        val_idx = torch.LongTensor(split["val"])
        test_idx = torch.LongTensor(split["test"])
        return train_idx, val_idx, test_idx

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return self._graph


class UniGATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads):
        super().__init__()

        self.W = nn.Linear(in_size, out_size * num_heads, bias=False)
        self.att_e = nn.Parameter(torch.randn(1, num_heads, out_size))
        self.dropout = nn.Dropout(0.6)
        self.num_heads = num_heads
        self.out_size = out_size
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_e)

    def forward(self, g, x):
        """Compute forward propagation based on the star-expanded hypergraph
        :attr:`g` and node features :attr:`x`.
        """
        with g.local_scope():
            x = self.W(x).view(-1, self.num_heads, self.out_size)
            g.nodes["node"].data["x"] = x
            g.update_all(
                fn.copy_u("x", "m"), fn.mean("m", "h"), etype="belongs-to"
            )
            # g.nodes['hyperedge'].data['h'] has dimensionality
            # (num_hyperedges, num_heads, out_size)
            alpha = g.nodes["hyperedge"].data["h"] * self.att_e
            alpha = alpha.sum(-1)
            # alpha: (num_hyperedges, num_heads, 1)

            alpha = F.leaky_relu(alpha, 0.2)
            g.nodes["hyperedge"].data["alpha"] = alpha
            g.apply_edges(fn.copy_u("alpha", "alpha"), etype="contains")
            alpha = dgl.nn.edge_softmax(
                g["contains"],
                g.edges["contains"].data["alpha"],
            )
            g.edges["contains"].data["alpha"] = self.dropout(alpha)
            g.update_all(
                fn.u_mul_e("h", "alpha", "m"),
                fn.sum("m", "h"),
                etype="contains",
            )
            return (
                g.nodes["node"]
                .data["h"]
                .view(-1, self.num_heads * self.out_size)
            )


class UniGAT(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv1 = UniGATConv(in_size, 8, 8)
        self.conv2 = UniGATConv(64, out_size, 1)
        self.dropout = nn.Dropout(0.6)

    def forward(self, g, x):
        x = self.dropout(x)
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x.squeeze(-1)


def train(graph, model, optimizer, train_idx):
    model.train()
    x = graph.nodes["node"].data["feat"]
    y = graph.nodes["node"].data["label"]
    y_hat = model(graph, x)
    loss = F.cross_entropy(y_hat[train_idx], y[train_idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(graph, model, val_idx, test_idx):
    model.eval()
    x = graph.nodes["node"].data["feat"]
    y = graph.nodes["node"].data["label"]
    y_hat = model(graph, x)
    _, y_pred = y_hat.max(1)
    val_acc = (y_pred == y)[val_idx].float().mean()
    test_acc = (y_pred == y)[test_idx].float().mean()
    return val_acc, test_acc


def main():
    dataset = HyperGCNCocitationCora()
    graph = dataset[0]
    # normalize features
    feat = graph.nodes["node"].data["feat"]
    feat = feat / feat.sum(1, keepdim=True)
    graph.nodes["node"].data["feat"] = feat

    test_accs = []
    for split in range(10):
        model = UniGAT(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4
        )
        train_idx, val_idx, test_idx = dataset.get_split(split)
        with tqdm.trange(200) as tq:
            for _ in tq:
                train_loss = train(graph, model, optimizer, train_idx)
                val_acc, test_acc = evaluate(graph, model, val_idx, test_idx)
                tq.set_postfix(
                    {
                        "Val Acc": f"{val_acc:.5f}",
                        "Test Acc": f"{test_acc:.5f}",
                    },
                    refresh=False,
                )
        test_accs.append(test_acc)
    print("Average test acc:", np.mean(test_accs))


if __name__ == "__main__":
    main()
