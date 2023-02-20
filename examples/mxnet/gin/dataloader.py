"""
MxNet compatible dataloader
"""

import math

import dgl

import numpy as np
from mxnet import nd
from mxnet.gluon.data import DataLoader, Sampler
from sklearn.model_selection import StratifiedKFold


class SubsetRandomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(
            [self.indices[i] for i in np.random.permutation(len(self.indices))]
        )

    def __len__(self):
        return len(self.indices)


# default collate function
def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = nd.array(g.ndata[key])
        # no edge feats
    batched_graph = dgl.batch(graphs)
    labels = [nd.reshape(label, (1,)) for label in labels]
    labels = nd.concat(*labels, dim=0)
    return batched_graph, labels


class GraphDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        collate_fn=collate,
        seed=0,
        shuffle=True,
        split_name="fold10",
        fold_idx=0,
        split_ratio=0.7,
    ):
        self.shuffle = shuffle
        self.seed = seed

        labels = [l for _, l in dataset]

        if split_name == "fold10":
            train_idx, valid_idx = self._split_fold10(
                labels, fold_idx, seed, shuffle
            )
        elif split_name == "rand":
            train_idx, valid_idx = self._split_rand(
                labels, split_ratio, seed, shuffle
            )
        else:
            raise NotImplementedError()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            batchify_fn=collate_fn,
        )
        self.valid_loader = DataLoader(
            dataset,
            sampler=valid_sampler,
            batch_size=batch_size,
            batchify_fn=collate_fn,
        )

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        """10 flod"""
        assert 0 <= fold_idx and fold_idx < 10, print(
            "fold_idx must be from 0 to 9."
        )

        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(
            np.zeros(len(labels)), [label.asnumpy() for label in labels]
        ):  # split(x, y)
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        print("train_set : test_set = %d : %d", len(train_idx), len(valid_idx))

        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.seed(seed)
        np.random.shuffle(indices)
        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print("train_set : test_set = %d : %d", len(train_idx), len(valid_idx))

        return train_idx, valid_idx
