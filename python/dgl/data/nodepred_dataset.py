import numpy as np
import torch as th


def idx_to_mask(idx, max_num):
    z = th.zeros(max_num, dtype=th.bool)
    z[idx] = 1
    return z


class NodepredDataset:
    def __init__(self, dataset, split_ratio=(0.8, 0.1, 0.1), seed=42):
        self.dataset = dataset
        self.graph = dataset[0]
        self.num_nodes = self.graph.num_nodes()

        self.graph = self.graph.add_self_loop()

        if "label" not in self.graph.ndata:
            assert "labels" in self.graph.ndata, "Dataset doesn't include label information"
            self.graph.ndata["label"] = self.graph.ndata["labels"]

        if "feat" not in self.graph.ndata:
            assert 'feature' in self.graph.ndata, "Only found {}".format(
                list(self.graph.ndata.keys()))
            self.graph.ndata["feat"] = self.graph.ndata["feature"]

        assert len(
            self.graph.ndata["feat"].shape) == 2, "Only support 2-dim features in graph"

        if "train_mask" not in self.graph.ndata or "val_mask" not in self.graph.ndata or "test_mask" not in self.graph.ndata:
            print("Original dataset didn't provide split information between train/val/test.\
                 Generate it based on the input split ratio")
            train_idx, val_idx, test_idx = self.split_nodes(
                self.num_nodes, split_ratio, seed)
            self.graph.ndata["train_mask"] = idx_to_mask(
                train_idx, self.num_nodes)
            self.graph.ndata["val_mask"] = idx_to_mask(val_idx, self.num_nodes)
            self.graph.ndata["test_mask"] = idx_to_mask(
                test_idx, self.num_nodes)

    def __getitem__(self, idx):
        if idx == 0:
            return self.graph
        else:
            raise NotImplemented("Only support index 0")

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @staticmethod
    def split_nodes(num_nodes, split_ratio, seed):
        rng = np.random.default_rng(seed)
        node_idx = rng.permutation(np.arange(num_nodes))
        split_ratio = np.array(
            split_ratio) / np.sum(split_ratio)
        train_size, val_size = int(
            num_nodes*split_ratio[0]), int(num_nodes*split_ratio[1])
        train_idx, val_idx, test_idx = np.split(
            node_idx, [train_size, train_size+val_size])
        return train_idx, val_idx, test_idx
