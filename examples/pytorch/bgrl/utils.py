import copy

import numpy as np
import torch

from dgl.data import (
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorCSDataset,
    CoauthorPhysicsDataset,
    PPIDataset,
    WikiCSDataset,
)
from dgl.dataloading import GraphDataLoader
from dgl.transforms import Compose, DropEdge, FeatMask, RowFeatNormalizer


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return (
                self.max_val
                * (
                    1
                    + np.cos(
                        (step - self.warmup_steps)
                        * np.pi
                        / (self.total_steps - self.warmup_steps)
                    )
                )
                / 2
            )
        else:
            raise ValueError(
                "Step ({}) > total number of steps ({}).".format(
                    step, self.total_steps
                )
            )


def get_graph_drop_transform(drop_edge_p, feat_mask_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.0:
        transforms.append(DropEdge(drop_edge_p))

    # drop features
    if feat_mask_p > 0.0:
        transforms.append(FeatMask(feat_mask_p, node_feat_names=["feat"]))

    return Compose(transforms)


def get_wiki_cs(transform=RowFeatNormalizer(subtract_min=True)):
    dataset = WikiCSDataset(transform=transform)
    g = dataset[0]
    std, mean = torch.std_mean(g.ndata["feat"], dim=0, unbiased=False)
    g.ndata["feat"] = (g.ndata["feat"] - mean) / std

    return [g]


def get_ppi():
    train_dataset = PPIDataset(mode="train")
    val_dataset = PPIDataset(mode="valid")
    test_dataset = PPIDataset(mode="test")
    train_val_dataset = [i for i in train_dataset] + [i for i in val_dataset]
    for idx, data in enumerate(train_val_dataset):
        data.ndata["batch"] = torch.zeros(data.num_nodes()) + idx
        data.ndata["batch"] = data.ndata["batch"].long()

    g = list(GraphDataLoader(train_val_dataset, batch_size=22, shuffle=True))

    return g, PPIDataset(mode="train"), PPIDataset(mode="valid"), test_dataset


def get_dataset(name, transform=RowFeatNormalizer(subtract_min=True)):
    dgl_dataset_dict = {
        "coauthor_cs": CoauthorCSDataset,
        "coauthor_physics": CoauthorPhysicsDataset,
        "amazon_computers": AmazonCoBuyComputerDataset,
        "amazon_photos": AmazonCoBuyPhotoDataset,
        "wiki_cs": get_wiki_cs,
        "ppi": get_ppi,
    }

    dataset_class = dgl_dataset_dict[name]
    train_data, val_data, test_data = None, None, None
    if name != "ppi":
        dataset = dataset_class(transform=transform)
    else:
        dataset, train_data, val_data, test_data = dataset_class()

    return dataset, train_data, val_data, test_data
