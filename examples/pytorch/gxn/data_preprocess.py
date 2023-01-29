import json
import logging
import os
import sys

import numpy as np
import torch

from dgl.data import LegacyTUDataset


def _load_check_mark(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {}


def _save_check_mark(path: str, marks: dict):
    with open(path, "w") as f:
        json.dump(marks, f)


def node_label_as_feature(dataset: LegacyTUDataset, mode="concat", save=True):
    """
    Description
    -----------
    Add node labels to graph node features dict

    Parameters
    ----------
    dataset : LegacyTUDataset
        The dataset object
    concat : str, optional
        How to add node label to the graph. Valid options are "add",
        "replace" and "concat".
        - "add": Directly add node_label to graph node feature dict.
        - "concat": Concatenate "feat" and "node_label"
        - "replace": Use "node_label" as "feat"
        Default: :obj:`"concat"`
    save : bool, optional
        Save the result dataset.
        Default: :obj:`True`
    """
    # check if node label is not available
    if (
        not os.path.exists(dataset._file_path("node_labels"))
        or len(dataset) == 0
    ):
        logging.warning("No Node Label Data")
        return dataset

    # check if has cached value
    check_mark_name = "node_label_as_feature"
    check_mark_path = os.path.join(
        dataset.save_path, "info_{}_{}.json".format(dataset.name, dataset.hash)
    )
    check_mark = _load_check_mark(check_mark_path)
    if (
        check_mark_name in check_mark
        and check_mark[check_mark_name]
        and not dataset._force_reload
    ):
        logging.warning("Using cached value in node_label_as_feature")
        return dataset
    logging.warning(
        "Adding node labels into node features..., mode={}".format(mode)
    )

    # check if graph has "feat"
    if "feat" not in dataset[0][0].ndata:
        logging.warning("Dataset has no node feature 'feat'")
        if mode.lower() == "concat":
            mode = "replace"

    # first read node labels
    DS_node_labels = dataset._idx_from_zero(
        np.loadtxt(dataset._file_path("node_labels"), dtype=int)
    )
    one_hot_node_labels = dataset._to_onehot(DS_node_labels)

    # read graph idx
    DS_indicator = dataset._idx_from_zero(
        np.genfromtxt(dataset._file_path("graph_indicator"), dtype=int)
    )
    node_idx_list = []
    for idx in range(np.max(DS_indicator) + 1):
        node_idx = np.where(DS_indicator == idx)
        node_idx_list.append(node_idx[0])

    # add to node feature dict
    for idx, g in zip(node_idx_list, dataset.graph_lists):
        node_labels_tensor = torch.tensor(one_hot_node_labels[idx, :])
        if mode.lower() == "concat":
            g.ndata["feat"] = torch.cat(
                (g.ndata["feat"], node_labels_tensor), dim=1
            )
        elif mode.lower() == "add":
            g.ndata["node_label"] = node_labels_tensor
        else:  # replace
            g.ndata["feat"] = node_labels_tensor

    if save:
        check_mark[check_mark_name] = True
        _save_check_mark(check_mark_path, check_mark)
        dataset.save()
    return dataset


def degree_as_feature(dataset: LegacyTUDataset, save=True):
    """
    Description
    -----------
    Use node degree (in one-hot format) as node feature

    Parameters
    ----------
    dataset : LegacyTUDataset
        The dataset object

    save : bool, optional
        Save the result dataset.
        Default: :obj:`True`
    """
    # first check if already have such feature
    check_mark_name = "degree_as_feat"
    feat_name = "feat"
    check_mark_path = os.path.join(
        dataset.save_path, "info_{}_{}.json".format(dataset.name, dataset.hash)
    )
    check_mark = _load_check_mark(check_mark_path)

    if (
        check_mark_name in check_mark
        and check_mark[check_mark_name]
        and not dataset._force_reload
    ):
        logging.warning("Using cached value in 'degree_as_feature'")
        return dataset

    logging.warning("Adding node degree into node features...")
    min_degree = sys.maxsize
    max_degree = 0
    for i in range(len(dataset)):
        degrees = dataset.graph_lists[i].in_degrees()
        min_degree = min(min_degree, degrees.min().item())
        max_degree = max(max_degree, degrees.max().item())

    vec_len = max_degree - min_degree + 1
    for i in range(len(dataset)):
        num_nodes = dataset.graph_lists[i].num_nodes()
        node_feat = torch.zeros((num_nodes, vec_len))
        degrees = dataset.graph_lists[i].in_degrees()
        node_feat[torch.arange(num_nodes), degrees - min_degree] = 1.0
        dataset.graph_lists[i].ndata[feat_name] = node_feat

    if save:
        check_mark[check_mark_name] = True
        dataset.save()
        _save_check_mark(check_mark_path, check_mark)
    return dataset
