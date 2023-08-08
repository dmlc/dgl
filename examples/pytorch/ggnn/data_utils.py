"""
Data utils for processing bAbI datasets
"""

import os
import string

import dgl

import torch
from dgl.data.utils import (
    _get_dgl_url,
    download,
    extract_archive,
    get_download_dir,
)
from torch.utils.data import DataLoader


def get_babi_dataloaders(batch_size, train_size=50, task_id=4, q_type=0):
    _download_babi_data()

    node_dict = dict(
        zip(list(string.ascii_uppercase), range(len(string.ascii_uppercase)))
    )

    if task_id == 4:
        edge_dict = {"n": 0, "s": 1, "w": 2, "e": 3}
        reverse_edge = {}
        return _ns_dataloader(
            train_size,
            q_type,
            batch_size,
            node_dict,
            edge_dict,
            reverse_edge,
            "04",
        )
    elif task_id == 15:
        edge_dict = {"is": 0, "has_fear": 1}
        reverse_edge = {}
        return _ns_dataloader(
            train_size,
            q_type,
            batch_size,
            node_dict,
            edge_dict,
            reverse_edge,
            "15",
        )
    elif task_id == 16:
        edge_dict = {"is": 0, "has_color": 1}
        reverse_edge = {0: 0}
        return _ns_dataloader(
            train_size,
            q_type,
            batch_size,
            node_dict,
            edge_dict,
            reverse_edge,
            "16",
        )
    elif task_id == 18:
        edge_dict = {">": 0, "<": 1}
        label_dict = {"false": 0, "true": 1}
        reverse_edge = {0: 1, 1: 0}
        return _gc_dataloader(
            train_size,
            q_type,
            batch_size,
            node_dict,
            edge_dict,
            label_dict,
            reverse_edge,
            "18",
        )
    elif task_id == 19:
        edge_dict = {"n": 0, "s": 1, "w": 2, "e": 3, "<end>": 4}
        reverse_edge = {0: 1, 1: 0, 2: 3, 3: 2}
        max_seq_length = 2
        return _path_finding_dataloader(
            train_size,
            batch_size,
            node_dict,
            edge_dict,
            reverse_edge,
            "19",
            max_seq_length,
        )


def _ns_dataloader(
    train_size, q_type, batch_size, node_dict, edge_dict, reverse_edge, path
):
    def _collate_fn(batch):
        graphs = []
        labels = []
        for d in batch:
            edges = d["edges"]

            node_ids = []
            for s, e, t in edges:
                if s not in node_ids:
                    node_ids.append(s)
                if t not in node_ids:
                    node_ids.append(t)
            g = dgl.graph([])
            g.add_nodes(len(node_ids))
            g.ndata["node_id"] = torch.tensor(node_ids, dtype=torch.long)

            nid2idx = dict(zip(node_ids, list(range(len(node_ids)))))

            # convert label to node index
            label = d["eval"][2]
            label_idx = nid2idx[label]
            labels.append(label_idx)

            edge_types = []
            for s, e, t in edges:
                g.add_edges(nid2idx[s], nid2idx[t])
                edge_types.append(e)
                if e in reverse_edge:
                    g.add_edges(nid2idx[t], nid2idx[s])
                    edge_types.append(reverse_edge[e])
            g.edata["type"] = torch.tensor(edge_types, dtype=torch.long)
            annotation = torch.zeros(len(node_ids), dtype=torch.long)
            annotation[nid2idx[d["eval"][0]]] = 1
            g.ndata["annotation"] = annotation.unsqueeze(-1)
            graphs.append(g)
        batch_graph = dgl.batch(graphs)
        labels = torch.tensor(labels, dtype=torch.long)
        return batch_graph, labels

    def _get_dataloader(data, shuffle):
        return DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_fn,
        )

    train_set, dev_set, test_sets = _convert_ns_dataset(
        train_size, node_dict, edge_dict, path, q_type
    )
    train_dataloader = _get_dataloader(train_set, True)
    dev_dataloader = _get_dataloader(dev_set, False)
    test_dataloaders = []
    for d in test_sets:
        dl = _get_dataloader(d, False)
        test_dataloaders.append(dl)

    return train_dataloader, dev_dataloader, test_dataloaders


def _convert_ns_dataset(train_size, node_dict, edge_dict, path, q_type):
    total_num = 11000

    def convert(file):
        dataset = []
        d = dict()
        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split()
                if line[0] == "1" and len(d) > 0:
                    d = dict()
                if line[1] == "eval":
                    # (src, edge, label)
                    d["eval"] = (
                        node_dict[line[2]],
                        edge_dict[line[3]],
                        node_dict[line[4]],
                    )
                    if d["eval"][1] == q_type:
                        dataset.append(d)
                        if len(dataset) >= total_num:
                            break
                else:
                    if "edges" not in d:
                        d["edges"] = []
                    d["edges"].append(
                        (
                            node_dict[line[1]],
                            edge_dict[line[2]],
                            node_dict[line[3]],
                        )
                    )
        return dataset

    download_dir = get_download_dir()
    filename = os.path.join(download_dir, "babi_data", path, "data.txt")
    data = convert(filename)

    assert len(data) == total_num

    train_set = data[:train_size]
    dev_set = data[950:1000]
    test_sets = []
    for i in range(10):
        test = data[1000 * (i + 1) : 1000 * (i + 2)]
        test_sets.append(test)

    return train_set, dev_set, test_sets


def _gc_dataloader(
    train_size,
    q_type,
    batch_size,
    node_dict,
    edge_dict,
    label_dict,
    reverse_edge,
    path,
):
    def _collate_fn(batch):
        graphs = []
        labels = []
        for d in batch:
            edges = d["edges"]

            node_ids = []
            for s, e, t in edges:
                if s not in node_ids:
                    node_ids.append(s)
                if t not in node_ids:
                    node_ids.append(t)
            g = dgl.graph([])
            g.add_nodes(len(node_ids))
            g.ndata["node_id"] = torch.tensor(node_ids, dtype=torch.long)

            nid2idx = dict(zip(node_ids, list(range(len(node_ids)))))

            labels.append(d["eval"][-1])

            edge_types = []
            for s, e, t in edges:
                g.add_edges(nid2idx[s], nid2idx[t])
                edge_types.append(e)
                if e in reverse_edge:
                    g.add_edges(nid2idx[t], nid2idx[s])
                    edge_types.append(reverse_edge[e])
            g.edata["type"] = torch.tensor(edge_types, dtype=torch.long)
            annotation = torch.zeros([len(node_ids), 2], dtype=torch.long)
            annotation[nid2idx[d["eval"][0]]][0] = 1
            annotation[nid2idx[d["eval"][2]]][1] = 1
            g.ndata["annotation"] = annotation
            graphs.append(g)
        batch_graph = dgl.batch(graphs)
        labels = torch.tensor(labels, dtype=torch.long)
        return batch_graph, labels

    def _get_dataloader(data, shuffle):
        return DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_fn,
        )

    train_set, dev_set, test_sets = _convert_gc_dataset(
        train_size, node_dict, edge_dict, label_dict, path, q_type
    )
    train_dataloader = _get_dataloader(train_set, True)
    dev_dataloader = _get_dataloader(dev_set, False)
    test_dataloaders = []
    for d in test_sets:
        dl = _get_dataloader(d, False)
        test_dataloaders.append(dl)

    return train_dataloader, dev_dataloader, test_dataloaders


def _convert_gc_dataset(
    train_size, node_dict, edge_dict, label_dict, path, q_type
):
    total_num = 11000

    def convert(file):
        dataset = []
        d = dict()
        with open(file, "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split()
                if line[0] == "1" and len(d) > 0:
                    d = dict()
                if line[1] == "eval":
                    # (src, edge, label)
                    if "eval" not in d:
                        d["eval"] = (
                            node_dict[line[2]],
                            edge_dict[line[3]],
                            node_dict[line[4]],
                            label_dict[line[5]],
                        )
                        if d["eval"][1] == q_type:
                            dataset.append(d)
                            if len(dataset) >= total_num:
                                break
                else:
                    if "edges" not in d:
                        d["edges"] = []
                    d["edges"].append(
                        (
                            node_dict[line[1]],
                            edge_dict[line[2]],
                            node_dict[line[3]],
                        )
                    )
        return dataset

    download_dir = get_download_dir()
    filename = os.path.join(download_dir, "babi_data", path, "data.txt")
    data = convert(filename)

    assert len(data) == total_num

    train_set = data[:train_size]
    dev_set = data[950:1000]
    test_sets = []
    for i in range(10):
        test = data[1000 * (i + 1) : 1000 * (i + 2)]
        test_sets.append(test)

    return train_set, dev_set, test_sets


def _path_finding_dataloader(
    train_size,
    batch_size,
    node_dict,
    edge_dict,
    reverse_edge,
    path,
    max_seq_length,
):
    def _collate_fn(batch):
        graphs = []
        ground_truths = []
        seq_lengths = []
        for d in batch:
            edges = d["edges"]

            node_ids = []
            for s, e, t in edges:
                if s not in node_ids:
                    node_ids.append(s)
                if t not in node_ids:
                    node_ids.append(t)
            g = dgl.graph([])
            g.add_nodes(len(node_ids))
            g.ndata["node_id"] = torch.tensor(node_ids, dtype=torch.long)

            nid2idx = dict(zip(node_ids, list(range(len(node_ids)))))

            truth = d["seq_out"] + [edge_dict["<end>"]] * (
                max_seq_length - len(d["seq_out"])
            )
            seq_len = len(d["seq_out"])
            ground_truths.append(truth)
            seq_lengths.append(seq_len)

            edge_types = []
            for s, e, t in edges:
                g.add_edges(nid2idx[s], nid2idx[t])
                edge_types.append(e)
                if e in reverse_edge:
                    g.add_edges(nid2idx[t], nid2idx[s])
                    edge_types.append(reverse_edge[e])
            g.edata["type"] = torch.tensor(edge_types, dtype=torch.long)
            annotation = torch.zeros([len(node_ids), 2], dtype=torch.long)
            annotation[nid2idx[d["eval"][0]]][0] = 1
            annotation[nid2idx[d["eval"][1]]][1] = 1
            g.ndata["annotation"] = annotation
            graphs.append(g)
        batch_graph = dgl.batch(graphs)
        ground_truths = torch.tensor(ground_truths, dtype=torch.long)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        return batch_graph, ground_truths, seq_lengths

    def _get_dataloader(data, shuffle):
        return DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_fn,
        )

    train_set, dev_set, test_sets = _convert_path_finding(
        train_size, node_dict, edge_dict, path
    )
    train_dataloader = _get_dataloader(train_set, True)
    dev_dataloader = _get_dataloader(dev_set, False)
    test_dataloaders = []
    for d in test_sets:
        dl = _get_dataloader(d, False)
        test_dataloaders.append(dl)

    return train_dataloader, dev_dataloader, test_dataloaders


def _convert_path_finding(train_size, node_dict, edge_dict, path):
    total_num = 11000

    def convert(file):
        dataset = []
        d = dict()
        with open(file, "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                if line[0] == "1" and len(d) > 0:
                    d = dict()
                if line[1] == "eval":
                    # (src, edge, label)
                    d["eval"] = (node_dict[line[3]], node_dict[line[4]])
                    d["seq_out"] = []
                    seq_out = line[5].split(",")
                    for e in seq_out:
                        d["seq_out"].append(edge_dict[e])
                    dataset.append(d)
                    if len(dataset) >= total_num:
                        break
                else:
                    if "edges" not in d:
                        d["edges"] = []
                    d["edges"].append(
                        (
                            node_dict[line[1]],
                            edge_dict[line[2]],
                            node_dict[line[3]],
                        )
                    )
        return dataset

    download_dir = get_download_dir()
    filename = os.path.join(download_dir, "babi_data", path, "data.txt")
    data = convert(filename)

    assert len(data) == total_num

    train_set = data[:train_size]
    dev_set = data[950:1000]
    test_sets = []
    for i in range(10):
        test = data[1000 * (i + 1) : 1000 * (i + 2)]
        test_sets.append(test)

    return train_set, dev_set, test_sets


def _download_babi_data():
    download_dir = get_download_dir()
    zip_file_path = os.path.join(download_dir, "babi_data.zip")

    data_url = _get_dgl_url("models/ggnn_babi_data.zip")
    download(data_url, path=zip_file_path)

    extract_dir = os.path.join(download_dir, "babi_data")
    if not os.path.exists(extract_dir):
        extract_archive(zip_file_path, extract_dir)
