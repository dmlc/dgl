import os
from functools import namedtuple

import dgl
import numpy as np
import torch
from dgl.data import PPIDataset
from dgl.data import load_data as _load_data
from sklearn.metrics import f1_score

class Logger(object):
    '''A custom logger to log stdout to a logging file.'''
    def __init__(self, path):
        """Initialize the logger.

        Paramters
        ---------
        path : str
            The file path to be stored in.
        """
        self.path = path

    def write(self, s):
        with open(self.path, 'a') as f:
            f.write(str(s))
        print(s)
        return

def arg_list(labels):
    hist, indexes, inverse, counts = np.unique(
        labels, return_index=True, return_counts=True, return_inverse=True)
    li = []
    for h in hist:
        li.append(np.argwhere(inverse == h))
    return li

def save_log_dir(args):
    log_dir = './log/{}/{}'.format(args.dataset, args.note)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def calc_f1(y_true, y_pred, multitask):
    if multitask:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
        f1_score(y_true, y_pred, average="macro")

def evaluate(model, g, labels, mask, multitask=False):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        f1_mic, f1_mac = calc_f1(labels.cpu().numpy(),
                                 logits.cpu().numpy(), multitask)
        return f1_mic, f1_mac

def load_data(args):
    '''Wraps the dgl's load_data utility to handle ppi special case'''
    if args.dataset != 'ppi':
        return _load_data(args)
    train_dataset = PPIDataset('train')
    val_dataset = PPIDataset('valid')
    test_dataset = PPIDataset('test')
    PPIDataType = namedtuple('PPIDataset', ['train_mask', 'test_mask',
                                            'val_mask', 'features', 'labels', 'num_labels', 'graph'])
    G = dgl.BatchedDGLGraph(
        [train_dataset.graph, val_dataset.graph, test_dataset.graph], edge_attrs=None, node_attrs=None)
    G = G.to_networkx()
    # hack to dodge the potential bugs of to_networkx
    for (n1, n2, d) in G.edges(data=True):
        d.clear()
    train_nodes_num = train_dataset.graph.number_of_nodes()
    test_nodes_num = test_dataset.graph.number_of_nodes()
    val_nodes_num = val_dataset.graph.number_of_nodes()
    nodes_num = G.number_of_nodes()
    assert(nodes_num == (train_nodes_num + test_nodes_num + val_nodes_num))
    # construct mask
    mask = np.zeros((nodes_num,), dtype=bool)
    train_mask = mask.copy()
    train_mask[:train_nodes_num] = True
    val_mask = mask.copy()
    val_mask[train_nodes_num:-test_nodes_num] = True
    test_mask = mask.copy()
    test_mask[-test_nodes_num:] = True

    # construct features
    features = np.concatenate(
        [train_dataset.features, val_dataset.features, test_dataset.features], axis=0)

    labels = np.concatenate(
        [train_dataset.labels, val_dataset.labels, test_dataset.labels], axis=0)

    data = PPIDataType(graph=G, train_mask=train_mask, test_mask=test_mask,
                       val_mask=val_mask, features=features, labels=labels, num_labels=121)
    return data
