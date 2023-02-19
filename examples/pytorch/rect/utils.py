import dgl
import torch
from dgl.data import CiteseerGraphDataset, CoraGraphDataset


def load_data(args):
    if args.dataset == "cora":
        data = CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    g = dgl.add_self_loop(g)
    return g, features, labels, train_mask, test_mask, data.num_classes, cuda


def svd_feature(features, d=200):
    """Get 200-dimensional node features, to avoid curse of dimensionality"""
    if features.shape[1] <= d:
        return features
    U, S, VT = torch.svd(features)
    res = torch.mm(U[:, 0:d], torch.diag(S[0:d]))
    return res


def process_classids(labels_temp):
    """Reorder the remaining classes with unseen classes removed.
    Input: the label only removing unseen classes
    Output: the label with reordered classes
    """
    labeldict = {}
    num = 0
    for i in labels_temp:
        labeldict[int(i)] = 1
    labellist = sorted(labeldict)
    for label in labellist:
        labeldict[int(label)] = num
        num = num + 1
    for i in range(labels_temp.numel()):
        labels_temp[i] = labeldict[int(labels_temp[i])]
    return labels_temp
