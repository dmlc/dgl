from collections import defaultdict

import numpy as np
import torch


def remove_unseen_classes_from_training(train_mask, labels, removed_class):
    """Remove the unseen classes (the first three classes by default) to get the zero-shot (i.e., completely imbalanced) label setting
    Input: train_mask, labels, removed_class
    Output: train_mask_zs: the bool list only containing seen classes
    """
    train_mask_zs = train_mask.clone()
    for i in range(train_mask_zs.numel()):
        if train_mask_zs[i] == 1 and (labels[i].item() in removed_class):
            train_mask_zs[i] = 0
    return train_mask_zs


def get_class_set(labels):
    """Get the class set.
    Input: labels [l, [c1, c2, ..]]
    Output：the labeled class set dict_keys([k1, k2, ..])
    """
    mydict = {}
    for y in labels:
        for label in y:
            mydict[int(label)] = 1
    return mydict.keys()


def get_label_attributes(train_mask_zs, nodeids, labellist, features):
    """Get the class-center (semanic knowledge) of each seen class.
    Suppose a node i is labeled as c, then attribute[c] += node_i_attribute, finally mean(attribute[c])
    Input: train_mask_zs, nodeids, labellist, features
    Output: label_attribute{}: label -> average_labeled_node_features (class centers)
    """
    _, feat_num = features.shape
    labels = get_class_set(labellist)
    label_attribute_nodes = defaultdict(list)
    for nodeid, labels in zip(nodeids, labellist):
        for label in labels:
            label_attribute_nodes[int(label)].append(int(nodeid))
    label_attribute = {}
    for label in label_attribute_nodes.keys():
        nodes = label_attribute_nodes[int(label)]
        selected_features = features[nodes, :]
        label_attribute[int(label)] = np.mean(selected_features, axis=0)
    return label_attribute


def get_labeled_nodes_label_attribute(train_mask_zs, labels, features, cuda):
    """Replace the original labels by their class-centers.
    For each label c in the training set, the following operations will be performed:
    Get label_attribute{} through function get_label_attributes, then res[i, :] = label_attribute[c]
    Input: train_mask_zs, labels, features
    Output: Y_{semantic} [l, ft]: tensor
    """
    X = torch.LongTensor(range(features.shape[0]))
    nodeids = []
    labellist = []
    for i in X[train_mask_zs].numpy().tolist():
        nodeids.append(str(i))
    for i in labels[train_mask_zs].cpu().numpy().tolist():
        labellist.append([str(i)])

    # 1. get the semantic knowledge (class centers) of all seen classes
    label_attribute = get_label_attributes(
        train_mask_zs=train_mask_zs,
        nodeids=nodeids,
        labellist=labellist,
        features=features.cpu().numpy(),
    )

    # 2. replace original labels by their class centers (semantic knowledge)
    res = np.zeros([len(nodeids), features.shape[1]])
    for i, labels in enumerate(labellist):
        # support mutiple labels
        c = len(labels)
        temp = np.zeros([c, features.shape[1]])
        for ii, label in enumerate(labels):
            temp[ii, :] = label_attribute[int(label)]
        temp = np.mean(temp, axis=0)
        res[i, :] = temp
    if cuda:
        res = torch.FloatTensor(res).cuda()
    else:
        res = torch.FloatTensor(res)
    return res
