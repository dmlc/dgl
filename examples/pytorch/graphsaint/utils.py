import json
import os
from functools import namedtuple

import dgl

import numpy as np
import scipy.sparse
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class Logger(object):
    """A custom logger to log stdout to a logging file."""

    def __init__(self, path):
        """Initialize the logger.

        Parameters
        ---------
        path : str
            The file path to be stored in.
        """
        self.path = path

    def write(self, s):
        with open(self.path, "a") as f:
            f.write(str(s))
        print(s)
        return


def save_log_dir(args):
    log_dir = "./log/{}/{}".format(args.dataset, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def calc_f1(y_true, y_pred, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), f1_score(
        y_true, y_pred, average="macro"
    )


def evaluate(model, g, labels, mask, multilabel=False):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        f1_mic, f1_mac = calc_f1(
            labels.cpu().numpy(), logits.cpu().numpy(), multilabel
        )
        return f1_mic, f1_mac


# load data of GraphSAINT and convert them to the format of dgl
def load_data(args, multilabel):
    if not os.path.exists("graphsaintdata") and not os.path.exists("data"):
        raise ValueError("The directory graphsaintdata does not exist!")
    elif os.path.exists("graphsaintdata") and not os.path.exists("data"):
        os.rename("graphsaintdata", "data")
    prefix = "data/{}".format(args.dataset)
    DataType = namedtuple("Dataset", ["num_classes", "train_nid", "g"])

    adj_full = scipy.sparse.load_npz("./{}/adj_full.npz".format(prefix)).astype(
        np.bool_
    )
    g = dgl.from_scipy(adj_full)
    num_nodes = g.num_nodes()

    adj_train = scipy.sparse.load_npz(
        "./{}/adj_train.npz".format(prefix)
    ).astype(np.bool_)
    train_nid = np.array(list(set(adj_train.nonzero()[0])))

    role = json.load(open("./{}/role.json".format(prefix)))
    mask = np.zeros((num_nodes,), dtype=bool)
    train_mask = mask.copy()
    train_mask[role["tr"]] = True
    val_mask = mask.copy()
    val_mask[role["va"]] = True
    test_mask = mask.copy()
    test_mask[role["te"]] = True

    feats = np.load("./{}/feats.npy".format(prefix))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    class_map = json.load(open("./{}/class_map.json".format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    if multilabel:
        # Multi-label binary classification
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_nodes, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_nodes,))
        for k, v in class_map.items():
            class_arr[k] = v

    g.ndata["feat"] = torch.tensor(feats, dtype=torch.float)
    g.ndata["label"] = torch.tensor(
        class_arr, dtype=torch.float if multilabel else torch.long
    )
    g.ndata["train_mask"] = torch.tensor(train_mask, dtype=torch.bool)
    g.ndata["val_mask"] = torch.tensor(val_mask, dtype=torch.bool)
    g.ndata["test_mask"] = torch.tensor(test_mask, dtype=torch.bool)

    data = DataType(g=g, num_classes=num_classes, train_nid=train_nid)
    return data
