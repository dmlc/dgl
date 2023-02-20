import os
import ssl

import dgl

import numpy as np
import torch
from six.moves import urllib
from torch.utils.data import DataLoader, Dataset


def download_file(dataset):
    print("Start Downloading data: {}".format(dataset))
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/{}".format(
        dataset
    )
    print("Start Downloading File....")
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open("./data/{}".format(dataset), "wb") as handle:
        handle.write(data.read())


class SnapShotDataset(Dataset):
    def __init__(self, path, npz_file):
        if not os.path.exists(path + "/" + npz_file):
            if not os.path.exists(path):
                os.mkdir(path)
            download_file(npz_file)
        zipfile = np.load(path + "/" + npz_file)
        self.x = zipfile["x"]
        self.y = zipfile["y"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx, ...], self.y[idx, ...]


def METR_LAGraphDataset():
    if not os.path.exists("data/graph_la.bin"):
        if not os.path.exists("data"):
            os.mkdir("data")
        download_file("graph_la.bin")
    g, _ = dgl.load_graphs("data/graph_la.bin")
    return g[0]


class METR_LATrainDataset(SnapShotDataset):
    def __init__(self):
        super(METR_LATrainDataset, self).__init__("data", "metr_la_train.npz")
        self.mean = self.x[..., 0].mean()
        self.std = self.x[..., 0].std()


class METR_LATestDataset(SnapShotDataset):
    def __init__(self):
        super(METR_LATestDataset, self).__init__("data", "metr_la_test.npz")


class METR_LAValidDataset(SnapShotDataset):
    def __init__(self):
        super(METR_LAValidDataset, self).__init__("data", "metr_la_valid.npz")


def PEMS_BAYGraphDataset():
    if not os.path.exists("data/graph_bay.bin"):
        if not os.path.exists("data"):
            os.mkdir("data")
        download_file("graph_bay.bin")
    g, _ = dgl.load_graphs("data/graph_bay.bin")
    return g[0]


class PEMS_BAYTrainDataset(SnapShotDataset):
    def __init__(self):
        super(PEMS_BAYTrainDataset, self).__init__("data", "pems_bay_train.npz")
        self.mean = self.x[..., 0].mean()
        self.std = self.x[..., 0].std()


class PEMS_BAYTestDataset(SnapShotDataset):
    def __init__(self):
        super(PEMS_BAYTestDataset, self).__init__("data", "pems_bay_test.npz")


class PEMS_BAYValidDataset(SnapShotDataset):
    def __init__(self):
        super(PEMS_BAYValidDataset, self).__init__("data", "pems_bay_valid.npz")
