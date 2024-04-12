import json
import os
from zipfile import ZipFile

import dgl

import numpy as np
import tqdm
from dgl.data.utils import download, get_download_dir
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class ShapeNet(object):
    def __init__(self, num_points=2048, normal_channel=True):
        self.num_points = num_points
        self.normal_channel = normal_channel

        SHAPENET_DOWNLOAD_URL = "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
        download_path = get_download_dir()
        data_filename = (
            "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
        )
        data_path = os.path.join(
            download_path,
            "shapenetcore_partanno_segmentation_benchmark_v0_normal",
        )
        if not os.path.exists(data_path):
            local_path = os.path.join(download_path, data_filename)
            if not os.path.exists(local_path):
                download(SHAPENET_DOWNLOAD_URL, local_path, verify_ssl=False)
            with ZipFile(local_path) as z:
                z.extractall(path=download_path)

        synset_file = "synsetoffset2category.txt"
        with open(os.path.join(data_path, synset_file)) as f:
            synset = [t.split("\n")[0].split("\t") for t in f.readlines()]
        self.synset_dict = {}
        for syn in synset:
            self.synset_dict[syn[1]] = syn[0]
        self.seg_classes = {
            "Airplane": [0, 1, 2, 3],
            "Bag": [4, 5],
            "Cap": [6, 7],
            "Car": [8, 9, 10, 11],
            "Chair": [12, 13, 14, 15],
            "Earphone": [16, 17, 18],
            "Guitar": [19, 20, 21],
            "Knife": [22, 23],
            "Lamp": [24, 25, 26, 27],
            "Laptop": [28, 29],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Mug": [36, 37],
            "Pistol": [38, 39, 40],
            "Rocket": [41, 42, 43],
            "Skateboard": [44, 45, 46],
            "Table": [47, 48, 49],
        }

        train_split_json = "shuffled_train_file_list.json"
        val_split_json = "shuffled_val_file_list.json"
        test_split_json = "shuffled_test_file_list.json"
        split_path = os.path.join(data_path, "train_test_split")
        with open(os.path.join(split_path, train_split_json)) as f:
            tmp = f.read()
            self.train_file_list = [
                os.path.join(data_path, t.replace("shape_data/", "") + ".txt")
                for t in json.loads(tmp)
            ]
        with open(os.path.join(split_path, val_split_json)) as f:
            tmp = f.read()
            self.val_file_list = [
                os.path.join(data_path, t.replace("shape_data/", "") + ".txt")
                for t in json.loads(tmp)
            ]
        with open(os.path.join(split_path, test_split_json)) as f:
            tmp = f.read()
            self.test_file_list = [
                os.path.join(data_path, t.replace("shape_data/", "") + ".txt")
                for t in json.loads(tmp)
            ]

    def train(self):
        return ShapeNetDataset(
            self, "train", self.num_points, self.normal_channel
        )

    def valid(self):
        return ShapeNetDataset(
            self, "valid", self.num_points, self.normal_channel
        )

    def trainval(self):
        return ShapeNetDataset(
            self, "trainval", self.num_points, self.normal_channel
        )

    def test(self):
        return ShapeNetDataset(
            self, "test", self.num_points, self.normal_channel
        )


class ShapeNetDataset(Dataset):
    def __init__(self, shapenet, mode, num_points, normal_channel=True):
        super(ShapeNetDataset, self).__init__()
        self.mode = mode
        self.num_points = num_points
        if not normal_channel:
            self.dim = 3
        else:
            self.dim = 6

        if mode == "train":
            self.file_list = shapenet.train_file_list
        elif mode == "valid":
            self.file_list = shapenet.val_file_list
        elif mode == "test":
            self.file_list = shapenet.test_file_list
        elif mode == "trainval":
            self.file_list = shapenet.train_file_list + shapenet.val_file_list
        else:
            raise "Not supported `mode`"

        data_list = []
        label_list = []
        category_list = []
        print("Loading data from split " + self.mode)
        for fn in tqdm.tqdm(self.file_list, ascii=True):
            with open(fn) as f:
                data = np.array(
                    [t.split("\n")[0].split(" ") for t in f.readlines()]
                ).astype(np.float)
            data_list.append(data[:, 0 : self.dim])
            label_list.append(data[:, 6].astype(int))
            category_list.append(shapenet.synset_dict[fn.split("/")[-2]])
        self.data = data_list
        self.label = label_list
        self.category = category_list

    def translate(self, x, scale=(2 / 3, 3 / 2), shift=(-0.2, 0.2), size=3):
        xyz1 = np.random.uniform(low=scale[0], high=scale[1], size=[size])
        xyz2 = np.random.uniform(low=shift[0], high=shift[1], size=[size])
        x = np.add(np.multiply(x, xyz1), xyz2).astype("float32")
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        inds = np.random.choice(
            self.data[i].shape[0], self.num_points, replace=True
        )
        x = self.data[i][inds, : self.dim]
        y = self.label[i][inds]
        cat = self.category[i]
        if self.mode == "train":
            x = self.translate(x, size=self.dim)
        x = x.astype(np.float)
        y = y.astype(int)
        return x, y, cat
