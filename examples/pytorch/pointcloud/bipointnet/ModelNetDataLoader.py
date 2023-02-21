import os
import warnings

import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Farthest point sampler works as follows:
    1. Initialize the sample set S with a random point
    2. Pick point P not in S, which maximizes the distance d(P, S)
    3. Repeat step 2 until |S| = npoint

    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(
        self,
        root,
        npoint=1024,
        split="train",
        fps=False,
        normal_channel=True,
        cache_size=15000,
    ):
        """
        Input:
            root: the root path to the local data files
            npoint: number of points from each cloud
            split: which split of the data, 'train' or 'test'
            fps: whether to sample points with farthest point sampler
            normal_channel: whether to use additional channel
            cache_size: the cache size of in-memory point clouds
        """
        self.root = root
        self.npoints = npoint
        self.fps = fps
        self.catfile = os.path.join(self.root, "modelnet40_shape_names.txt")

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids["train"] = [
            line.rstrip()
            for line in open(os.path.join(self.root, "modelnet40_train.txt"))
        ]
        shape_ids["test"] = [
            line.rstrip()
            for line in open(os.path.join(self.root, "modelnet40_test.txt"))
        ]

        assert split == "train" or split == "test"
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [
            (
                shape_names[i],
                os.path.join(self.root, shape_names[i], shape_ids[split][i])
                + ".txt",
            )
            for i in range(len(shape_ids[split]))
        ]
        print("The size of %s data is %d" % (split, len(self.datapath)))

        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
            if self.fps:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0 : self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls
