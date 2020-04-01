import os, json, tqdm
import numpy as np
import dgl
from zipfile import ZipFile
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from dgl.data.utils import download, get_download_dir

class ModelNet(object):
    def __init__(self, path, num_points, max_num_points=2048):
        import h5py
        assert max_num_points >= num_points
        self.f = h5py.File(path)
        self.max_num_points = max_num_points
        self.num_points = num_points

        self.n_train = self.f['train/data'].shape[0]
        # self.n_valid = int(self.n_train / 5)
        self.n_valid = 0
        self.n_train -= self.n_valid
        self.n_test = self.f['test/data'].shape[0]

    def train(self):
        return ModelNetDataset(self, 'train')

    def valid(self):
        return ModelNetDataset(self, 'valid')

    def test(self):
        return ModelNetDataset(self, 'test')

def calc_dist(edges):
    dist = ((edges.src['x'] - edges.dst['x']) ** 2).sum(1, keepdim=True)
    return {'dist': dist}

class ModelNetDataset(Dataset):
    def __init__(self, modelnet, mode):
        super(ModelNetDataset, self).__init__()
        self.max_num_points = modelnet.max_num_points
        self.num_points = modelnet.num_points
        self.mode = mode

        if mode == 'train':
            self.data = modelnet.f['train/data'][:modelnet.n_train]
            self.label = modelnet.f['train/label'][:modelnet.n_train]
        elif mode == 'valid':
            self.data = modelnet.f['train/data'][modelnet.n_train:]
            self.label = modelnet.f['train/label'][modelnet.n_train:]
        elif mode == 'test':
            self.data = modelnet.f['test/data'].value
            self.label = modelnet.f['test/label'].value

    def translate(self, x, scale=(2/3, 3/2), shift=(-0.2, 0.2)):
        xyz1 = np.random.uniform(low=scale[0], high=scale[1], size=[3])
        xyz2 = np.random.uniform(low=shift[0], high=shift[1], size=[3])
        x = np.add(np.multiply(x, xyz1), xyz2).astype('float32')
        return x

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        if self.mode == 'train':
            inds = np.random.choice(self.max_num_points, self.num_points)
            x = self.data[i][inds]
            x = self.translate(x)
        else:
            x = self.data[i][:self.num_points]
        y = self.label[i]
        # complete graph
        n_nodes = x.shape[0]
        # np_csr = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        # csr = csr_matrix(np_csr)
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        # g.from_scipy_sparse_matrix(csr)
        g.ndata['x'] = x
        '''
        g.ndata['sampled'] = np.zeros((n_nodes, 1)).astype('long').copy()
        src = []
        dst = []
        for i in range(n_nodes - 1):
            for j in range(i+1, n_nodes):
                src.append(i)
                dst.append(j)
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        g.apply_edges(calc_dist)
        '''
        return g, y

class ShapeNet(object):
    def __init__(self, num_points=2048, normal_channel=True):
        self.num_points = num_points
        self.normal_channel = normal_channel

        SHAPENET_DOWNLOAD_URL = "https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
        download_path = get_download_dir()
        data_filename = "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
        data_path = os.path.join(download_path, "shapenetcore_partanno_segmentation_benchmark_v0_normal")
        if not os.path.exists(data_path):
            local_path = os.path.join(download_path, data_filename)
            if not os.path.exists(local_path):
                download(SHAPENET_DOWNLOAD_URL, local_path)
            with ZipFile(local_path) as z:
                z.extractall(path=download_path)

        synset_file = "synsetoffset2category.txt"
        with open(os.path.join(data_path, synset_file)) as f:
            synset = [t.split('\n')[0].split('\t') for t in f.readlines()]
        self.synset_dict = {}
        for syn in synset:
            self.synset_dict[syn[1]] = syn[0]
        self.seg_classes = {'Earphone': [16, 17, 18],
                            'Motorbike': [30, 31, 32, 33, 34, 35],
                            'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11],
                            'Laptop': [28, 29],
                            'Cap': [6, 7],
                            'Skateboard': [44, 45, 46],
                            'Mug': [36, 37],
                            'Guitar': [19, 20, 21],
                            'Bag': [4, 5],
                            'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49],
                            'Airplane': [0, 1, 2, 3],
                            'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15],
                            'Knife': [22, 23]}

        train_split_json = 'shuffled_train_file_list.json'
        val_split_json = 'shuffled_val_file_list.json'
        test_split_json = 'shuffled_test_file_list.json'
        split_path = os.path.join(data_path, 'train_test_split')
        with open(os.path.join(split_path, train_split_json)) as f:
            tmp = f.read()
            self.train_file_list = [os.path.join(data_path, t.replace('shape_data/', '') + '.txt') for t in json.loads(tmp)]
        with open(os.path.join(split_path, val_split_json)) as f:
            tmp = f.read()
            self.val_file_list = [os.path.join(data_path, t.replace('shape_data/', '') + '.txt') for t in json.loads(tmp)]
        with open(os.path.join(split_path, test_split_json)) as f:
            tmp = f.read()
            self.test_file_list = [os.path.join(data_path, t.replace('shape_data/', '') + '.txt') for t in json.loads(tmp)]

    def train(self):
        return ShapeNetDataset(self, 'train', self.num_points, self.normal_channel)

    def valid(self):
        return ShapeNetDataset(self, 'valid', self.num_points, self.normal_channel)

    def trainval(self):
        return ShapeNetDataset(self, 'trainval', self.num_points, self.normal_channel)

    def test(self):
        return ShapeNetDataset(self, 'test', self.num_points, self.normal_channel)

class ShapeNetDataset(Dataset):
    def __init__(self, shapenet, mode, num_points, normal_channel=True):
        super(ShapeNetDataset, self).__init__()
        self.mode = mode
        self.num_points = num_points
        if not normal_channel:
            self.dim = 3
        else:
            self.dim = 6

        if mode == 'train':
            self.file_list = shapenet.train_file_list
        elif mode == 'valid':
            self.file_list = shapenet.val_file_list
        elif mode == 'test':
            self.file_list = shapenet.test_file_list
        elif mode == 'trainval':
            self.file_list = shapenet.train_file_list + shapenet.val_file_list
        else:
            raise "Not supported `mode`"

        data_list = []
        label_list = []
        category_list = []
        print('Loading data from split ' + self.mode)
        for fn in tqdm.tqdm(self.file_list):
            with open(fn) as f:
                data = np.array([t.split('\n')[0].split(' ') for t in f.readlines()]).astype(np.float)
            data_list.append(data[:, 0:6])
            label_list.append(data[:, 6].astype(np.int))
            category_list.append(shapenet.synset_dict[fn.split('/')[-2]])
        self.data = data_list
        self.label = label_list
        self.category = category_list

    def translate(self, x, scale=(2/3, 3/2), shift=(-0.2, 0.2), size=3):
        xyz1 = np.random.uniform(low=scale[0], high=scale[1], size=[size])
        xyz2 = np.random.uniform(low=shift[0], high=shift[1], size=[size])
        x = np.add(np.multiply(x, xyz1), xyz2).astype('float32')
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        inds = np.random.choice(self.data[i].shape[0], self.num_points, replace=True)
        x = self.data[i][inds,:self.dim]
        y = self.label[i][inds]
        cat = self.category[i]
        if self.mode == 'train':
            x = self.translate(x, size=self.dim)
        n_nodes = x.shape[0]
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        # g.from_scipy_sparse_matrix(csr)
        g.ndata['x'] = x.astype(np.float)
        g.ndata['y'] = y.astype(np.int)
        return g, cat
