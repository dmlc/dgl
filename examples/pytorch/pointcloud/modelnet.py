import numpy as np
import dgl
from torch.utils.data import Dataset

class ModelNet(object):
    def __init__(self, path, num_points, max_num_points=2048):
        import h5py
        assert max_num_points >= num_points
        self.f = h5py.File(path)
        self.max_num_points = max_num_points
        self.num_points = num_points

        self.n_train = self.f['train/data'].shape[0]
        self.n_valid = int(self.n_train / 5)
        self.n_train -= self.n_valid
        self.n_test = self.f['test/data'].shape[0]

    def train(self):
        return ModelNetDataset(self, 'train')

    def valid(self):
        return ModelNetDataset(self, 'valid')

    def test(self):
        return ModelNetDataset(self, 'test')

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
            np.random.shuffle(x)
        else:
            x = self.data[i][:self.num_points]
        y = self.label[i]
        g = dgl.DGLGraph()
        g.add_nodes(x.shape[0])
        g.ndata['x'] = x
        return g, x, y
