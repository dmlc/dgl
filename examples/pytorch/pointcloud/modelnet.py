import numpy as np

class ModelNet(object):
    def __init__(self, path, batch_size):
        import h5py
        self.f = h5py.File(path)
        self.batch_size = batch_size

        self.n_train = self.f['train/data'].shape[0]
        self.n_valid = int(self.n_train / 5)
        self.n_train -= self.n_valid
        self.n_test = self.f['test/data'].shape[0]

    def train(self):
        self.data = self.f['train/data'][:self.n_train]
        self.label = self.f['train/label'][:self.n_train]

    def valid(self):
        self.data = self.f['train/data'][self.n_train:]
        self.label = self.f['train/label'][self.n_train:]

    def test(self):
        self.data = self.f['test/data'].value
        self.label = self.f['test/label'].value

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        perm = np.random.permutation(data.shape[0])

        for i in range(0, len(perm), self.batch_size):
            samples = perm[i:i+self.batch_size]
            x = self.data[samples]
            y = self.label[samples]
            yield x, y
