
import torch as T
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from itertools import product
from util import *
import os
import cv2
import numpy as NP
import numpy.random as RNG

def mnist_bbox(data):
    n_rows, n_cols = data.size()
    rowwise_max = data.max(0)[0]
    colwise_max = data.max(1)[0]
    rowwise_max_mask = rowwise_max == 0
    colwise_max_mask = colwise_max == 0

    left = T.cumprod(rowwise_max_mask, 0).sum()
    top = T.cumprod(colwise_max_mask, 0).sum()
    right = n_cols - T.cumprod(reverse(rowwise_max_mask, 0), 0).sum()
    bottom = n_rows - T.cumprod(reverse(colwise_max_mask, 0), 0).sum()

    x = (left + right) / 2
    y = (top + bottom) / 2
    w = right - left
    h = bottom - top

    return T.FloatTensor([x, y, w, h])

class MNISTMulti(Dataset):
    dir_ = 'multi'
    seeds = {'train': 1000, 'valid': 2000, 'test': 3000}
    attr_prefix = {'train': 'training', 'valid': 'valid', 'test': 'test'}
    n_classes = 10

    @property
    def _meta(self):
        return '%d-%d-%d-%d.pt' % (
                self.image_rows,
                self.image_cols,
                self.n_digits,
                self.backrand)

    @property
    def training_file(self):
        return os.path.join(self.dir_, 'training-' + self._meta)

    @property
    def test_file(self):
        return os.path.join(self.dir_, 'test-' + self._meta)

    @property
    def valid_file(self):
        return os.path.join(self.dir_, 'valid-' + self._meta)

    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 download=False,
                 image_rows=100,
                 image_cols=100,
                 n_digits=1,
                 size_multiplier=1,
                 backrand=0):
        self.mode = mode
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.n_digits = n_digits
        self.backrand = backrand

        if os.path.exists(self.dir_):
            if os.path.isfile(self.dir_):
                raise NotADirectoryError(self.dir_)
            elif os.path.exists(getattr(self, self.attr_prefix[mode] + '_file')):
                data = T.load(getattr(self, self.attr_prefix[mode] + '_file'))
                for k in data:
                    setattr(self, mode + '_' + k, data[k])
                self.size = getattr(self, mode + '_data').size()[0]
                return
        elif not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

        valid_src_size = 10000 // n_digits

        for _mode in ['train', 'valid', 'test']:
            _train = (_mode != 'test')
            mnist = MNIST(root, _train, transform, target_transform, download)

            if _mode == 'train':
                src_data = mnist.train_data[:-valid_src_size]
                src_labels = mnist.train_labels[:-valid_src_size]
            elif _mode == 'valid':
                src_data = mnist.train_data[-valid_src_size:]
                src_labels = mnist.train_labels[-valid_src_size:]
            elif _mode == 'test':
                src_data = mnist.test_data
                src_labels = mnist.test_labels

            with T.random.fork_rng():
                T.random.manual_seed(self.seeds[_mode])

                n_samples, n_rows, n_cols = src_data.size()
                n_new_samples = n_samples * n_digits
                data = T.ByteTensor(n_new_samples, image_rows, image_cols).zero_()
                labels = T.LongTensor(n_new_samples, n_digits).zero_()
                locs = T.LongTensor(n_new_samples, n_digits, 4).zero_()

                for i, j in product(range(n_digits), range(n_digits * size_multiplier)):
                    pos_rows = (T.LongTensor(n_samples).random_() %
                                (image_rows - n_rows))
                    pos_cols = (T.LongTensor(n_samples).random_() %
                                (image_cols - n_cols))
                    perm = T.randperm(n_samples)
                    for k, idx in zip(
                            range(n_samples * j, n_samples * (j + 1)), perm):
                        cur_rows = RNG.randint(n_rows // 3 * 2, n_rows)
                        cur_cols = RNG.randint(n_rows // 3 * 2, n_cols)
                        row = RNG.randint(image_rows - cur_rows)
                        col = RNG.randint(image_cols - cur_cols)
                        cur_data = T.from_numpy(
                                cv2.resize(
                                    src_data[idx].numpy(),
                                    (cur_cols, cur_rows))
                                )
                        data[k, row:row+cur_rows, col:col+cur_cols][cur_data != 0] = cur_data[cur_data != 0]
                        labels[k, i] = src_labels[idx]
                        locs[k, i] = mnist_bbox(cur_data)
                        locs[k, i, 0] += col
                        locs[k, i, 1] += row

                if backrand:
                    data += (data.new(*data.size()).random_() % backrand) * (data == 0)

            T.save({
                'data': data,
                'labels': labels,
                'locs': locs,
                }, getattr(self, self.attr_prefix[_mode] + '_file'))

            if _mode == mode:
                setattr(self, mode + '_data', data)
                setattr(self, mode + '_labels', labels)
                setattr(self, mode + '_locs', locs)
                self.size = data.size()[0]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return tuple(getattr(self, self.mode + '_' + k)[i] for k in ['data', 'labels', 'locs'])
