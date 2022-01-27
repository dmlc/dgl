"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import os
import json

from .dgl_dataset import DGLDataset
from . import utils

__all__ = ['AsNodePredDataset']

class AsNodePredDataset(DGLDataset):
    """Repurpose a dataset for node prediction task.

    The created dataset will include data needed for semi-supervised transductive
    node prediction. It will keep only the first graph in the provided dataset and
    generate train/val/test masks according to the given spplit ratio. The generated
    masks will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    split_ntype : str, optional
        The node type to add split mask for.

    Returns
    -------
    DGLDataset
        A new dataset with only one graph. Train/val/test masks are stored in the
        ndata of the graph.

    Examples
    --------
    >>> ds = dgl.data.AmazonCoBuyComputerDataset()
    >>> print(ds)
    Dataset("amazon_co_buy_computer", num_graphs=1, save_path=...)
    >>> new_ds = dgl.data.AsNodePredDataset(ds, [0.8, 0.1, 0.1])
    >>> print(new_ds)
    Dataset("amazon_co_buy_computer-as-nodepred", num_graphs=1, save_path=...)
    >>> print('train_mask' in new_ds[0].ndata)
    True
    """
    def __init__(self,
                 dataset,
                 split_ratio=[0.8, 0.1, 0.1],
                 split_ntype=None,
                 **kwargs):
        self.g = dataset[0]
        self.split_ratio = split_ratio
        self.split_ntype = split_ntype
        self.num_classes = dataset.num_classes
        super().__init__(dataset.name + '-as-nodepred', **kwargs)

    def process(self):
        if self.verbose:
            print('Generating train/val/test masks...')
        utils.add_nodepred_split(self, self.split_ratio)

    def has_cache(self):
        return os.path.isfile(os.path.join(self.save_path, 'graph.bin'))

    def load(self):
        with open(os.path.join(self.save_path, 'info.json'), 'r') as f:
            info = json.load(f)
            if info['split_ratio'] != self.split_ratio or info['split_ntype'] != self.split_ntype:
                raise ValueError('Provided split ratio is different from the cached file. '
                                 'Re-process the dataset.')
            self.split_ratio = info['split_ratio']
            self.split_ntype = info['split_ntype']
            self.num_classes = info['num_classes']
        gs, _ = utils.load_graphs(os.path.join(self.save_path, 'graph.bin'))
        self.g = gs[0]

    def save(self):
        utils.save_graphs(os.path.join(self.save_path, 'graph.bin'), [self.g])
        with open(os.path.join(self.save_path, 'info.json'), 'w') as f:
            json.dump({
                'split_ratio' : self.split_ratio,
                'split_ntype' : self.split_ntype,
                'num_classes' : self.num_classes}, f)

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1
