"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import os
import json

from .dgl_dataset import DGLDataset
from . import utils
from .. import backend as F

__all__ = ['AsNodePredDataset']

class AsNodePredDataset(DGLDataset):
    """Repurpose a dataset for a standard semi-supervised transductive
    node prediction task.

    The class converts a given dataset into a new dataset object that:

      - Contains only one graph, accessible from ``dataset[0]``.
      - The graph stores:
        - Node labels in ``g.ndata['label']``.
        - Train/val/test masks in ``g.ndata['train_mask']``, ``g.ndata['val_mask']``,
          and ``g.ndata['test_mask']`` respectively.
      - In addition, the dataset contains the following attributes:
        - ``num_classes``, the number of classes to predict.

    If the input dataset contains heterogeneous graphs, users need to specify the
    ``target_ntype`` argument to indicate which node type to make predictions for.
    In this case:

      - Node labels are stored in ``g.nodes[target_ntype].data['label']``.
      - Training masks are stored in ``g.nodes[target_ntype].data['train_mask']``.
        So do validation and test masks.
    
    The class will keep only the first graph in the provided dataset and
    generate train/val/test masks according to the given spplit ratio. The generated
    masks will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    target_ntype : str, optional
        The node type to add split mask for.

    Attributes
    ----------
    num_classes : int
        Number of classes to predict.

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
                 target_ntype=None,
                 **kwargs):
        self.g = dataset[0].clone()
        self.split_ratio = split_ratio
        self.target_ntype = target_ntype
        self.num_classes = getattr(dataset, 'num_classes', None)
        super().__init__(dataset.name + '-as-nodepred', **kwargs)

    def process(self):
        if 'label' not in self.g.nodes[self.target_ntype].data:
            raise ValueError("Missing node labels. Make sure labels are stored "
                             "under name 'label'.")
        if self.num_classes is None:
            self.num_classes = len(F.unique(self.g.nodes[self.target_ntype].data['label']))
        if self.verbose:
            print('Generating train/val/test masks...')
        utils.add_nodepred_split(self, self.split_ratio, self.target_ntype)

    def has_cache(self):
        return os.path.isfile(os.path.join(self.save_path, 'graph.bin'))

    def load(self):
        with open(os.path.join(self.save_path, 'info.json'), 'r') as f:
            info = json.load(f)
            if (info['split_ratio'] != self.split_ratio
                or info['target_ntype'] != self.target_ntype):
                raise ValueError('Provided split ratio is different from the cached file. '
                                 'Re-process the dataset.')
            self.split_ratio = info['split_ratio']
            self.target_ntype = info['target_ntype']
            self.num_classes = info['num_classes']
        gs, _ = utils.load_graphs(os.path.join(self.save_path, 'graph.bin'))
        self.g = gs[0]

    def save(self):
        utils.save_graphs(os.path.join(self.save_path, 'graph.bin'), [self.g])
        with open(os.path.join(self.save_path, 'info.json'), 'w') as f:
            json.dump({
                'split_ratio' : self.split_ratio,
                'target_ntype' : self.target_ntype,
                'num_classes' : self.num_classes}, f)

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1
