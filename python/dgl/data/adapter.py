"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import os
import json
import numpy as np

from .. import backend as F
from ..convert import graph as create_dgl_graph
from ..sampling.negative import _calc_redundancy
from .dgl_dataset import DGLDataset
from . import utils

__all__ = ['AsNodePredDataset', 'AsEdgePredDataset']

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
        self.num_classes = dataset.num_classes
        super().__init__(dataset.name + '-as-nodepred', **kwargs)

    def process(self):
        if 'label' not in self.g.nodes[self.target_ntype].data:
            raise ValueError("Missing node labels. Make sure labels are stored "
                             "under name 'label'.")
        if any(s not in self.g.nodes[self.target_ntype].ndata for s in ["train_mask", "val_mask", "test_mask"]):
            # only generate when information not available
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



def negative_sample(g, num_samples):
    """Random sample negative edges from graph, excluding self-loops"""
    num_nodes = g.num_nodes()
    redundancy = _calc_redundancy(
        num_samples, g.num_edges(), num_nodes ** 2)
    sample_size = int(num_samples*(1+redundancy))
    edges = np.random.randint(0, num_nodes, size=(2, sample_size))
    edges = np.unique(edges, axis=1)
    # remove self loop
    mask_self_loop = edges[0] == edges[1]
    # remove existing edges
    has_edges = F.asnumpy(g.has_edges_between(edges[0], edges[1]))
    mask = ~(np.logical_or(mask_self_loop, has_edges))
    edges = edges[:, mask]
    if edges.shape[1] >= num_samples:
        edges = edges[:, :num_samples]
    return F.tensor(edges)


class AsEdgePredDataset(DGLDataset):
    """Repurpose a dataset for edge prediction task.

    The created dataset will include data needed for link prediction. 
    It will keep only the first graph in the provided dataset and
    generate train/val/test edges according to the given split ratio,
    and the correspondent negative edges based on the neg_ratio. The generated
    edges will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    neg_ratio : int, optional
        Indicate how much negative samples to be sampled
        The number of the negative samples will be neg_ratio * num_positive_edges.

    Returns
    -------
    DGLDataset
        A new dataset with only one graph. Train/val/test masks are stored in the
        ndata of the graph.

    Examples
    --------
    >>> ds = dgl.data.CoraGraphDataset()
    >>> print(ds)
    Dataset("cora_v2", num_graphs=1, save_path=...)
    >>> new_ds = dgl.data.AsNodePredDataset(ds, [0.8, 0.1, 0.1])
    >>> print(new_ds)
    Dataset("cora_v2-as-edgepred", num_graphs=1, save_path=/home/ubuntu/.dgl/cora_v2-as-edgepred)
    >>> print(hasattr(new_ds, "get_test_edges"))
    True
    """

    def __init__(self,
                 dataset,
                 split_ratio=[0.8, 0.1, 0.1],
                 neg_ratio=3,
                 add_self_loop=True,
                 **kwargs):
        self.g = dataset[0]
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.neg_ratio = neg_ratio
        self.add_self_loop = add_self_loop
        super().__init__(dataset.name + '-as-edgepred', hash_key=(neg_ratio, split_ratio, add_self_loop), **kwargs)

    def process(self):
        if hasattr(self.dataset, "get_edge_split"):
            # This is likely to be an ogb dataset
            self.edge_split = self.dataset.get_edge_split()
            self.train_graph = self.g

            pos_e_tensor, neg_e_tensor = self.edge_split["valid"][
                "edge"], self.edge_split["valid"]["edge_neg"]
            pos_e = (pos_e_tensor[:, 0], pos_e_tensor[:, 1])
            neg_e = (neg_e_tensor[:, 0], neg_e_tensor[:, 1])
            self.val_edges = pos_e, neg_e

            pos_e_tensor, neg_e_tensor = self.edge_split["test"][
                "edge"], self.edge_split["test"]["edge_neg"]
            pos_e = (pos_e_tensor[:, 0], pos_e_tensor[:, 1])
            neg_e = (neg_e_tensor[:, 0], neg_e_tensor[:, 1])
            self.test_edges = pos_e, neg_e
        else:
            ratio = self.split_ratio
            graph = self.dataset[0]
            n = graph.num_edges()
            src, dst = graph.edges()
            n_train, n_val, n_test = int(
                n * ratio[0]), int(n * ratio[1]), int(n * ratio[2])

            idx = np.random.permutation(n)
            train_pos_idx = idx[:n_train]
            val_pos_idx = idx[n_train:n_train+n_val]
            test_pos_idx = idx[n_train+n_val:]
            neg_src, neg_dst = negative_sample(
                graph, self.neg_ratio*(n_val+n_test))
            neg_n_val, neg_n_test = self.neg_ratio * n_val, self.neg_ratio * n_test
            neg_val_src, neg_val_dst = neg_src[:neg_n_val], neg_dst[:neg_n_val]
            neg_test_src, neg_test_dst = neg_src[neg_n_val:], neg_dst[neg_n_val:]
            self.val_edges = (src[val_pos_idx], dst[val_pos_idx]
                              ), (neg_val_src, neg_val_dst)
            self.test_edges = (src[test_pos_idx],
                               dst[test_pos_idx]), (neg_test_src, neg_test_dst)
            self.train_graph = create_dgl_graph(
                (src[train_pos_idx], dst[train_pos_idx]), num_nodes=self.num_nodes)
            self.train_graph.ndata["feat"] = graph.ndata["feat"]
        if self.add_self_loop:
            self.train_graph = self.train_graph.add_self_loop()

    def has_cache(self):
        return os.path.isfile(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))

    def load(self):
        gs, tensor_dict = utils.load_graphs(
            os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))
        self.g = gs[0]
        self.train_graph = self.g
        self.val_edges = (tensor_dict["val_pos_src"], tensor_dict["val_pos_dst"]), (
            tensor_dict["val_neg_src"], tensor_dict["val_neg_dst"])
        self.test_edges = (tensor_dict["test_pos_src"], tensor_dict["test_pos_dst"]), (
            tensor_dict["test_neg_src"], tensor_dict["test_neg_dst"])

        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'r') as f:
            info = json.load(f)
            self.split_ratio = info["split_ratio"]
            self.neg_ratio = info["neg_ratio"]
            self.add_self_loop = bool(info["add_self_loop"])

    def save(self):
        tensor_dict = {
            "val_pos_src": self.val_edges[0][0],
            "val_pos_dst": self.val_edges[0][1],
            "val_neg_src": self.val_edges[1][0],
            "val_neg_dst": self.val_edges[1][1],
            "test_pos_src": self.test_edges[0][0],
            "test_pos_dst": self.test_edges[0][1],
            "test_neg_src": self.test_edges[1][0],
            "test_neg_dst": self.test_edges[1][1],
        }
        utils.save_graphs(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)), [
                          self.train_graph], tensor_dict)
        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'w') as f:
            json.dump({
                'split_ratio': self.split_ratio,
                'neg_ratio': self.neg_ratio,
                "add_self_loop": self.add_self_loop}, f)

    @property
    def feat_size(self):
        return self.train_graph.ndata["feat"].shape[-1]

    @property
    def num_nodes(self):
        return self.g.num_nodes()

    def get_train_graph(self):
        return self.train_graph

    def get_val_edges(self):
        return self.val_edges

    def get_test_edges(self):
        return self.test_edges

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1
