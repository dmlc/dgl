"""Wiki-CS Dataset"""
import itertools
import os
import json
import numpy as np
from .. import backend as F
from ..convert import graph
from .dgl_dataset import DGLBuiltinDataset
from ..transforms import to_bidirected, reorder_graph
from .utils import generate_mask_tensor, load_graphs, save_graphs, _get_dgl_url


class WikiCSDataset(DGLBuiltinDataset):
    r"""Wiki-CS is a Wikipedia-based dataset for node classification.

    The dataset consists of nodes corresponding to Computer Science articles, with edges based on
    hyperlinks and 10 classes representing different branches of the field.

    Reference: `<https://arxiv.org/abs/2109.01116>`_

    WikiCS dataset statistics:

    - Classes : 10
    - Nodes: 11,701
    - Edges: 216,123
    - Features dimension: 300
    - Label rate: 0.05
    - Mean degree: 36.94
    - Average shortest path length: 3.01
    - Different train, validation, stopping splits: 20

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset.
        Default: False
    verbose: bool
        Whether to print out progress information.
        Default: False
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_classes : int
        Number of node classes

    Examples
    --------
    >>> dataset = WikiCSDataset()
    >>> len(dataset)
    1
    >>> for g in dataset:
    ....    # get edge feature
    ....    feat = g.ndata['feat']
    ....    # your code here
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        _url = _get_dgl_url('dataset/wiki_cs.zip')
        super(WikiCSDataset, self).__init__(name='wiki_cs',
                                            raw_dir=raw_dir,
                                            url=_url,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform)

    def process(self):
        """process raw data to graph, labels and masks"""
        data = json.load(open(os.path.join(self.raw_path, 'data.json')))
        features = F.tensor(np.array(data['features'], dtype=F.data_type_dict['float32']))
        labels = F.tensor(np.array(data['labels'], dtype=F.data_type_dict['int64']))

        train_masks = np.array(data['train_masks'], dtype=bool).T
        val_masks = np.array(data['val_masks'], dtype=bool).T
        stopping_masks = np.array(data['stopping_masks'], dtype=bool).T
        test_mask = np.array(data['test_mask'], dtype=bool)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = np.array(list(itertools.chain(*edges)))
        src, dst = edges[:, 0], edges[:, 1]

        g = graph((src, dst))
        g = to_bidirected(g)

        g.ndata['feat'] = features
        g.ndata['label'] = labels
        g.ndata['train_mask'] = generate_mask_tensor(train_masks)
        g.ndata['val_mask'] = generate_mask_tensor(val_masks)
        g.ndata['stopping_mask'] = generate_mask_tensor(stopping_masks)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)

        g = reorder_graph(g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)

        self._graph = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        g, _ = load_graphs(graph_path)
        self._graph = g[0]

    @property
    def num_classes(self):
        return 10

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    def __getitem__(self, idx):
        r""" Get graph object

        Parameters
        ----------
        idx : int
            Item index, WikiCSDataset has only one graph object

        Returns
        -------
        :class:`dgl.DGLGraph`

            graph structure, labels, features and masks.

            - ``ndata['label']``: ground truth labels
            - ``ndata['feat']``: node features
            - ``ndata['train_mask']``: training mask
            - ``ndata['val_mask']``: validation mask
            - ``ndata['stopping_mask']``: stopping mask
            - ``ndata['test_mask']``: test mask
        """
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)
