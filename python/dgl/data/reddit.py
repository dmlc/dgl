""" Reddit dataset for community detection """
from __future__ import absolute_import

import scipy.sparse as sp
import numpy as np
import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs, deprecate_property
from .. import backend as F
from ..convert import from_scipy


class RedditDataset(DGLBuiltinDataset):
    r""" Reddit dataset for community detection (node classification)

    .. deprecated:: 0.5.0

        - ``graph`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]

        - ``num_labels`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> num_classes = dataset.num_classes

        - ``train_mask`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.ndata['train_mask']

        - ``val_mask`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.ndata['val_mask']

        - ``test_mask`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.ndata['test_mask']

        - ``features`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> features = graph.ndata['feat']

        - ``labels`` is deprecated, it is replaced by:

            >>> dataset = RedditDataset()
            >>> graph = dataset[0]
            >>> labels = graph.ndata['label']

    This is a graph dataset from Reddit posts made in the month of September, 2014.
    The node label in this case is the community, or “subreddit”, that a post belongs to.
    The authors sampled 50 large communities and built a post-to-post graph, connecting
    posts if the same user comments on both. In total this dataset contains 232,965
    posts with an average degree of 492. We use the first 20 days for training and the
    remaining days for testing (with 30% used for validation).

    Reference: `<http://snap.stanford.edu/graphsage/>`_

    Statistics

    - Nodes: 232,965
    - Edges: 114,615,892
    - Node feature size: 602
    - Number of training samples: 153,431
    - Number of validation samples: 23,831
    - Number of test samples: 55,703

    Parameters
    ----------
    self_loop : bool
        Whether load dataset with self loop connections. Default: False
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_classes : int
        Number of classes for each node
    graph : :class:`dgl.DGLGraph`
        Graph of the dataset
    num_labels : int
        Number of classes for each node
    train_mask: numpy.ndarray
        Mask of training nodes
    val_mask: numpy.ndarray
        Mask of validation nodes
    test_mask: numpy.ndarray
        Mask of test nodes
    features : Tensor
        Node features
    labels :  Tensor
        Node labels

    Examples
    --------
    >>> data = RedditDataset()
    >>> g = data[0]
    >>> num_classes = data.num_classes
    >>>
    >>> # get node feature
    >>> feat = g.ndata['feat']
    >>>
    >>> # get data split
    >>> train_mask = g.ndata['train_mask']
    >>> val_mask = g.ndata['val_mask']
    >>> test_mask = g.ndata['test_mask']
    >>>
    >>> # get labels
    >>> label = g.ndata['label']
    >>>
    >>> # Train, Validation and Test
    """
    def __init__(self, self_loop=False, raw_dir=None, force_reload=False, verbose=False):
        self_loop_str = ""
        if self_loop:
            self_loop_str = "_self_loop"
        _url = _get_dgl_url("dataset/reddit{}.zip".format(self_loop_str))
        self._self_loop_str = self_loop_str
        super(RedditDataset, self).__init__(name='reddit{}'.format(self_loop_str),
                                            url=_url,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
        # graph
        coo_adj = sp.load_npz(os.path.join(
            self.raw_path, "reddit{}_graph.npz".format(self._self_loop_str)))
        self._graph = from_scipy(coo_adj)
        # features and labels
        reddit_data = np.load(os.path.join(self.raw_path, "reddit_data.npz"))
        features = reddit_data["feature"]
        labels = reddit_data["label"]
        # tarin/val/test indices
        node_types = reddit_data["node_types"]
        train_mask = (node_types == 1)
        val_mask = (node_types == 2)
        test_mask = (node_types == 3)
        self._graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
        self._graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
        self._graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
        self._graph.ndata['feat'] = F.tensor(features, dtype=F.data_type_dict['float32'])
        self._graph.ndata['label'] = F.tensor(labels, dtype=F.data_type_dict['int64'])
        self._print_info()

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
        self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
        self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())
        self._print_info()

    def _print_info(self):
        if self.verbose:
            print('Finished data loading.')
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._graph.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(F.nonzero_1d(self._graph.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(F.nonzero_1d(self._graph.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(F.nonzero_1d(self._graph.ndata['test_mask']).shape[0]))

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 41

    @property
    def num_labels(self):
        deprecate_property('dataset.num_labels', 'dataset.num_classes')
        return self.num_classes

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'dataset[0]')
        return self._graph

    @property
    def train_mask(self):
        deprecate_property('dataset.train_mask', 'graph.ndata[\'train_mask\']')
        return F.asnumpy(self._graph.ndata['train_mask'])

    @property
    def val_mask(self):
        deprecate_property('dataset.val_mask', 'graph.ndata[\'val_mask\']')
        return F.asnumpy(self._graph.ndata['val_mask'])

    @property
    def test_mask(self):
        deprecate_property('dataset.test_mask', 'graph.ndata[\'test_mask\']')
        return F.asnumpy(self._graph.ndata['test_mask'])

    @property
    def features(self):
        deprecate_property('dataset.features', 'graph.ndata[\'feat\']')
        return self._graph.ndata['feat']

    @property
    def labels(self):
        deprecate_property('dataset.labels', 'graph.ndata[\'label\']')
        return self._graph.ndata['label']

    def __getitem__(self, idx):
        r""" Get graph by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node labels, node features and splitting masks:

            - ``ndata['label']``: node label
            - ``ndata['feat']``: node feature
            - ``ndata['train_mask']``： mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']:`` mask for test node set
        """
        assert idx == 0, "Reddit Dataset only has one graph"
        return self._graph

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1
