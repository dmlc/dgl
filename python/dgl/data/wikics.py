"""Wiki-CS Dataset"""
import itertools
import json
import os

import numpy as np

from .. import backend as F
from ..convert import graph
from ..transforms import reorder_graph, to_bidirected
from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs


class WikiCSDataset(DGLBuiltinDataset):
    r"""Wiki-CS is a Wikipedia-based dataset for node classification from `Wiki-CS: A Wikipedia-Based
    Benchmark for Graph Neural Networks <https://arxiv.org/abs/2007.02901v2>`_

    The dataset consists of nodes corresponding to Computer Science articles, with edges based on
    hyperlinks and 10 classes representing different branches of the field.

    WikiCS dataset statistics:

    - Nodes: 11,701
    - Edges: 431,726 (note that the original dataset has 216,123 edges but DGL adds
      the reverse edges and removes the duplicate edges, hence with a different number)
    - Number of classes: 10
    - Node feature size: 300
    - Number of different train, validation, stopping splits: 20
    - Number of test split: 1

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset.
        Default: False
    verbose : bool
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
    >>> from dgl.data import WikiCSDataset
    >>> dataset = WikiCSDataset()
    >>> dataset.num_classes
    10
    >>> g = dataset[0]
    >>> # get node feature
    >>> feat = g.ndata['feat']
    >>> # get node labels
    >>> labels = g.ndata['label']
    >>> # get data split
    >>> train_mask = g.ndata['train_mask']
    >>> val_mask = g.ndata['val_mask']
    >>> stopping_mask = g.ndata['stopping_mask']
    >>> test_mask = g.ndata['test_mask']
    >>> # The shape of train, val and stopping masks are (num_nodes, num_splits).
    >>> # The num_splits is the number of different train, validation, stopping splits.
    >>> # Due to the number of test spilt is 1, the shape of test mask is (num_nodes,).
    >>> print(train_mask.shape, val_mask.shape, stopping_mask.shape)
    (11701, 20) (11701, 20) (11701, 20)
    >>> print(test_mask.shape)
    (11701,)
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=False, transform=None
    ):
        _url = _get_dgl_url("dataset/wiki_cs.zip")
        super(WikiCSDataset, self).__init__(
            name="wiki_cs",
            raw_dir=raw_dir,
            url=_url,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        """process raw data to graph, labels and masks"""
        with open(os.path.join(self.raw_path, "data.json")) as f:
            data = json.load(f)
        features = F.tensor(np.array(data["features"]), dtype=F.float32)
        labels = F.tensor(np.array(data["labels"]), dtype=F.int64)

        train_masks = np.array(data["train_masks"], dtype=bool).T
        val_masks = np.array(data["val_masks"], dtype=bool).T
        stopping_masks = np.array(data["stopping_masks"], dtype=bool).T
        test_mask = np.array(data["test_mask"], dtype=bool)

        edges = [[(i, j) for j in js] for i, js in enumerate(data["links"])]
        edges = np.array(list(itertools.chain(*edges)))
        src, dst = edges[:, 0], edges[:, 1]

        g = graph((src, dst))
        g = to_bidirected(g)

        g.ndata["feat"] = features
        g.ndata["label"] = labels
        g.ndata["train_mask"] = generate_mask_tensor(train_masks)
        g.ndata["val_mask"] = generate_mask_tensor(val_masks)
        g.ndata["stopping_mask"] = generate_mask_tensor(stopping_masks)
        g.ndata["test_mask"] = generate_mask_tensor(test_mask)

        g = reorder_graph(
            g,
            node_permute_algo="rcmk",
            edge_permute_algo="dst",
            store_ids=False,
        )

        self._graph = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        g, _ = load_graphs(graph_path)
        self._graph = g[0]

    @property
    def num_classes(self):
        return 10

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    def __getitem__(self, idx):
        r"""Get graph object

        Parameters
        ----------
        idx : int
            Item index, WikiCSDataset has only one graph object

        Returns
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
            - ``ndata['train_mask']``: train mask is for retrieving the nodes for training.
            - ``ndata['val_mask']``: val mask is for retrieving the nodes for hyperparameter tuning.
            - ``ndata['stopping_mask']``: stopping mask is for retrieving the nodes for early stopping criterion.
            - ``ndata['test_mask']``: test mask is for retrieving the nodes for testing.

        """
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)
