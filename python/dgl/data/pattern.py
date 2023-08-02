""" PATTERNDataset for inductive learning. """
import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, load_graphs


class PATTERNDataset(DGLBuiltinDataset):
    r"""PATTERN dataset for graph pattern recognition task.

    Each graph G contains 5 communities with sizes randomly selected between [5, 35].
    The SBM of each community is p = 0.5, q = 0.35, and the node features on G are
    generated with a uniform random distribution with a vocabulary of size 3, i.e. {0, 1, 2}.
    Then randomly generate 100 patterns P composed of 20 nodes with intra-probability :math:`p_P` = 0.5
    and extra-probability :math:`q_P` = 0.5 (i.e. 50% of nodes in P are connected to G). The node features
    for P are also generated as a random signal with values {0, 1, 2}. The graphs are of sizes
    44-188 nodes. The output node labels have value 1 if the node belongs to P and value 0 if it is in G.

    Reference `<https://arxiv.org/pdf/2003.00982.pdf>`_

    Statistics:

    - Train examples: 10,000
    - Valid examples: 2,000
    - Test examples: 2,000
    - Number of classes for each node: 2

    Parameters
    ----------
    mode : str
        Must be one of ('train', 'valid', 'test').
        Default: 'train'
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
        Number of classes for each node.

    Examples
    --------
    >>> from dgl.data import PATTERNDataset
    >>> data = PATTERNDataset(mode='train')
    >>> data.num_classes
    2
    >>> len(trainset)
    10000
    >>> data[0]
    Graph(num_nodes=108, num_edges=4884, ndata_schemes={'feat': Scheme(shape=(), dtype=torch.int64), 'label': Scheme(shape=(), dtype=torch.int16)}
    edata_schemes={'feat': Scheme(shape=(1,), dtype=torch.float32)})
    """

    def __init__(
        self,
        mode="train",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        _url = _get_dgl_url("dataset/SBM_PATTERN.zip")

        super(PATTERNDataset, self).__init__(
            name="pattern",
            url=_url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        self.load()

    @property
    def graph_path(self):
        return os.path.join(
            self.save_path, "SBM_PATTERN_{}.bin".format(self.mode)
        )

    def has_cache(self):
        return os.path.exists(self.graph_path)

    def load(self):
        self._graphs, _ = load_graphs(self.graph_path)

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 2

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self._graphs)

    def __getitem__(self, idx):
        r"""Get the idx^th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features, node labels and edge features.

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
            - ``edata['feat']``: edge features
        """
        if self._transform is None:
            return self._graphs[idx]
        else:
            return self._transform(self._graphs[idx])
