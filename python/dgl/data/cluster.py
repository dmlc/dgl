""" CLUSTERDataset for inductive learning. """
import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, load_graphs


class CLUSTERDataset(DGLBuiltinDataset):
    r"""CLUSTER dataset for semi-supervised clustering task.

    Each graph contains 6 SBM clusters with sizes randomly selected between
    [5, 35] and probabilities p = 0.55, q = 0.25. The graphs are of sizes 40
    -190 nodes. Each node can take an input feature value in {0, 1, 2, ..., 6}
    and values 1~6 correspond to classes 0~5 respectively, while value 0 means
    that the class of the node is unknown. There is only one labeled node that
    is randomly assigned to each community and most node features are set to 0.

    Reference `<https://arxiv.org/pdf/2003.00982.pdf>`_

    Statistics:

    - Train examples: 10,000
    - Valid examples: 1,000
    - Test examples: 1,000
    - Number of classes for each node: 6

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
    >>> from dgl.data import CLUSTERDataset
    >>>
    >>> trainset = CLUSTERDataset(mode='train')
    >>>
    >>> trainset.num_classes
    6
    >>> len(trainset)
    10000
    >>> trainset[0]
    Graph(num_nodes=117, num_edges=4104,
          ndata_schemes={'label': Scheme(shape=(), dtype=torch.int16),
                         'feat': Scheme(shape=(), dtype=torch.int64)}
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
        self._url = _get_dgl_url("dataset/SBM_CLUSTER.zip")
        self.mode = mode

        super(CLUSTERDataset, self).__init__(
            name="cluster",
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        self.load()

    def has_cache(self):
        graph_path = os.path.join(
            self.save_path, "CLUSTER_{}.bin".format(self.mode)
        )
        return os.path.exists(graph_path)

    def load(self):
        graph_path = os.path.join(
            self.save_path, "CLUSTER_{}.bin".format(self.mode)
        )
        self._graphs, _ = load_graphs(graph_path)

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 6

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
