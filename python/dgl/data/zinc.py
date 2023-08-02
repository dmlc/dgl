import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, load_graphs


class ZINCDataset(DGLBuiltinDataset):
    r"""ZINC dataset for the graph regression task.

    A subset (12K) of ZINC molecular graphs (250K) dataset is used to
    regress a molecular property known as the constrained solubility.
    For each molecular graph, the node features are the types of heavy
    atoms, between which the edge features are the types of bonds.
    Each graph contains 9-37 nodes and 16-84 edges.

    Reference `<https://arxiv.org/pdf/2003.00982.pdf>`_

    Statistics:

    Train examples: 10,000
    Valid examples: 1,000
    Test examples: 1,000
    Average number of nodes: 23.16
    Average number of edges: 39.83
    Number of atom types: 28
    Number of bond types: 4

    Parameters
    ----------
    mode : str, optional
        Should be chosen from ["train", "valid", "test"]
        Default: "train".
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: "~/.dgl/".
    force_reload : bool
        Whether to reload the dataset.
        Default: False.
    verbose : bool
        Whether to print out progress information.
        Default: False.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_atom_types : int
        Number of atom types.
    num_bond_types : int
        Number of bond types.

    Examples
    ---------
    >>> from dgl.data import ZINCDataset

    >>> training_set = ZINCDataset(mode="train")
    >>> training_set.num_atom_types
    28
    >>> len(training_set)
    10000
    >>> graph, label = training_set[0]
    >>> graph
    Graph(num_nodes=29, num_edges=64,
        ndata_schemes={'feat': Scheme(shape=(), dtype=torch.int64)}
        edata_schemes={'feat': Scheme(shape=(), dtype=torch.int64)})
    """

    def __init__(
        self,
        mode="train",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        self._url = _get_dgl_url("dataset/ZINC12k.zip")
        self.mode = mode

        super(ZINCDataset, self).__init__(
            name="zinc",
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        self.load()

    @property
    def graph_path(self):
        return os.path.join(self.save_path, "ZincDGL_{}.bin".format(self.mode))

    def has_cache(self):
        return os.path.exists(self.graph_path)

    def load(self):
        self._graphs, self._labels = load_graphs(self.graph_path)

    @property
    def num_atom_types(self):
        return 28

    @property
    def num_bond_types(self):
        return 4

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        r"""Get one example by index.

        Parameters
        ----------
        idx : int
            The sample index.

        Returns
        -------
        dgl.DGLGraph
            Each graph contains:

            - ``ndata['feat']``: Types of heavy atoms as node features
            - ``edata['feat']``: Types of bonds as edge features

        Tensor
            Constrained solubility as graph label
        """
        labels = self._labels["g_label"]
        if self._transform is None:
            return self._graphs[idx], labels[idx]
        else:
            return self._transform(self._graphs[idx]), labels[idx]
