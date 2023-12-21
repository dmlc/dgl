"""QM7b dataset for graph property prediction (regression)."""
import os

from scipy import io

from .. import backend as F
from ..convert import graph as dgl_graph

from .dgl_dataset import DGLDataset
from .utils import check_sha1, download, load_graphs, save_graphs


class QM7bDataset(DGLDataset):
    r"""QM7b dataset for graph property prediction (regression)

    This dataset consists of 7,211 molecules with 14 regression targets.
    Nodes means atoms and edges means bonds. Edge data 'h' means
    the entry of Coulomb matrix.

    Reference: `<http://quantum-machine.org/datasets/>`_

    Statistics:

    - Number of graphs: 7,211
    - Number of regression targets: 14
    - Average number of nodes: 15
    - Average number of edges: 245
    - Edge feature size: 1

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_tasks : int
        Number of prediction tasks
    num_labels : int
        (DEPRECATED, use num_tasks instead) Number of prediction tasks

    Raises
    ------
    UserWarning
        If the raw data is changed in the remote server by the author.

    Examples
    --------
    >>> data = QM7bDataset()
    >>> data.num_tasks
    14
    >>>
    >>> # iterate over the dataset
    >>> for g, label in data:
    ...     edge_feat = g.edata['h']  # get edge feature
    ...     # your code here...
    ...
    >>>
    """

    _url = (
        "http://deepchem.io.s3-website-us-west-1.amazonaws.com/"
        "datasets/qm7b.mat"
    )
    _sha1_str = "4102c744bb9d6fd7b40ac67a300e49cd87e28392"

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=False, transform=None
    ):
        super(QM7bDataset, self).__init__(
            name="qm7b",
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        mat_path = os.path.join(self.raw_dir, self.name + ".mat")
        self.graphs, self.label = self._load_graph(mat_path)

    def _load_graph(self, filename):
        data = io.loadmat(filename)
        labels = F.tensor(data["T"], dtype=F.data_type_dict["float32"])
        feats = data["X"]
        num_graphs = labels.shape[0]
        graphs = []
        for i in range(num_graphs):
            edge_list = feats[i].nonzero()
            g = dgl_graph(edge_list)
            g.edata["h"] = F.tensor(
                feats[i][edge_list[0], edge_list[1]].reshape(-1, 1),
                dtype=F.data_type_dict["float32"],
            )
            graphs.append(g)
        return graphs, labels

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        save_graphs(str(graph_path), self.graphs, {"labels": self.label})

    def has_cache(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        return os.path.exists(graph_path)

    def load(self):
        graphs, label_dict = load_graphs(
            os.path.join(self.save_path, "dgl_graph.bin")
        )
        self.graphs = graphs
        self.label = label_dict["labels"]

    def download(self):
        file_path = os.path.join(self.raw_dir, self.name + ".mat")
        download(self.url, path=file_path)
        if not check_sha1(file_path, self._sha1_str):
            raise UserWarning(
                "File {} is downloaded but the content hash does not match."
                "The repo may be outdated or download may be incomplete. "
                "Otherwise you can create an issue for it.".format(self.name)
            )

    @property
    def num_tasks(self):
        """Number of prediction tasks."""
        return self.num_labels

    @property
    def num_labels(self):
        """Number of prediction tasks."""
        return 14

    @property
    def num_classes(self):
        """Number of prediction tasks."""
        return 14

    def __getitem__(self, idx):
        r"""Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
        """
        if self._transform is None:
            g = self.graphs[idx]
        else:
            g = self._transform(self.graphs[idx])
        return g, self.label[idx]

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return len(self.graphs)


QM7b = QM7bDataset
