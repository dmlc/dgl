""" QM9 dataset for graph property prediction (regression) """

import os

import numpy as np

from .. import backend as F
from ..convert import graph as dgl_graph

from .dgl_dataset import DGLDataset
from .utils import _get_dgl_url, download, extract_archive


class QM9EdgeDataset(DGLDataset):
    r"""QM9Edge dataset for graph property prediction (regression)

    This dataset consists of 130,831 molecules with 19 regression targets.
    Nodes correspond to atoms and edges correspond to bonds.

    This dataset differs from :class:`~dgl.data.QM9Dataset` in the following aspects:
        1. It includes the bonds in a molecule in the edges of the corresponding graph while the edges in :class:`~dgl.data.QM9Dataset` are purely distance-based.
        2. It provides edge features, and node features in addition to the atoms' coordinates and atomic numbers.
        3. It provides another 7 regression tasks(from 12 to 19).

    This class is built based on a preprocessed version of the dataset, and we provide the preprocessing datails `here <https://gist.github.com/hengruizhang98/a2da30213b2356fff18b25385c9d3cd2>`_.

    Reference:

    - `"MoleculeNet: A Benchmark for Molecular Machine Learning" <https://arxiv.org/abs/1703.00564>`_
    - `"Neural Message Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_

    For
    Statistics:

    - Number of graphs: 130,831.
    - Number of regression targets: 19.

    Node attributes:

    - pos: the 3D coordinates of each atom.
    - attr: the 11D atom features.

    Edge attributes:

    - edge_attr: the 4D bond features.

    Regression targets:

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Keys   | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | mu     | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | alpha  | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | homo   | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | lumo   | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | gap    | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | r2     | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | zpve   | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | U0     | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | U      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | H      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | G      | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Cv     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | U0_atom| :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | U_atom | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | H_atom | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | G_atom | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | A      | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | B      | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | C      | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Parameters
    ----------
    label_keys : list
        Names of the regression property, which should be a subset of the keys in the table above.
        If not provided, it will load all the labels.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False.
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
    >>> data = QM9EdgeDataset(label_keys=['mu', 'alpha'])
    >>> data.num_tasks
    2

    >>> # iterate over the dataset
    >>> for graph, labels in data:
    ...     print(graph) # get information of each graph
    ...     print(labels) # get labels of the corresponding graph
    ...     # your code here...
    >>>
    """

    keys = [
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
        "U0_atom",
        "U_atom",
        "H_atom",
        "G_atom",
        "A",
        "B",
        "C",
    ]
    map_dict = {}

    for i, key in enumerate(keys):
        map_dict[key] = i

    def __init__(
        self,
        label_keys=None,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None,
    ):
        if label_keys is None:
            self.label_keys = None
            self.num_labels = 19
        else:
            self.label_keys = [self.map_dict[i] for i in label_keys]
            self.num_labels = len(label_keys)

        self._url = _get_dgl_url("dataset/qm9_edge.npz")

        super(QM9EdgeDataset, self).__init__(
            name="qm9Edge",
            raw_dir=raw_dir,
            url=self._url,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        if not os.path.exists(self.npz_path):
            download(self._url, path=self.npz_path)

    def process(self):
        self.load()

    @property
    def npz_path(self):
        return f"{self.raw_dir}/qm9_edge.npz"

    def has_cache(self):
        return os.path.exists(self.npz_path)

    def save(self):
        np.savez_compressed(
            self.npz_path,
            n_node=self.n_node,
            n_edge=self.n_edge,
            node_attr=self.node_attr,
            node_pos=self.node_pos,
            edge_attr=self.edge_attr,
            src=self.src,
            dst=self.dst,
            targets=self.targets,
        )

    def load(self):
        data_dict = np.load(self.npz_path, allow_pickle=True)

        self.n_node = data_dict["n_node"]
        self.n_edge = data_dict["n_edge"]
        self.node_attr = data_dict["node_attr"]
        self.node_pos = data_dict["node_pos"]
        self.edge_attr = data_dict["edge_attr"]
        self.targets = data_dict["targets"]

        self.src = data_dict["src"]
        self.dst = data_dict["dst"]

        self.n_cumsum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.ne_cumsum = np.concatenate([[0], np.cumsum(self.n_edge)])

    def __getitem__(self, idx):
        r"""Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        dgl.DGLGraph
           The graph contains:

           - ``ndata['pos']``: the coordinates of each atom
           - ``ndata['attr']``: the features of each atom
           - ``edata['edge_attr']``: the features of each bond

        Tensor
            Property values of molecular graphs
        """

        pos = self.node_pos[self.n_cumsum[idx] : self.n_cumsum[idx + 1]]
        src = self.src[self.ne_cumsum[idx] : self.ne_cumsum[idx + 1]]
        dst = self.dst[self.ne_cumsum[idx] : self.ne_cumsum[idx + 1]]

        g = dgl_graph((src, dst))

        g.ndata["pos"] = F.tensor(pos, dtype=F.data_type_dict["float32"])
        g.ndata["attr"] = F.tensor(
            self.node_attr[self.n_cumsum[idx] : self.n_cumsum[idx + 1]],
            dtype=F.data_type_dict["float32"],
        )
        g.edata["edge_attr"] = F.tensor(
            self.edge_attr[self.ne_cumsum[idx] : self.ne_cumsum[idx + 1]],
            dtype=F.data_type_dict["float32"],
        )

        label = F.tensor(
            self.targets[idx][self.label_keys],
            dtype=F.data_type_dict["float32"],
        )

        if self._transform is not None:
            g = self._transform(g)

        return g, label

    def __len__(self):
        r"""Number of graphs in the dataset.

        Returns
        -------
        int
        """
        return self.n_node.shape[0]

    @property
    def num_tasks(self):
        r"""
        Returns
        -------
        int
            Number of prediction tasks
        """
        return self.num_labels


QM9Edge = QM9EdgeDataset
