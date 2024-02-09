"""QM9 dataset for graph property prediction (regression)."""
import os

import dgl

import numpy as np
import scipy.sparse as sp
import torch
from dgl.convert import graph as dgl_graph
from dgl.data import QM9Dataset
from dgl.data.utils import load_graphs, save_graphs
from tqdm import trange


class QM9(QM9Dataset):
    r"""QM9 dataset for graph property prediction (regression)

    This dataset consists of 130,831 molecules with 12 regression targets.
    Nodes correspond to atoms and edges correspond to bonds.

    Reference:

    - `"Quantum-Machine.org" <http://quantum-machine.org/datasets/>`_
    - `"Directional Message Passing for Molecular Graphs" <https://arxiv.org/abs/2003.03123>`_

    Statistics:

    - Number of graphs: 130,831
    - Number of regression targets: 12

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

    Parameters
    ----------
    label_keys: list
        Names of the regression property, which should be a subset of the keys in the table above.
    edge_funcs: list
        A list of edge-wise user-defined functions <https://docs.dgl.ai/en/0.6.x/api/python/udf.html#edge-wise-user-defined-function> for chemical bonds. Default: None
    cutoff: float
        Cutoff distance for interatomic interactions, i.e. two atoms are connected in the corresponding graph if the distance between them is no larger than this.
        Default: 5.0 Angstrom
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True

    Attributes
    ----------
    num_labels : int
        Number of labels for each graph, i.e. number of prediction tasks

    Raises
    ------
    UserWarning
        If the raw data is changed in the remote server by the author.

    Examples
    --------
    >>> data = QM9Dataset(label_keys=['mu', 'gap'], cutoff=5.0)
    >>> data.num_classes
    2
    >>>
    >>> # iterate over the dataset
    >>> for g, label in data:
    ...     R = g.ndata['R'] # get coordinates of each atom
    ...     Z = g.ndata['Z'] # get atomic numbers of each atom
    ...     # your code here...
    >>>
    """

    def __init__(
        self,
        label_keys,
        edge_funcs=None,
        cutoff=5.0,
        raw_dir=None,
        force_reload=False,
        verbose=False,
    ):
        self.edge_funcs = edge_funcs
        self._keys = [
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
        ]

        super(QM9, self).__init__(
            label_keys=label_keys,
            cutoff=cutoff,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    @property
    def graph_path(self):
        return f"{self.save_path}/dgl_graph.bin"

    @property
    def line_graph_path(self):
        return f"{self.save_path}/dgl_line_graph.bin"

    def has_cache(self):
        """step 1, if True, goto step 5; else goto download(step 2), then step 3"""
        return os.path.exists(self.graph_path) and os.path.exists(
            self.line_graph_path
        )

    def process(self):
        """step 3"""
        npz_path = f"{self.raw_dir}/qm9_eV.npz"
        data_dict = np.load(npz_path, allow_pickle=True)
        # data_dict['N'] contains the number of atoms in each molecule,
        # data_dict['R'] consists of the atomic coordinates,
        # data_dict['Z'] consists of the atomic numbers.
        # Atomic properties (Z and R) of all molecules are concatenated as single tensors,
        # so you need this value to select the correct atoms for each molecule.
        self.N = data_dict["N"]
        self.R = data_dict["R"]
        self.Z = data_dict["Z"]
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        # graph labels
        self.label_dict = {}
        for k in self._keys:
            self.label_dict[k] = torch.tensor(data_dict[k], dtype=torch.float32)

        self.label = torch.stack(
            [self.label_dict[key] for key in self.label_keys], dim=1
        )
        # graphs & features
        self.graphs, self.line_graphs = self._load_graph()

    def _load_graph(self):
        num_graphs = self.label.shape[0]
        graphs = []
        line_graphs = []

        for idx in trange(num_graphs):
            n_atoms = self.N[idx]
            # get all the atomic coordinates of the idx-th molecular graph
            R = self.R[self.N_cumsum[idx] : self.N_cumsum[idx + 1]]
            # calculate the distance between all atoms
            dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            # keep all edges that don't exceed the cutoff and delete self-loops
            adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(
                n_atoms, dtype=np.bool_
            )
            adj = adj.tocoo()
            u, v = torch.tensor(adj.row), torch.tensor(adj.col)
            g = dgl_graph((u, v))
            g.ndata["R"] = torch.tensor(R, dtype=torch.float32)
            g.ndata["Z"] = torch.tensor(
                self.Z[self.N_cumsum[idx] : self.N_cumsum[idx + 1]],
                dtype=torch.long,
            )

            # add user-defined features
            if self.edge_funcs is not None:
                for func in self.edge_funcs:
                    g.apply_edges(func)

            graphs.append(g)
            l_g = dgl.line_graph(g, backtracking=False)
            line_graphs.append(l_g)

        return graphs, line_graphs

    def save(self):
        """step 4"""
        save_graphs(str(self.graph_path), self.graphs, self.label_dict)
        save_graphs(str(self.line_graph_path), self.line_graphs)

    def load(self):
        """step 5"""
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.line_graphs, _ = load_graphs(self.line_graph_path)
        self.label = torch.stack(
            [label_dict[key] for key in self.label_keys], dim=1
        )

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
            - ``ndata['R']``: the coordinates of each atom
            - ``ndata['Z']``: the atomic number
        Tensor
            Property values of molecular graphs
        """
        return self.graphs[idx], self.line_graphs[idx], self.label[idx]
