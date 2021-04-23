"""QM9 dataset for graph property prediction (regression)."""
import os
import numpy as np
import scipy.sparse as sp

from .dgl_dataset import DGLDataset
from .utils import download, _get_dgl_url
from ..convert import graph as dgl_graph
from ..transform import to_bidirected
from .. import backend as F

class QM9Dataset(DGLDataset):
    r"""QM9 dataset for graph property prediction (regression)

    This dataset consists of 130,831 molecules with 12 regression targets.
    Nodes correspond to atoms and edges correspond to close atom pairs.

    This dataset differs from :class:`~dgl.data.QM9EdgeDataset` in the following aspects:
        1. Edges in this dataset are purely distance-based.
        2. It only provides atoms' coordinates and atomic numbers as node features
        3. It only provides 12 regression targets.

    Reference: 
    
    - `"Quantum-Machine.org" <http://quantum-machine.org/datasets/>`_,
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
    cutoff: float
        Cutoff distance for interatomic interactions, i.e. two atoms are connected in the corresponding graph if the distance between them is no larger than this.
        Default: 5.0 Angstrom
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

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
    >>> data.num_labels
    2
    >>>
    >>> # iterate over the dataset
    >>> for g, label in data:
    ...     R = g.ndata['R'] # get coordinates of each atom
    ...     Z = g.ndata['Z'] # get atomic numbers of each atom
    ...     # your code here...
    >>>
    """

    def __init__(self,
                 label_keys,
                 cutoff=5.0,
                 raw_dir=None,
                 force_reload=False,
                 verbose=False):
    
        self.cutoff = cutoff
        self.label_keys = label_keys
        self._url = _get_dgl_url('dataset/qm9_eV.npz')

        super(QM9Dataset, self).__init__(name='qm9',
                                         url=self._url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        npz_path = f'{self.raw_dir}/qm9_eV.npz'
        data_dict = np.load(npz_path, allow_pickle=True)
        # data_dict['N'] contains the number of atoms in each molecule.
        # Atomic properties (Z and R) of all molecules are concatenated as single tensors,
        # so you need this value to select the correct atoms for each molecule.
        self.N = data_dict['N']
        self.R = data_dict['R']
        self.Z = data_dict['Z']
        self.label = np.stack([data_dict[key] for key in self.label_keys], axis=1)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

    def download(self):
        file_path = f'{self.raw_dir}/qm9_eV.npz'
        if not os.path.exists(file_path):
            download(self._url, path=file_path)

    @property
    def num_labels(self):
        r"""
        Returns
        --------
        int
            Number of labels for each graph, i.e. number of prediction tasks.
        """
        return self.label.shape[1]

    def __getitem__(self, idx):
        r""" Get graph and label by index

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
        label = F.tensor(self.label[idx], dtype=F.data_type_dict['float32'])
        n_atoms = self.N[idx]
        R = self.R[self.N_cumsum[idx]:self.N_cumsum[idx + 1]]
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(n_atoms, dtype=np.bool)
        adj = adj.tocoo()
        u, v = F.tensor(adj.row), F.tensor(adj.col)
        g = dgl_graph((u, v))
        g = to_bidirected(g)
        g.ndata['R'] = F.tensor(R, dtype=F.data_type_dict['float32'])
        g.ndata['Z'] = F.tensor(self.Z[self.N_cumsum[idx]:self.N_cumsum[idx + 1]], 
                                dtype=F.data_type_dict['int64'])
        return g, label

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return self.label.shape[0]

QM9 = QM9Dataset