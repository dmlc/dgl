"""QM9 dataset for graph property prediction (regression)."""
import os
import numpy as np
import scipy.sparse as sp
import dgl
import torch

from dgl.data import DGLDataset
from dgl.data.utils import download, load_graphs, save_graphs

class QM9Dataset(DGLDataset):
    r"""QM9 dataset for graph property prediction (regression)

    This dataset consists of 13,0831 molecules with 12 regression targets.
    - id: [0, 1, ..., 133884], a unique identifier for each molecule.
    - R: [2358210, 3], the coordinates of each atom.
    - Z: [2358210, ], the atomic number.
    - N: [130831, ], the number of atoms in each molecule. Atomic properties like Z and R are just concatenated globally, so you need this value to select the correct atoms for each molecule.

    Statistics:

    - Number of graphs: 13,0831
    - Number of regression targets: 12

    Parameters
    ----------
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
    >>> data = QM9Dataset()
    >>> data.num_labels
    12
    >>>
    >>> # iterate over the dataset
    >>> for g, label in data:
    ...     # get feature
    ...     # your code here...
    >>>
    """

    _url = 'qm9_eV.npz'

    def __init__(self, cutoff, label_keys, raw_dir=None, force_reload=False, verbose=False):
        self.cutoff = cutoff
        self.label_keys = label_keys

        super(QM9Dataset, self).__init__(name='qm9',
                                         url=self._url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        npz_path = f'{self.raw_dir}/{self._url}'
        data_dict = np.load(npz_path, allow_pickle=True)
        self.N = data_dict['N']
        self.R = data_dict['R']
        self.Z = data_dict['Z']
        self.label = np.stack([data_dict[key] for key in self.label_keys], axis=1)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])

    def download(self):
        file_path = f'{self.raw_dir}/{self._url}'
        if not os.path.exists(file_path):
            download(self._url, path=file_path)

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return self.label.shape[1]

    def __getitem__(self, idx):
        r""" Get graph and label by index
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
        """
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]
        
        labels = torch.tensor(self.label[idx], dtype=torch.float32)
        graphs = []
        for i in idx:
            n_atoms = self.N[i]
            R = self.R[self.N_cumsum[i]:self.N_cumsum[i + 1]]
            dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(n_atoms, dtype=np.bool)
            adj = adj.tocoo()
            u, v = torch.tensor(adj.row), torch.tensor(adj.col)
            g = dgl.graph((u, v))
            g = dgl.to_bidirected(g)
            g.ndata['R'] = torch.tensor(R, dtype=torch.float32)
            g.ndata['Z'] = torch.tensor(self.Z[self.N_cumsum[i]:self.N_cumsum[i + 1]], dtype=torch.int)
            graphs.append(g)
        return graphs, labels

    def __len__(self):
        r"""Number of graphs in the dataset.
        Return
        -------
        int
        """
        return self.label.shape[0]

if __name__ == "__main__":
    QM9 = QM9Dataset(cutoff=5, label_keys=['mu', 'homo'], raw_dir='./')
    print(QM9[[1, 2, 3, 4, 5]])