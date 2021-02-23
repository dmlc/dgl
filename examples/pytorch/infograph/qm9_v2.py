import numpy as np
import os
from tqdm import tqdm

import torch as th

import dgl
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import download, load_graphs, _get_dgl_url, extract_archive

class QM9Dataset_v2(DGLDataset):
    r"""QM9 dataset for graph property prediction (regression)
    This dataset consists of 13,0831 molecules with 19 regression targets.
    Node means atom and edge means bond.
    
    Reference: `"MoleculeNet: A Benchmark for Molecular Machine Learning" <https://arxiv.org/abs/1703.00564>`_
               Atom features come from `"Neural Message Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_
    
    Statistics:

    - Number of graphs: 13,0831
    - Number of regression targets: 19

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
    | c      | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+----------------------------------------
    Parameters
    ----------
    label_keys: list
        Names of the regression property, which should be a subset of the keys in the table above.
        If not provided, will load all the labels.
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
    >>> data = QM9Dataset_v2(label_keys=['mu', 'alpha'])
    >>> data.num_labels
    
    
    >>> # make each graph dense
    >>> data.to_dense()
    
    >>> # iterate over the dataset
    >>> for graph, labels in data:
    ...     print(graph) # get information of each graph
    ...     print(labels) # get labels of the corresponding graph
    ...     # your code here...
    >>>
    
    """
    def __init__(self, 
                 label_keys = None,
                 raw_dir=None, 
                 force_reload=False, 
                 verbose=True):
        
        self.label_keys = label_keys
        self._url = _get_dgl_url('dataset/qm9_ver2.zip')
        super(QM9Dataset_v2, self).__init__(name='qm9_v2',
                                            url=self._url,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)
        
    def process(self):
        print('begin loading dataset')
        graphs, label_dict = load_graphs(os.path.join(self.raw_dir, 'qm9_v2.bin'))
        self.graphs = graphs
        if self.label_keys == None:
            self.labels = np.stack([label_dict[key] for key in label_dict.keys()], axis=1)
        else:
            self.labels = np.stack([label_dict[key] for key in self.label_keys], axis=1)

    def to_dense(self):
        r""" Transfrom each graph to a dense graph and add additional edge attribute(distance between two atoms)
        
        Note: This operation will deprecate graph.ndata['pos']
        
        """
        n_graph = self.labels.shape[0]
        for id in tqdm(range(n_graph), desc = 'processing graphs'):
            graph = self.graphs[id]
            n_nodes = graph.num_nodes()
            row = th.arange(n_nodes, dtype = th.long)
            col = th.arange(n_nodes, dtype = th.long)

            row = row.view(-1,1).repeat(1, n_nodes).view(-1)
            col = col.repeat(n_nodes)

            src = graph.edges()[0]
            dst = graph.edges()[1]

            idx = src * n_nodes + dst
            size = list(graph.edata['edge_attr'].size())
            size[0] = n_nodes * n_nodes
            edge_attr = graph.edata['edge_attr'].new_zeros(size)
            
            edge_attr[idx] = graph.edata['edge_attr']
            
            pos = graph.ndata['pos']
            dist = th.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
            
            new_edge_attr =  th.cat([edge_attr, dist.type_as(edge_attr)], dim = -1)
            
            new_graph = dgl.graph((row,col))
            
            new_graph.ndata['attr'] = graph.ndata['attr']
            new_graph.edata['edge_attr'] = new_edge_attr
            new_graph = new_graph.remove_self_loop()
            
            self.graphs[id] = new_graph

    def download(self):
        file_path = f'{self.raw_dir}/qm9_v2.zip'
        if not os.path.exists(file_path):
            download(self._url, path=file_path)
            extract_archive(file_path, self.raw_dir, overwrite = True)

    @property
    def num_labels(self):
        r"""
        Returns
        --------
        int
            Number of labels for each graph, i.e. number of prediction tasks.
        """
        return self.labels.shape[1]

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

            - ``ndata['pos']``: the coordinates of each atom
            - ``ndata['attr']``: the atomic attributes
            - ``edata['edge_attr']``: the bond attributes

        Tensor
            Property values of molecular graphs
        """
        
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return self.labels.shape[0]