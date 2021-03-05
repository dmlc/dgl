""" 
    QM9 dataset for graph property prediction (regression).
    When building this class, we partially refer to the implementation from https://github.com/Jack-XHP/DGL_QM9EDGE  
"""

import os
import numpy as np

from .dgl_dataset import DGLDataset
from .utils import download, extract_archive, _get_dgl_url
from ..convert import graph as dgl_graph
from .. import backend as F

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    rdkit = None

HAR2EV = 27.2113825435      # 1 Hartree = 27.2114 eV 
KCALMOL2EV = 0.04336414     # 1 kcal/mol = 0.043363 eV
conversion = F.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class QM9EdgeDataset(DGLDataset):
    r"""QM9Edge dataset for graph property prediction (regression)
    This dataset consists of 130,831 molecules with 19 regression targets.
    Nodes correspond to atoms and edges correspond to bonds.
    
    This dataset differs from :class:`~dgl.data.QM9Dataset` in the following points:
        1. It includes the bonds in a molecule in the edges of the corresponding graph while the edges in :class:`~dgl.data.QM9Dataset` are purely distance-based.
        2. It provides edge features, and node features in addition to the atom's locations and atomic number.
        3. The number of regression targets is expanded from 13 to 19 (in :class:`~dgl.data.QM9Dataset`, it's 13).

    Reference: `"MoleculeNet: A Benchmark for Molecular Machine Learning" <https://arxiv.org/abs/1703.00564>`_
               Atom features come from `"Neural Message Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_
    
    Statistics:

    - Number of graphs: 130,831
    - Number of regression targets: 19
    
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
    +--------+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------
    
    Parameters
    ----------
    label_keys: list
        Names of the regression property, which should be a subset of the keys in the table above.
        If not provided, it will load all the labels.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False.
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
    >>> data = QM9EdgeDataset(label_keys=['mu', 'alpha'])
    >>> data.num_labels
    2
    
    >>> # iterate over the dataset
    >>> for graph, labels in data:
    ...     print(graph) # get information of each graph
    ...     print(labels) # get labels of the corresponding graph
    ...     # your code here...
    >>>
    """
    
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    
    keys = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom',
            'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    map_dict = {}
    
    for i, key in enumerate(keys):
        map_dict[key] = i
        
    if rdkit is not None:
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        
    def __init__(self, 
                 label_keys=None,
                 raw_dir=None, 
                 force_reload=False, 
                 verbose=True):
         
        if label_keys == None:
            self.label_keys = None
            self.num_labels = 19
        else:
            self.label_keys = [self.map_dict[i] for i in label_keys]
            self.num_labels = len(label_keys)
            
                
        self._url = _get_dgl_url('dataset/qm9_edge.npz')
        1
        super(QM9EdgeDataset, self).__init__(name='qm9Edge',
                                             raw_dir=raw_dir,
                                             url=self._url,
                                             force_reload=force_reload,
                                             verbose=verbose)
    
    def download(self):
        
        if rdkit is None:
            print('Using a pre-processed version of the dataset.'
                  'Please install `rdkit` to alternatively process the raw data.')
            
            file_path = f'{self.raw_dir}/qm9_edge.npz'
            if not os.path.exists(file_path):
                download(self._url, path=file_path)
    
        else:
            if not os.path.exists(f'{self.raw_dir}/gdb9.sdf.csv'):
                file_path = download(self.raw_url, self.raw_dir)
                extract_archive(file_path, self.raw_dir, overwrite=True)
                os.unlink(file_path)

            if not os.path.exists(f'{self.raw_url2}/uncharacterized.txt'):
                file_path = download(self.raw_url2, self.raw_dir)
                os.replace(os.path.join(self.raw_dir, '3195404'), os.path.join(self.raw_dir, 'uncharacterized.txt'))
    
    def process(self):
        if rdkit is None:
            print('loading downloaded files')
            npz_path = os.path.join(self.raw_dir, "qm9_edge.npz")
        
            data_dict = np.load(npz_path, allow_pickle=True)

            self.N_node = data_dict['N_node']
            self.N_edge = data_dict['N_edge']
            self.Node_attr = data_dict['Node_attr']
            self.Node_pos = data_dict['Node_pos']
            self.Edge_attr = data_dict['Edge_attr']
            self.Target = data_dict['Target']
            
            self.src = data_dict['src']
            self.dst = data_dict['dst']
            
            self.N_cumsum = np.concatenate([[0], np.cumsum(self.N_node)])
            self.NE_cumsum = np.concatenate([[0], np.cumsum(self.N_edge)])
        else:
            with open(os.path.join(self.raw_dir, "gdb9.sdf.csv"), 'r') as f:
                target = f.read().split('\n')[1:-1]
                target = [[float(x) for x in line.split(',')[1:20]] for line in target]
                target = F.tensor(target, dtype=F.data_type_dict['float32'])
                target = F.cat([target[:, 3:], target[:, :3]], dim=-1)
                target = (target * conversion.view(1, -1)).tolist()

            with open(os.path.join(self.raw_dir, "uncharacterized.txt"), 'r') as f:
                skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

            suppl = Chem.SDMolSupplier(os.path.join(self.raw_dir, "gdb9.sdf"), removeHs=False, sanitize=False)    
        
            n_node = []
            n_edge = []
            node_pos = []
            node_attr = []

            src = []
            dst = []
            
            edge_attr = []
            targets = []
            
            print('Loading graphs:')
            for i, mol in enumerate(suppl):
                if i in skip:
                    continue
                    
                N = mol.GetNumAtoms()

                pos = suppl.GetItemText(i).split('\n')[4:4 + N]
                pos = [[float(x) for x in line.split()[:3]] for line in pos]

                type_idx = []
                atomic_number = []
                aromatic = []
                sp = []
                sp2 = []
                sp3 = []
                
                for atom in mol.GetAtoms():
                    type_idx.append(self.types[atom.GetSymbol()])
                    atomic_number.append(atom.GetAtomicNum())
                    aromatic.append(1 if atom.GetIsAromatic() else 0)
                    hybridization = atom.GetHybridization()
                    sp.append(1 if hybridization == HybridizationType.SP else 0)
                    sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                    sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

                z = F.tensor(atomic_number, dtype=F.data_type_dict['int64'])

                row, col, edge_type = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [self.bonds[bond.GetBondType()]]

                edge_index = F.tensor([row, col], dtype=F.data_type_dict['int64'])
                edge_type = F.tensor(edge_type, dtype=F.data_type_dict['int64'])
                edge_attr = np.eye(len(self.bonds))[edge_type]

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]
                
                row, col = edge_index
                
                hs = (z == 1).to(F.data_type_dict['float32'])
                out = F.zeros([int(col.max())+1], dtype=hs[row].dtype, ctx= F.cpu())
                num_hs = np.array(out.scatter_add(-1, col, hs[row]))

                row = np.array(row)
                col = np.array(col)

                x1 = np.eye(len(self.types))[type_idx]
                x2 = np.array([atomic_number, aromatic, sp, sp2, sp3, num_hs]).transpose()
                x = np.concatenate((x1,x2), axis = 1)
                
                n_node.append(N)
                n_edge.append(mol.GetNumBonds() * 2)

                node_pos.append(np.array(pos))
                node_attr.append(x)
                
                src += list(row)
                dst += list(col)
                edge_attr.append(edge_attr)   
                targets.append(np.array(target[i]).reshape([1,19]))
       
            node_attr = np.concatenate(node_attr, axis = 0)
            node_pos = np.concatenate(node_pos, axis = 0)
            edge_attr = np.concatenate(edge_attr, axis = 0)
            targets = np.concatenate(targets, axis = 0)
            
            self.n_node = n_node
            self.n_edge = n_edge
            self.node_attr = node_attr
            self.node_pos = node_pos
            self.edge_attr = edge_attr
            self.targets = targets
            
            self.src = src
            self.dst = dst

            self.n_cumsum = np.concatenate([[0], np.cumsum(self.n_node)])
            self.ne_cumsum = np.concatenate([[0], np.cumsum(self.n_edge)])
        
    def has_cache(self):
        npz_path = os.path.join(self.raw_dir, "qm9_edge.npz")
        return os.path.exists(npz_path)
    
    def save(self):
        np.savez_compressed(os.path.join(self.raw_dir, "qm9_edge.npz"), 
                            n_node=self.n_node,
                            n_edge=self.n_edge,
                            node_attr=self.node_attr,
                            node_pos=self.node_pos,
                            edge_attr=self.edge_attr,
                            src=self.src,
                            dst=self.dst,  
                            targets=self.targets)
    def load(self):
    
        npz_path = os.path.join(self.raw_dir, "qm9_edge.npz")
        data_dict = np.load(npz_path, allow_pickle=True)

        self.n_node = data_dict['n_node']
        self.n_edge = data_dict['n_edge']
        self.node_attr = data_dict['node_attr']
        self.node_pos = data_dict['node_pos']
        self.edge_attr = data_dict['edge_attr']
        self.targets = data_dict['targets']
        
        self.src = data_dict['src']
        self.dst = data_dict['dst']
        
        self.n_cumsum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.ne_cumsum = np.concatenate([[0], np.cumsum(self.n_edge)])
    
    def __getitem__(self, idx):
        r""" Get graph and label by index 
        
        Parameters
        ----------
        idx : int
            Item index
            
        Returns
        -------
        dgl.DGLGraph
           The graph contains:11
           
           - ``ndata['pos']``: the coordinates of each atom
           - ``ndata['attr']``: the features of each atom
           - ``edata['edge_attr']``: the features of each bond
           
        Tensor
            Property values of molecular graphs
        """
        
        pos = self.node_pos[self.n_cumsum[idx]:self.n_cumsum[idx+1]]
        src = self.src[self.ne_cumsum[idx]:self.ne_cumsum[idx+1]]
        dst = self.dst[self.ne_cumsum[idx]:self.ne_cumsum[idx+1]]

        g = dgl_graph((src, dst))
          
        g.ndata['pos'] = F.tensor(pos, dtype=F.data_type_dict['float32'])
        g.ndata['attr'] = F.tensor(self.node_attr[self.n_cumsum[idx]:self.n_cumsum[idx+1]], dtype=F.data_type_dict['float32'])
        g.edata['edge_attr'] = F.tensor(self.edge_attr[self.ne_cumsum[idx]:self.ne_cumsum[idx+1]], dtype=F.data_type_dict['float32'])
        
        
        label = F.tensor(self.targets[idx][self.label_keys], dtype=F.data_type_dict['float32'])
        
        return g, label

    def __len__(self):
        r""" Number of graphs in the dataset.
        
        Returns
        -------
        int
        """
        return self.n_node.shape[0]


QM9Edge = QM9EdgeDataset
