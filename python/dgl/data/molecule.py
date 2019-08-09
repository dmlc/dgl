import numpy as np
import os
import pickle
import torch
from functools import partial
from torch.utils.data import Dataset

import dgl
from dgl import DGLGraph
from .utils import download, get_download_dir, _get_dgl_url

try:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

__all__ = ['default_atom_featurizer', 'mol2dgl', 'Tox21']

_urls = {
    'tox21': 'dataset/tox21.csv.gz'
}

def one_of_k_encoding_unk(x, allowable_set):
    """One-hot encoding.

    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.

    Returns
    -------
    list
        List of boolean values where only one value is True.
        If the i-th value is True, then we must have
        x == allowable_set[i]
    """
    assert x in allowable_set, 'Failed to perform one-hot encoding as' \
                               'the input does not match any option'
    return list(map(lambda s: x == s, allowable_set))

def default_atom_featurizer(mol, atom_data_field='h'):
    """A default featurizer for atoms

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include 0 - 10.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include 0 - 6.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include 0 - 4.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to be 'h'

    Returns
    -------
    dict
        Atom features
    """
    def _featurize(atom):
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',
                      'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                      'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                      'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                      'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_types) + \
                  one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(),
                                        [Chem.rdchem.HybridizationType.SP,
                                         Chem.rdchem.HybridizationType.SP2,
                                         Chem.rdchem.HybridizationType.SP3,
                                         Chem.rdchem.HybridizationType.SP3D,
                                         Chem.rdchem.HybridizationType.SP3D2]) + \
                  [atom.GetIsAromatic()] + \
                  one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        return results

    num_atoms = mol.GetNumAtoms()
    atom_features = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_features.append(_featurize(atom))
    atom_features = np.stack(atom_features)
    atom_features = torch.from_numpy(atom_features).float()

    return {atom_data_field: atom_features}

def mol2dgl(mol, add_self_loop=False, atom_featurizer=None, bond_featurizer=None):
    """Convert RDKit molecule object into a DGLGraph

    The ith atom in the molecule, i.e. mol.GetAtomWithIdx(i), corresponds to the
    ith node in the returned DGLGraph.

    The ith bond in the molecule, i.e. mol.GetBondWithIdx(i), corresponds to the
    (2i)-th and (2i+1)-th edges in the returned DGLGraph. The (2i)-th and (2i+1)-th
    edges will be separately from u to v and v to u, where u is bond.GetBeginAtomIdx()
    and v is bond.GetEndAtomIdx().

    If self loops are added, the last n edges will separately be self loops for
    atoms 0, 1, ..., n-1.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance
    add_self_loop : bool
        Whether to add self loops in DGLGraphs
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    """
    # Graph construction
    g = DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    g.add_edges(src_list, dst_list)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    # Featurization
    if atom_featurizer is not None:
        g.ndata.update(atom_featurizer(mol))

    if bond_featurizer is not None:
        g.edata.update(bond_featurizer(mol))

    return g

class Tox21(object):
    def __init__(self, atom_data_field='h', frac_train=0.8, frac_val=0.1,
                 frac_test=0.1, add_self_loop=False):
        """
        The Toxicology in the 21st Century (https://tripod.nih.gov/tox21/challenge/)
        initiative created a public database measuring toxicity of compounds, which
        has been used in the 2014 Tox21 Data Challenge. The dataset contains qualitative
        toxicity measurements for 8014 compounds on 12 different targets, including nuclear
        receptors and stress response pathways. Each target results in a binary label.

        The dataset is randomly split into a training, validation and test set.

        A common issue for multi-task prediction is that some datapoints are not labeled for
        all tasks. This is also the case for Tox21. In data pre-processing, we set non-existing
        labels to be 0 so that they can be placed in tensors and masking them for loss computation.
        See examples below for more details.

        Parameters
        ----------
        atom_data_field : str
            Name for storing atom features in DGLGraphs, default to be 'h'
        frac_train : float
            Proportion of data to use for training, default to be 0.8
        frac_val : float
            Proportion of data to use for validation, default to be 0.1
        frac_test : float
            Proportion of data to use for test, default to be 0.1
        add_self_loop : bool
            Whether to add self loops in DGLGraphs, default to be False

        Examples
        --------
        >>> dataset = Tox21()
        >>> dataset.tasks
        ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        >>> dataset.add_self_loop
        False
        >>> dataset.atom_data_field
        'h'
        """
        super(Tox21, self).__init__()
        self.tasks = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
            'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        self.add_self_loop = add_self_loop
        self.atom_data_field = atom_data_field

        self.load_data()
        self.preprocess_data()
        self.weight_balancing()

        self.split_dataset(frac_train, frac_val, frac_test)

    @property
    def num_datapoints(self):
        """Dataset size

        Returns
        -------
        int
            Number of molecules in the dataset
        """
        return len(self.smiles)

    @property
    def num_tasks(self):
        """Number of tasks

        Returns
        -------
        int
            Number of tasks in the dataset
        """
        return len(self.tasks)

    @property
    def feat_size(self):
        """Atom feature size

        Returns
        -------
        int
            Atom feature size
        """
        return self.graphs[0].ndata[self.atom_data_field].shape[1]

    def load_data(self):
        self.data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(_urls['tox21']), path=self.data_path)
        print('Loading data from csv file...')
        df = pd.read_csv(self.data_path)
        # np.nan suggests non-existing labels, and we replace them by ""
        df = df.replace(np.nan, str(""), regex=True)
        self.df = df

    def preprocess_data(self):
        # Convert a smile to a dgl graph and featurize for it.
        self.smiles = self.df['smiles'].tolist()
        n_tasks = len(self.tasks)
        num_datapoints = self.num_datapoints

        path_to_dgl_graphs = 'dgl_graphs.pkl'
        if os.path.exists(path_to_dgl_graphs):
            print('Loading previously saved dgl graphs...')
            with open(path_to_dgl_graphs, 'rb') as f:
                graphs = pickle.load(f)
        else:
            graphs = []
            for id, s in enumerate(self.smiles):
                print('Processing smile {:d}/{:d}'.format(id + 1, num_datapoints))
                mol = Chem.MolFromSmiles(s)
                # Canonically index atoms in each molecule
                new_order = rdmolfiles.CanonicalRankAtoms(mol)
                mol = rdmolops.RenumberAtoms(mol, new_order)
                graphs.append(mol2dgl(
                    mol, atom_featurizer=partial(default_atom_featurizer, atom_data_field=self.atom_data_field)))

            with open(path_to_dgl_graphs, 'wb') as f:
                pickle.dump(graphs, f)

        self.graphs = graphs

        # Prepare labels
        labels = np.hstack(
            [np.reshape(np.array(self.df[task].values), (num_datapoints, 1)) for task in self.tasks])

        # Some data points do not have all labels and we want to exclude them for loss computation.
        w = np.ones((num_datapoints, n_tasks))
        missing = np.zeros_like(labels).astype(int)

        for i in range(num_datapoints):
            for task in range(n_tasks):
                if labels[i, task] == "":
                    missing[i, task] = 1

        for i in range(num_datapoints):
            for task in range(n_tasks):
                if missing[i, task]:
                    labels[i, task] = 0.
                    w[i, task] = 0.

        self.labels = labels.astype(float)
        self.w = w.astype(float)

    def weight_balancing(self):
        task_pos_weights = []
        n_tasks = len(self.tasks)

        for i in range(n_tasks):
            task_label = self.labels[:, i]
            task_w = self.w[:, i]
            task_label_ = task_label[task_w != 0]
            num_positives = np.count_nonzero(task_label_)
            num_negatives = len(task_label_) - num_positives

            if num_positives > 0:
                pos_weight = float(num_negatives) / num_positives
                pos_indices = np.logical_and(task_label == 1, task_w != 0)
                self.w[pos_indices, i] = pos_weight
                task_pos_weights.append(pos_weight)
            else:
                task_pos_weights.append(1)

        self.task_pos_weights = torch.tensor(task_pos_weights)

    def split_dataset(self, frac_train, frac_val, frac_test):
        frac_total = frac_train + frac_val + frac_test
        assert frac_total == 1, \
            'Expect frac_train + frac_val + frac_test = 1, got {:.4f}'.format(frac_total)
        num_datapoints = self.num_datapoints
        train_cutoff = int(frac_train * num_datapoints)
        val_cutoff = int((frac_train + frac_val) * num_datapoints)

        total_indices = range(num_datapoints)
        train_indices = total_indices[:train_cutoff]
        val_indices = total_indices[train_cutoff:val_cutoff]
        test_indices = total_indices[val_cutoff:]

        self.train_set = MolSubset(train_indices, self.graphs, self.labels, self.w)
        self.val_set = MolSubset(val_indices, self.graphs, self.labels, self.w)
        self.test_set = MolSubset(test_indices, self.graphs, self.labels, self.w)

    def collate(self, data):
        graphs, labels, weights = map(list, zip(*data))
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        return bg, torch.stack(labels, dim=0), torch.stack(weights, dim=0)

class MolSubset(Dataset):
    def __init__(self, subset_indices, graphs, labels, weights):
        super(MolSubset, self).__init__()
        self.subset_indices = subset_indices
        self.graphs = [graphs[i] for i in subset_indices]
        self.labels = torch.from_numpy(labels[subset_indices])
        self.weights = torch.from_numpy(weights[subset_indices])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item].float(), self.weights[item]
