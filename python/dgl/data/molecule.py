import dgl.backend as F
import numpy as np
import os
import pickle

from dgl import DGLGraph
from .utils import download, get_download_dir, _get_dgl_url, Subset

try:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

__all__ = ['one_hot_encoding', 'BaseAtomFeaturizer', 'DefaultAtomFeaturizer', 'mol2dgl',
           'consecutive_split', 'BinaryClassificationDataset', 'Tox21']

_urls = {
    'tox21': 'dataset/tox21.csv.gz'
}

def one_hot_encoding(x, allowable_set):
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
        x == allowable_set[i].
    """
    return list(map(lambda s: x == s, allowable_set))

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers

    All atom featurizers that map a molecule to atom features should subclass it.
    All subclasses should overrite :math:`_featurize_atom`, which featurizes a single
    atom and :math:`__call__`, which featurize all atoms in a molecule.
    """
    def _featurize_atom(self, atom):
        return NotImplementedError

    def __call__(self, mol):
        return NotImplementedError

class DefaultAtomFeaturizer(BaseAtomFeaturizer):
    """A default featurizer for atoms.

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to be 'h'.
    """
    def __init__(self, atom_data_field='h'):
        super(DefaultAtomFeaturizer, self).__init__()
        self.atom_data_field = atom_data_field

    @property
    def feat_size(self):
        """Returns feature size"""
        return 74

    def _featurize_atom(self, atom):
        """Featurize an atom

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom

        Returns
        -------
        results : list
            List of feature values, including boolean values and numbers
        """
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',
                      'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                      'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                      'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                      'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        results = one_hot_encoding(atom.GetSymbol(), atom_types) + \
                  one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_hot_encoding(atom.GetHybridization(),
                                        [Chem.rdchem.HybridizationType.SP,
                                         Chem.rdchem.HybridizationType.SP2,
                                         Chem.rdchem.HybridizationType.SP3,
                                         Chem.rdchem.HybridizationType.SP3D,
                                         Chem.rdchem.HybridizationType.SP3D2]) + \
                  [atom.GetIsAromatic()] + \
                  one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        return results

    def __call__(self, mol):
        """Featurize a molecule

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Atom features of shape (N, 74),
            where N is the number of atoms in the molecule
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append(self._featurize_atom(atom))
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.float32))

        return {self.atom_data_field: atom_features}

def mol2dgl(mol, add_self_loop=False, atom_featurizer=None, bond_featurizer=None):
    """Convert RDKit molecule object into a DGLGraph.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance.
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
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

def consecutive_split(dataset, frac_train, frac_val, frac_test):
    """Split the dataset into three subsets with consecutive indices
    for training, validation and test.

    Parameters
    ----------
    dataset
        len(dataset) should return the number of total datapoints
    frac_train : float
        Proportion of data to use for training
    frac_val : float
        Proportion of data to use for validation
    frac_test : float
        Proportion of data to use for test

    Returns
    -------
    train_indices : list
        List of indices for datapoints in the training set
    val_indices : list
        List of indices for datapoints in the validation set
    test_indices : list
        List of indices for datapoints in the test set
    """
    frac_total = frac_train + frac_val + frac_test
    assert np.allclose(frac_total, 1.), \
        'Expect frac_train + frac_val + frac_test = 1, got {:.4f}'.format(frac_total)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    val_cutoff = int((frac_train + frac_val) * num_datapoints)

    total_indices = range(num_datapoints)
    train_indices = total_indices[:train_cutoff]
    val_indices = total_indices[train_cutoff:val_cutoff]
    test_indices = total_indices[val_cutoff:]

    return train_indices, val_indices, test_indices

class BinaryClassificationDataset(object):
    """An abstract class for binary classification dataset on molecules.

    This dataset will be compatible for both single label and multi label prediction.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. In data pre-processing, we set non-existing labels to be 0 so that they
    can be placed in tensors and used for masking in loss computation.

    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs will be saved for reloading so that we do not need to reconstruct them every time.

    To inherit this class and develop your dataset, you only need to implement ``_load_data(self)``.

    Parameters
    ----------
    tasks : list
        List of strings representing task names.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    shuffle : bool
        Whether to shuffle the dataset when loading.
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    weight_balance : bool
        Whether to perform weight balancing to address the issue of
        unbalanced dataset. If True, we will set the weight of negative
        samples to be 1 and the weight of positive samples to be the number
        of negative samples divided by the number of positive samples.
    frac_train : float
        Proportion of data to use for training.
    frac_val : float
        Proportion of data to use for validation.
    frac_test : float
        Proportion of data to use for test.
    dataset_split_func : callable
        Maps (dataset, frac_train, frac_val, frac_test) to the indices
        for training, validation and test subsets as three lists.
    """
    def __init__(self, tasks, atom_featurizer, bond_featurizer,
                 shuffle, add_self_loop, weight_balance,
                 frac_train, frac_val, frac_test, dataset_split_func):
        super(BinaryClassificationDataset, self).__init__()
        self.tasks = tasks
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.shuffle = shuffle
        self.add_self_loop = add_self_loop
        self.weight_balance = weight_balance

        # Load dataset
        self._load_data()
        # Construct DGLGraphs and featurize molecules
        self._preprocess_data()

        # Weight balancing for dataset imbalance
        self._task_pos_weights = [1 for _ in range(len(tasks))]
        self._weight_balancing()

        self.labels = F.zerocopy_from_numpy(self.labels.astype(np.float32))
        self.w = F.zerocopy_from_numpy(self.w.astype(np.float32))

        # Split the dataset
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.frac_test = frac_test
        self.dataset_split_func = dataset_split_func
        self._split_dataset()

    def __len__(self):
        """Dataset size

        Returns
        -------
        int
            Number of molecules in the dataset
        """
        return len(self.smiles)

    def __getitem__(self, item):
        """Get the ith datapoint

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for all tasks
        Tensor of dtype float32
            Weights of the datapoint for all tasks
        """
        return self.smiles[item], self.graphs[item], self.labels[item], self.w[item]

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
    def task_pos_weights(self):
        """Get weights for positive samples on each task

        Returns
        -------
        list
            list[i] gives the weight of positive samples on task i
        """
        return self._task_pos_weights

    def _load_data(self):
        """Load data.

        The missing values will be replaced by "".

        After calling this function, one attribute will be set:

        * self.df is a dataframe that should contain task values under column
          names of self.tasks as well as a column of smiles under name 'smiles'
        """
        return NotImplementedError

    def _preprocess_data(self):
        """Pre-process the dataset

        * Convert molecules from smiles format into DGLGraphs
          and featurize their atoms
        * Set missing labels to be 0 and use a binary weighting
          matrix to mask them in loss computation
        """
        if self.shuffle:
            self.df = self.df.sample(frac=1)

        # Convert smiles to DGLGraphs and featurize their atoms.
        self.smiles = self.df['smiles'].tolist()
        n_tasks = len(self.tasks)
        num_datapoints = len(self)

        path_to_dgl_graphs = 'dgl_graphs.pkl'
        if os.path.exists(path_to_dgl_graphs):
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            with open(path_to_dgl_graphs, 'rb') as f:
                graphs = pickle.load(f)
        else:
            # First-time construction of DGLGraphs
            graphs = []
            for id, s in enumerate(self.smiles):
                print('Processing smile {:d}/{:d}'.format(id + 1, num_datapoints))
                mol = Chem.MolFromSmiles(s)
                # Canonically index atoms in each molecule
                new_order = rdmolfiles.CanonicalRankAtoms(mol)
                mol = rdmolops.RenumberAtoms(mol, new_order)
                graphs.append(mol2dgl(
                    mol, atom_featurizer=self.atom_featurizer,
                    bond_featurizer=self.bond_featurizer))

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
                    # The missing labels are now replaced with label 0 and will be masked
                    # for loss computation
                    labels[i, task] = 0.
                    w[i, task] = 0.

        self.labels = labels
        self.w = w

    def _weight_balancing(self):
        """Perform re-balancing for each task.

        It's quite common that the number of positive samples and the
        number of negative samples are significantly different. To compensate
        for the class imbalance issue, we can weight each datapoint in
        loss computation.

        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.

        If weight balancing is performed, two attributes will be affected:

        * self.w now consists of weights for balancing
        * self._task_pos_weights is set, which is a list of positive sample weights
          for each task.
        """
        if not self.weight_balance:
            return

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

        self._task_pos_weights = task_pos_weights

    def _split_dataset(self):
        """Split the dataset into three subsets for training,
        validation and test.
        """
        train_indices, val_indices, test_indices = self.dataset_split_func(
            self, self.frac_train, self.frac_val, self.frac_test)
        self.train_set = Subset(self, train_indices)
        self.val_set = Subset(self, val_indices)
        self.test_set = Subset(self, test_indices)

class Tox21(BinaryClassificationDataset):
    """Tox21 dataset.

    The Toxicology in the 21st Century (https://tripod.nih.gov/tox21/challenge/)
    initiative created a public database measuring toxicity of compounds, which
    has been used in the 2014 Tox21 Data Challenge. The dataset contains qualitative
    toxicity measurements for 8014 compounds on 12 different targets, including nuclear
    receptors and stress response pathways. Each target results in a binary label.

    A common issue for multi-task prediction is that some datapoints are not labeled for
    all tasks. This is also the case for Tox21. In data pre-processing, we set non-existing
    labels to be 0 so that they can be placed in tensors and used for masking in loss computation.
    See examples below for more details.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.

    Parameters
    ----------
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to be None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to be None.
    shuffle : bool
        Whether to shuffle the dataset when loading. Default to be False.
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to be False.
    weight_balance : bool
        Whether to perform weight balancing to address the issue of
        unbalanced dataset. If True, we will set the weight of negative
        samples to be 1 and the weight of positive samples to be the number
        of negative samples divided by the number of positive samples.
        Default to be True.
    frac_train : float
        Proportion of data to use for training. Default to be 0.8.
    frac_val : float
        Proportion of data to use for validation. Default to be 0.1.
    frac_test : float
        Proportion of data to use for test. Default to be 0.1.
    dataset_split_func : callable
        Maps (dataset, frac_train, frac_val, frac_test) to the indices
        for training, validation and test subsets as three lists.
        Default to be consecutive_split.

    Examples
    --------
    >>> dataset = Tox21()
    >>> dataset.tasks
    ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
     'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    >>> dataset.add_self_loop
    False

    The training set, validation set and test set can be separately accessed
    ``dataset.train_set``, ``dataset.val_set`` and ``dataset.test_set``.
    Below we show how to use ``dataset.train_set`` and you can use the rest
    two with same operations.

    To know the number of datapoints in the training set

    >>> len(dataset.train_set)
    6264

    To access the first datapoint in the training set

    >>> dataset.train_set[0]
    (DGLGraph(num_nodes=16, num_edges=34),
    tensor([0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
    tensor([1.0000, 1.0000, 7.5273, 0.0000, 0.0000, 1.0000, 1.0000, 5.1911, 1.0000,
            1.0000, 1.0000, 1.0000]))

    The three elements of the datapoint are separately the constructed
    :class:`~dgl.DGLGraph`, the labels for all tasks and the weights of the datapoint
    for all tasks.

    The weights will be used for datapoint weighting in loss function computation.

    * If :math:`weights[i] = 0`, then this means the datapoint does not have a label for task i.
    * A label re-balancing is performed. For negative samples, we use a weight :math:`1`, for
      positive samples the weight is decided as the number of negative samples divided by
      the number of positive samples.

    You can also access the positive sample weight for all tasks via

    >>> dataset.task_pos_weights
    tensor([22.5113, 27.5148,  7.5273, 18.4033,  6.8096, 18.8714, 33.6774,  5.1911,
            25.7879, 16.3844,  5.3290, 15.0142])
    """
    def __init__(self, atom_featurizer=None, bond_featurizer=None, shuffle=False,
                 add_self_loop=False, weight_balance=True,
                 frac_train=0.8, frac_val=0.1, frac_test=0.1,
                 dataset_split_func=consecutive_split):
        super(Tox21, self).__init__(tasks=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
                                           'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                                           'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
                                    atom_featurizer=atom_featurizer,
                                    bond_featurizer=bond_featurizer,
                                    shuffle=shuffle,
                                    add_self_loop=add_self_loop,
                                    weight_balance=weight_balance,
                                    frac_train=frac_train,
                                    frac_val=frac_val,
                                    frac_test=frac_test,
                                    dataset_split_func=dataset_split_func)

    def _load_data(self):
        """Load data from a csv file.

        If this is the first time to use the dataset, the csv file
        will get downloaded first. The nan values in the csv will
        be replaced by "".

        After calling this function, two attributes will be set:

        * self.df stores a dataframe for smiles and task values
        * self.smiles stores smiles in a list
        """
        self.data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(_urls['tox21']), path=self.data_path)
        print('Loading data from csv file...')
        df = pd.read_csv(self.data_path)
        # np.nan suggests non-existing labels, and we replace them by ""
        df = df.replace(np.nan, str(""), regex=True)
        self.df = df
        self.smiles = self.df['smiles'].tolist()
