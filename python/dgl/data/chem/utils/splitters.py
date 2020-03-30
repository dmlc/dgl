"""Various methods for splitting chemical datasets.

We mostly adapt them from deepchem
(https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py).
"""
import numpy as np

from collections import defaultdict
from functools import partial
from itertools import accumulate, chain

from ...utils import split_dataset, Subset
from .... import backend as F
from ....contrib.deprecation import deprecated

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.rdmolops import FastFindRings
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    pass

__all__ = ['ConsecutiveSplitter',
           'RandomSplitter',
           'MolecularWeightSplitter',
           'ScaffoldSplitter',
           'SingleTaskStratifiedSplitter']

def base_k_fold_split(split_method, dataset, k, log):
    """Split dataset for k-fold cross validation.

    Parameters
    ----------
    split_method : callable
        Arbitrary method for splitting the dataset
        into training, validation and test subsets.
    dataset
        We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
        gives the ith datapoint.
    k : int
        Number of folds to use and should be no smaller than 2.
    log : bool
        Whether to print a message at the start of preparing each fold.

    Returns
    -------
    all_folds : list of 2-tuples
        Each element of the list represents a fold and is a 2-tuple (train_set, val_set).
    """
    assert k >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k)
    all_folds = []
    frac_per_part = 1./ k
    for i in range(k):
        if log:
            print('Processing fold {:d}/{:d}'.format(i+1, k))
        # We are reusing the code for train-validation-test split.
        train_set1, val_set, train_set2 = split_method(dataset,
                                                       frac_train=i * frac_per_part,
                                                       frac_val=frac_per_part,
                                                       frac_test=1. - (i + 1) * frac_per_part)
        # For cross validation, each fold consists of only a train subset and
        # a validation subset.
        train_set = Subset(dataset, train_set1.indices + train_set2.indices)
        all_folds.append((train_set, val_set))
    return all_folds

def train_val_test_sanity_check(frac_train, frac_val, frac_test):
    """Sanity check for train-val-test split

    Ensure that the fractions of the dataset to use for training,
    validation and test add up to 1.

    Parameters
    ----------
    frac_train : float
        Fraction of the dataset to use for training.
    frac_val : float
        Fraction of the dataset to use for validation.
    frac_test : float
        Fraction of the dataset to use for test.
    """
    total_fraction = frac_train + frac_val + frac_test
    assert np.allclose(total_fraction, 1.), \
        'Expect the sum of fractions for training, validation and ' \
        'test to be 1, got {:.4f}'.format(total_fraction)

def indices_split(dataset, frac_train, frac_val, frac_test, indices):
    """Reorder datapoints based on the specified indices and then take consecutive
    chunks as subsets.

    Parameters
    ----------
    dataset
        We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
        gives the ith datapoint.
    frac_train : float
        Fraction of data to use for training.
    frac_val : float
        Fraction of data to use for validation.
    frac_test : float
        Fraction of data to use for test.
    indices : list or ndarray
        Indices specifying the order of datapoints.

    Returns
    -------
    list of length 3
        Subsets for training, validation and test, which are all :class:`Subset` instances.
    """
    frac_list = np.asarray([frac_train, frac_val, frac_test])
    assert np.allclose(np.sum(frac_list), 1.), \
        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])

    return [Subset(dataset, list(indices[offset - length:offset]))
            for offset, length in zip(accumulate(lengths), lengths)]

def count_and_log(message, i, total, log_every_n):
    """Print a message to reflect the progress of processing once a while.

    Parameters
    ----------
    message : str
        Message to print.
    i : int
        Current index.
    total : int
        Total count.
    log_every_n : None or int
        Molecule related computation can take a long time for a large dataset and we want
        to learn the progress of processing. This can be done by printing a message whenever
        a batch of ``log_every_n`` molecules have been processed. If None, no messages will
        be printed.
    """
    if (log_every_n is not None) and ((i+1) % log_every_n == 0):
        print('{} {:d}/{:d}'.format(message, i+1, total))

def prepare_mols(dataset, mols, sanitize, log_every_n=1000):
    """Prepare RDKit molecule instances.

    Parameters
    ----------
    dataset
        We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
        gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
        ith datapoint.
    mols : None or list of rdkit.Chem.rdchem.Mol
        None or pre-computed RDKit molecule instances. If not None, we expect a
        one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
        ``mols[i]`` corresponds to ``dataset.smiles[i]``.
    sanitize : bool
        This argument only comes into effect when ``mols`` is None and decides whether
        sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
    log_every_n : None or int
        Molecule related computation can take a long time for a large dataset and we want
        to learn the progress of processing. This can be done by printing a message whenever
        a batch of ``log_every_n`` molecules have been processed. If None, no messages will
        be printed. Default to 1000.

    Returns
    -------
    mols : list of rdkit.Chem.rdchem.Mol
        RDkit molecule instances where there is a one-on-one correspondence between
        ``dataset.smiles`` and ``mols``, i.e. ``mols[i]`` corresponds to ``dataset.smiles[i]``.
    """
    if mols is not None:
        # Sanity check
        assert len(mols) == len(dataset), \
            'Expect mols to be of the same size as that of the dataset, ' \
            'got {:d} and {:d}'.format(len(mols), len(dataset))
    else:
        if log_every_n is not None:
            print('Start initializing RDKit molecule instances...')
        mols = []
        for i, s in enumerate(dataset.smiles):
            count_and_log('Creating RDKit molecule instance',
                          i, len(dataset.smiles), log_every_n)
            mols.append(Chem.MolFromSmiles(s, sanitize=sanitize))

    return mols

class ConsecutiveSplitter(object):
    """Split datasets with the input order.

    The dataset is split without permutation, so the splitting is deterministic.
    """

    @staticmethod
    @deprecated('Import ConsecutiveSplitter from dgllife.utils.splitters instead.', 'class')
    def train_val_test_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1):
        """Split the dataset into three consecutive chunks for training, validation and test.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
            gives the ith datapoint.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.

        Returns
        -------
        list of length 3
            Subsets for training, validation and test, which are all :class:`Subset` instances.
        """
        return split_dataset(dataset, frac_list=[frac_train, frac_val, frac_test], shuffle=False)

    @staticmethod
    @deprecated('Import ConsecutiveSplitter from dgllife.utils.splitters instead.', 'class')
    def k_fold_split(dataset, k=5, log=True):
        """Split the dataset for k-fold cross validation by taking consecutive chunks.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
            gives the ith datapoint.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        log : bool
            Whether to print a message at the start of preparing each fold.

        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple (train_set, val_set).
        """
        return base_k_fold_split(ConsecutiveSplitter.train_val_test_split, dataset, k, log)

class RandomSplitter(object):
    """Randomly reorder datasets and then split them.

    The dataset is split with permutation and the splitting is hence random.
    """
    @staticmethod
    @deprecated('Import RandomSplitter from dgllife.utils.splitters instead.', 'class')
    def train_val_test_split(dataset, frac_train=0.8, frac_val=0.1,
                             frac_test=0.1, random_state=None):
        """Randomly permute the dataset and then split it into
        three consecutive chunks for training, validation and test.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
            gives the ith datapoint.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.
        random_state : None, int or array_like, optional
            Random seed used to initialize the pseudo-random number generator.
            Can be any integer between 0 and 2**32 - 1 inclusive, an array
            (or other sequence) of such integers, or None (the default).
            If seed is None, then RandomState will try to read data from /dev/urandom
            (or the Windows analogue) if available or seed from the clock otherwise.

        Returns
        -------
        list of length 3
            Subsets for training, validation and test.
        """
        return split_dataset(dataset, frac_list=[frac_train, frac_val, frac_test],
                             shuffle=True, random_state=random_state)

    @staticmethod
    @deprecated('Import RandomSplitter from dgllife.utils.splitters instead.', 'class')
    def k_fold_split(dataset, k=5, random_state=None, log=True):
        """Randomly permute the dataset and then split it
        for k-fold cross validation by taking consecutive chunks.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
            gives the ith datapoint.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        random_state : None, int or array_like, optional
            Random seed used to initialize the pseudo-random number generator.
            Can be any integer between 0 and 2**32 - 1 inclusive, an array
            (or other sequence) of such integers, or None (the default).
            If seed is None, then RandomState will try to read data from /dev/urandom
            (or the Windows analogue) if available or seed from the clock otherwise.
        log : bool
            Whether to print a message at the start of preparing each fold. Default to True.

        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple (train_set, val_set).
        """
        # Permute the dataset only once so that each datapoint
        # will appear once in exactly one fold.
        indices = np.random.RandomState(seed=random_state).permutation(len(dataset))

        return base_k_fold_split(partial(indices_split, indices=indices), dataset, k, log)

class MolecularWeightSplitter(object):
    """Sort molecules based on their weights and then split them."""
    @staticmethod
    @deprecated('Import MolecularWeightSplitter from dgllife.utils.splitters instead.', 'class')
    def molecular_weight_indices(molecules, log_every_n):
        """Reorder molecules based on molecular weights.

        Parameters
        ----------
        molecules : list of rdkit.Chem.rdchem.Mol
            Pre-computed RDKit molecule instances. We expect a one-on-one
            correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed.

        Returns
        -------
        indices : list or ndarray
            Indices specifying the order of datapoints, which are basically
            argsort of the molecular weights.
        """
        if log_every_n is not None:
            print('Start computing molecular weights.')
        mws = []
        for i, mol in enumerate(molecules):
            count_and_log('Computing molecular weight for compound',
                          i, len(molecules), log_every_n)
            mws.append(Chem.rdMolDescriptors.CalcExactMolWt(mol))

        return np.argsort(mws)

    @staticmethod
    @deprecated('Import MolecularWeightSplitter from dgllife.utils.splitters instead.', 'class')
    def train_val_test_split(dataset, mols=None, sanitize=True, frac_train=0.8,
                             frac_val=0.1, frac_test=0.1, log_every_n=1000):
        """Sort molecules based on their weights and then split them into
        three consecutive chunks for training, validation and test.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        mols : None or list of rdkit.Chem.rdchem.Mol
            None or pre-computed RDKit molecule instances. If not None, we expect a
            one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``. Default to None.
        sanitize : bool
            This argument only comes into effect when ``mols`` is None and decides whether
            sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to be True.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed. Default to 1000.

        Returns
        -------
        list of length 3
            Subsets for training, validation and test, which are all :class:`Subset` instances.
        """
        # Perform sanity check first as molecule instance initialization and descriptor
        # computation can take a long time.
        train_val_test_sanity_check(frac_train, frac_val, frac_test)
        molecules = prepare_mols(dataset, mols, sanitize, log_every_n)
        sorted_indices = MolecularWeightSplitter.molecular_weight_indices(molecules, log_every_n)

        return indices_split(dataset, frac_train, frac_val, frac_test, sorted_indices)

    @staticmethod
    @deprecated('Import MolecularWeightSplitter from dgllife.utils.splitters instead.', 'class')
    def k_fold_split(dataset, mols=None, sanitize=True, k=5, log_every_n=1000):
        """Sort molecules based on their weights and then split them
        for k-fold cross validation by taking consecutive chunks.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        mols : None or list of rdkit.Chem.rdchem.Mol
            None or pre-computed RDKit molecule instances. If not None, we expect a
            one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``. Default to None.
        sanitize : bool
            This argument only comes into effect when ``mols`` is None and decides whether
            sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to be True.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed. Default to 1000.

        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple (train_set, val_set).
        """
        molecules = prepare_mols(dataset, mols, sanitize, log_every_n)
        sorted_indices = MolecularWeightSplitter.molecular_weight_indices(molecules, log_every_n)

        return base_k_fold_split(partial(indices_split, indices=sorted_indices), dataset, k,
                                 log=(log_every_n is not None))

class ScaffoldSplitter(object):
    """Group molecules based on their Bemis-Murcko scaffolds and then split the groups.

    Group molecules so that all molecules in a group have a same scaffold (see reference).
    The dataset is then split at the level of groups.

    References
    ----------
    Bemis, G. W.; Murcko, M. A. “The Properties of Known Drugs.
        1. Molecular Frameworks.” J. Med. Chem. 39:2887-93 (1996).
    """

    @staticmethod
    @deprecated('Import ScaffoldSplitter from dgllife.utils.splitters instead.', 'class')
    def get_ordered_scaffold_sets(molecules, include_chirality, log_every_n):
        """Group molecules based on their Bemis-Murcko scaffolds and
        order these groups based on their sizes.

        The order is decided by comparing the size of groups, where groups with a larger size
        are placed before the ones with a smaller size.

        Parameters
        ----------
        molecules : list of rdkit.Chem.rdchem.Mol
            Pre-computed RDKit molecule instances. We expect a one-on-one
            correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``.
        include_chirality : bool
            Whether to consider chirality in computing scaffolds.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed.

        Returns
        -------
        scaffold_sets : list
            Each element of the list is a list of int,
            representing the indices of compounds with a same scaffold.
        """
        if log_every_n is not None:
            print('Start computing Bemis-Murcko scaffolds.')
        scaffolds = defaultdict(list)
        for i, mol in enumerate(molecules):
            count_and_log('Computing Bemis-Murcko for compound',
                          i, len(molecules), log_every_n)
            # For mols that have not been sanitized, we need to compute their ring information
            try:
                FastFindRings(mol)
                mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=include_chirality)
                # Group molecules that have the same scaffold
                scaffolds[mol_scaffold].append(i)
            except:
                print('Failed to compute the scaffold for molecule {:d} '
                      'and it will be excluded.'.format(i+1))

        # Order groups of molecules by first comparing the size of groups
        # and then the index of the first compound in the group.
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        return scaffold_sets

    @staticmethod
    @deprecated('Import ScaffoldSplitter from dgllife.utils.splitters instead.', 'class')
    def train_val_test_split(dataset, mols=None, sanitize=True, include_chirality=False,
                             frac_train=0.8, frac_val=0.1, frac_test=0.1, log_every_n=1000):
        """Split the dataset into training, validation and test set based on molecular scaffolds.

        This spliting method ensures that molecules with a same scaffold will be collectively
        in only one of the training, validation or test set. As a result, the fraction
        of dataset to use for training and validation tend to be smaller than ``frac_train``
        and ``frac_val``, while the fraction of dataset to use for test tends to be larger
        than ``frac_test``.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        mols : None or list of rdkit.Chem.rdchem.Mol
            None or pre-computed RDKit molecule instances. If not None, we expect a
            one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``. Default to None.
        sanitize : bool
            This argument only comes into effect when ``mols`` is None and decides whether
            sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to True.
        include_chirality : bool
            Whether to consider chirality in computing scaffolds. Default to False.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed. Default to 1000.

        Returns
        -------
        list of length 3
            Subsets for training, validation and test, which are all :class:`Subset` instances.
        """
        # Perform sanity check first as molecule related computation can take a long time.
        train_val_test_sanity_check(frac_train, frac_val, frac_test)
        molecules = prepare_mols(dataset, mols, sanitize)
        scaffold_sets = ScaffoldSplitter.get_ordered_scaffold_sets(
            molecules, include_chirality, log_every_n)

        train_indices, val_indices, test_indices = [], [], []
        train_cutoff = int(frac_train * len(molecules))
        val_cutoff = int((frac_train + frac_val) * len(molecules))
        for group_indices in scaffold_sets:
            if len(train_indices) + len(group_indices) > train_cutoff:
                if len(train_indices) + len(val_indices) + len(group_indices) > val_cutoff:
                    test_indices.extend(group_indices)
                else:
                    val_indices.extend(group_indices)
            else:
                train_indices.extend(group_indices)

        return [Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices)]

    @staticmethod
    @deprecated('Import ScaffoldSplitter from dgllife.utils.splitters instead.', 'class')
    def k_fold_split(dataset, mols=None, sanitize=True,
                     include_chirality=False, k=5, log_every_n=1000):
        """Group molecules based on their scaffolds and sort groups based on their sizes.
        The groups are then split for k-fold cross validation.

        Same as usual k-fold splitting methods, each molecule will appear only once
        in the validation set among all folds. In addition, this method ensures that
        molecules with a same scaffold will be collectively in either the training
        set or the validation set for each fold.

        Note that the folds can be highly imbalanced depending on the
        scaffold distribution in the dataset.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        mols : None or list of rdkit.Chem.rdchem.Mol
            None or pre-computed RDKit molecule instances. If not None, we expect a
            one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``. Default to None.
        sanitize : bool
            This argument only comes into effect when ``mols`` is None and decides whether
            sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to True.
        include_chirality : bool
            Whether to consider chirality in computing scaffolds. Default to False.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed. Default to 1000.

        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple (train_set, val_set).
        """
        assert k >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k)

        molecules = prepare_mols(dataset, mols, sanitize)
        scaffold_sets = ScaffoldSplitter.get_ordered_scaffold_sets(
            molecules, include_chirality, log_every_n)

        # k buckets that form a relatively balanced partition of the dataset
        index_buckets = [[] for _ in range(k)]
        for group_indices in scaffold_sets:
            bucket_chosen = int(np.argmin([len(bucket) for bucket in index_buckets]))
            index_buckets[bucket_chosen].extend(group_indices)

        all_folds = []
        for i in range(k):
            if log_every_n is not None:
                print('Processing fold {:d}/{:d}'.format(i + 1, k))
            train_indices = list(chain.from_iterable(index_buckets[:i] + index_buckets[i+1:]))
            val_indices = index_buckets[i]
            all_folds.append((Subset(dataset, train_indices), Subset(dataset, val_indices)))

        return all_folds

class SingleTaskStratifiedSplitter(object):
    """Splits the dataset by stratification on a single task.

    We sort the molecules based on their label values for a task and then repeatedly
    take buckets of datapoints to augment the training, validation and test subsets.
    """
    @staticmethod
    @deprecated('Import SingleTaskStratifiedSplitter from '
                'dgllife.utils.splitters instead.', 'class')
    def train_val_test_split(dataset, labels, task_id, frac_train=0.8, frac_val=0.1,
                             frac_test=0.1, bucket_size=10, random_state=None):
        """Split the dataset into training, validation and test subsets as stated above.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        labels : tensor of shape (N, T)
            Dataset labels all tasks. N for the number of datapoints and T for the number
            of tasks.
        task_id : int
            Index for the task.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.
        bucket_size : int
            Size of bucket of datapoints. Default to 10.
        random_state : None, int or array_like, optional
            Random seed used to initialize the pseudo-random number generator.
            Can be any integer between 0 and 2**32 - 1 inclusive, an array
            (or other sequence) of such integers, or None (the default).
            If seed is None, then RandomState will try to read data from /dev/urandom
            (or the Windows analogue) if available or seed from the clock otherwise.

        Returns
        -------
        list of length 3
            Subsets for training, validation and test, which are all :class:`Subset` instances.
        """
        train_val_test_sanity_check(frac_train, frac_val, frac_test)

        if random_state is not None:
            np.random.seed(random_state)

        if not isinstance(labels, np.ndarray):
            labels = F.asnumpy(labels)
        task_labels = labels[:, task_id]
        sorted_indices = np.argsort(task_labels)

        train_bucket_cutoff = int(np.round(frac_train * bucket_size))
        val_bucket_cutoff = int(np.round(frac_val * bucket_size)) + train_bucket_cutoff

        train_indices, val_indices, test_indices = [], [], []

        while sorted_indices.shape[0] >= bucket_size:
            current_batch, sorted_indices = np.split(sorted_indices, [bucket_size])
            shuffled = np.random.permutation(range(bucket_size))
            train_indices.extend(
                current_batch[shuffled[:train_bucket_cutoff]].tolist())
            val_indices.extend(
                current_batch[shuffled[train_bucket_cutoff:val_bucket_cutoff]].tolist())
            test_indices.extend(
                current_batch[shuffled[val_bucket_cutoff:]].tolist())

        # Place rest samples in the training set.
        train_indices.extend(sorted_indices.tolist())

        return [Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices)]

    @staticmethod
    @deprecated('Import SingleTaskStratifiedSplitter from '
                'dgllife.utils.splitters instead.', 'class')
    def k_fold_split(dataset, labels, task_id, k=5, log=True):
        """Sort molecules based on their label values for a task and then split them
        for k-fold cross validation by taking consecutive chunks.

        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        labels : tensor of shape (N, T)
            Dataset labels all tasks. N for the number of datapoints and T for the number
            of tasks.
        task_id : int
            Index for the task.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        log : bool
            Whether to print a message at the start of preparing each fold.

        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple (train_set, val_set).
        """
        if not isinstance(labels, np.ndarray):
            labels = F.asnumpy(labels)
        task_labels = labels[:, task_id]
        sorted_indices = np.argsort(task_labels).tolist()
        
        return base_k_fold_split(partial(indices_split, indices=sorted_indices), dataset, k, log)
