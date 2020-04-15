"""Node and edge featurization for molecular graphs."""
# pylint: disable= no-member, arguments-differ, invalid-name
import itertools
import os.path as osp

from collections import defaultdict
from functools import partial
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

import numpy as np
import torch
import dgl.backend as F

__all__ = ['one_hot_encoding',
           'atom_type_one_hot',
           'atomic_number_one_hot',
           'atomic_number',
           'atom_degree_one_hot',
           'atom_degree',
           'atom_total_degree_one_hot',
           'atom_total_degree',
           'atom_explicit_valence_one_hot',
           'atom_explicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_implicit_valence',
           'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot',
           'atom_total_num_H',
           'atom_formal_charge_one_hot',
           'atom_formal_charge',
           'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons',
           'atom_is_aromatic_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring_one_hot',
           'atom_is_in_ring',
           'atom_chiral_tag_one_hot',
           'atom_mass',
           'ConcatFeaturizer',
           'BaseAtomFeaturizer',
           'CanonicalAtomFeaturizer',
           'WeaveAtomFeaturizer',
           'bond_type_one_hot',
           'bond_is_conjugated_one_hot',
           'bond_is_conjugated',
           'bond_is_in_ring_one_hot',
           'bond_is_in_ring',
           'bond_stereo_one_hot',
           'BaseBondFeaturizer',
           'CanonicalBondFeaturizer',
           'WeaveEdgeFeaturizer']

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.

    Parameters
    ----------
    x
        Value to encode.
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element.

    Returns
    -------
    list
        List of boolean values where at most one value is True.
        The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
        and ``len(allowable_set) + 1`` otherwise.

    Examples
    --------
    >>> from dgllife.utils import one_hot_encoding
    >>> one_hot_encoding('C', ['C', 'O'])
    [True, False]
    >>> one_hot_encoding('S', ['C', 'O'])
    [False, False]
    >>> one_hot_encoding('S', ['C', 'O'], encode_unknown=True)
    [False, False, True]
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

#################################################################
# Atom featurization
#################################################################

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Atom types to consider. Default: ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``,
        ``Cl``, ``Br``, ``Mg``, ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``,
        ``K``, ``Tl``, ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
        ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``, ``Cr``,
        ``Pt``, ``Hg``, ``Pb``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atomic_number_one_hot
    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atomic_number_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the atomic number of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atomic numbers to consider. Default: ``1`` - ``100``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atom_type_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(1, 101))
    return one_hot_encoding(atom.GetAtomicNum(), allowable_set, encode_unknown)

def atomic_number(atom):
    """Get the atomic number for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
       List containing one int only.

    See Also
    --------
    atomic_number_one_hot
    atom_type_one_hot
    """
    return [atom.GetAtomicNum()]

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom degrees to consider. Default: ``0`` - ``10``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_total_degree
    atom_total_degree_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_degree(atom):
    """Get the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_degree_one_hot
    atom_total_degree
    atom_total_degree_one_hot
    """
    return [atom.GetDegree()]

def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom including Hs.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list
        Total degrees to consider. Default: ``0`` - ``5``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_degree_one_hot
    atom_total_degree
    """
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)

def atom_total_degree(atom):
    """The degree of an atom including Hs.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_degree_one_hot
    atom_degree
    atom_degree_one_hot
    """
    return [atom.GetTotalDegree()]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the explicit valence of an aotm.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom explicit valences to consider. Default: ``1`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_explicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(1, 7))
    return one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)

def atom_explicit_valence(atom):
    """Get the explicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_explicit_valence_one_hot
    """
    return [atom.GetExplicitValence()]

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom implicit valences to consider. Default: ``0`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    atom_implicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_implicit_valence(atom):
    """Get the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Reurns
    ------
    list
        List containing one int only.

    See Also
    --------
    atom_implicit_valence_one_hot
    """
    return [atom.GetImplicitValence()]

# pylint: disable=I1101
def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Total number of Hs to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_total_num_H
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_total_num_H(atom):
    """Get the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_num_H_one_hot
    """
    return [atom.GetTotalNumHs()]

def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the formal charge of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Formal charges to consider. Default: ``-2`` - ``2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_formal_charge
    """
    if allowable_set is None:
        allowable_set = list(range(-2, 3))
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """Get formal charge for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_formal_charge_one_hot
    """
    return [atom.GetFormalCharge()]

def atom_partial_charge(atom):
    """Get Gasteiger partial charge for an atom.

    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.

    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return [float(gasteiger_charge)]

def atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the number of radical electrons of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Number of radical electrons to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_num_radical_electrons
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)

def atom_num_radical_electrons(atom):
    """Get the number of radical electrons for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_num_radical_electrons_one_hot
    """
    return [atom.GetNumRadicalElectrons()]

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_aromatic
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.GetIsAromatic(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_aromatic_one_hot
    """
    return [atom.GetIsAromatic()]

def atom_is_in_ring_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """Get whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_in_ring_one_hot
    """
    return [atom.IsInRing()]

def atom_chiral_tag_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chiral tag of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.ChiralType
        Chiral tags to consider. Default: ``rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_OTHER``.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                         Chem.rdchem.ChiralType.CHI_OTHER]
    return one_hot_encoding(atom.GetChiralTag(), allowable_set, encode_unknown)

def atom_mass(atom, coef=0.01):
    """Get the mass of an atom and scale it.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.

    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * coef]

class ConcatFeaturizer(object):
    """Concatenate the evaluation results of multiple functions as a single feature.

    Parameters
    ----------
    func_list : list
        List of functions for computing molecular descriptors from objects of a same
        particular data type, e.g. ``rdkit.Chem.rdchem.Atom``. Each function is of signature
        ``func(data_type) -> list of float or bool or int``. The resulting order of
        the features will follow that of the functions in the list.
    """
    def __init__(self, func_list):
        self.func_list = func_list

    def __call__(self, x):
        """Featurize the input data.

        Parameters
        ----------
        x :
            Data to featurize.

        Returns
        -------
        list
            List of feature values, which can be of type bool, float or int.
        """
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers.

    Loop over all atoms in a molecule and featurize them with the ``featurizer_funcs``.

    **We assume the resulting DGLGraph will not contain any virtual nodes and a node i in the
    graph corresponds to exactly atom i in the molecule.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Atom) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.

    Examples
    --------

    >>> from dgllife.utils import BaseAtomFeaturizer, atom_mass, atom_degree_one_hot
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = BaseAtomFeaturizer({'mass': atom_mass, 'degree': atom_degree_one_hot})
    >>> atom_featurizer(mol)
    {'mass': tensor([[0.1201],
                     [0.1201],
                     [0.1600]]),
     'degree': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size for atom mass
    >>> print(atom_featurizer.feat_size('mass'))
    1
    >>> # Get feature size for atom degree
    >>> print(atom_featurizer.feat_size('degree'))
    11

    See Also
    --------
    CanonicalAtomFeaturizer
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

    def __call__(self, mol):
        """Featurize all atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features

class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
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

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from dgllife.utils import CanonicalAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                      1., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                      0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
                      0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    74

    See Also
    --------
    BaseAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot]
            )})

class WeaveAtomFeaturizer(object):
    """Atom featurizer in Weave.

    The atom featurization performed in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__, which considers:

    * atom types
    * chirality
    * formal charge
    * partial charge
    * aromatic atom
    * hybridization
    * hydrogen bond donor
    * hydrogen bond acceptor
    * the number of rings the atom belongs to for ring size between 3 and 8

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
    atom_types : list of str or None
        Atom types to consider for one-hot encoding. If None, we will use a default
        choice of ``'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'``.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice of ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``.
    hybridization_types : list of Chem.rdchem.HybridizationType or None
        Atom hybridization types to consider for one-hot encoding. If None, we will use a
        default choice of ``Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3``.
    """
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(WeaveAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3]
        self._hybridization_types = hybridization_types

        self._featurizer = ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=atom_types, encode_unknown=True),
            partial(atom_chiral_tag_one_hot, allowable_set=chiral_types),
            atom_formal_charge, atom_partial_charge, atom_is_aromatic,
            partial(atom_hybridization_one_hot, allowable_set=hybridization_types)
        ])

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]

        return feats.shape[-1]

    def get_donor_acceptor_info(self, mol_feats):
        """Bookkeep whether an atom is donor/acceptor for hydrogen bonds.

        Parameters
        ----------
        mol_feats : tuple of rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeature
            Features for molecules.

        Returns
        -------
        is_donor : dict
            Mapping atom ids to binary values indicating whether atoms
            are donors for hydrogen bonds
        is_acceptor : dict
            Mapping atom ids to binary values indicating whether atoms
            are acceptors for hydrogen bonds
        """
        is_donor = defaultdict(bool)
        is_acceptor = defaultdict(bool)
        # Get hydrogen bond donor/acceptor information
        for feats in mol_feats:
            if feats.GetFamily() == 'Donor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_donor[u] = True
            elif feats.GetFamily() == 'Acceptor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acceptor[u] = True

        return is_donor, is_acceptor

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()

        # Get information for donor and acceptor
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        is_donor, is_acceptor = self.get_donor_acceptor_info(mol_feats)

        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom)
            # Donor/acceptor indicator
            feats.append(float(is_donor[i]))
            feats.append(float(is_acceptor[i]))
            # Count the number of rings the atom belongs to for ring size between 3 and 8
            count = [0 for _ in range(3, 9)]
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            feats.extend(count)
            atom_features.append(feats)
        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondType
        Bond types to consider. Default: ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

def bond_is_conjugated_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_conjugated
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.GetIsConjugated(), allowable_set, encode_unknown)

def bond_is_conjugated(bond):
    """Get whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_conjugated_one_hot
    """
    return [bond.GetIsConjugated()]

def bond_is_in_ring_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)

def bond_is_in_ring(bond):
    """Get whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_in_ring_one_hot
    """
    return [bond.IsInRing()]

def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the stereo configuration of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of rdkit.Chem.rdchem.BondStereo
        Stereo configurations to consider. Default: ``rdkit.Chem.rdchem.BondStereo.STEREONONE``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOANY``, ``rdkit.Chem.rdchem.BondStereo.STEREOZ``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOE``, ``rdkit.Chem.rdchem.BondStereo.STEREOCIS``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOTRANS``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,
                         Chem.rdchem.BondStereo.STEREOANY,
                         Chem.rdchem.BondStereo.STEREOZ,
                         Chem.rdchem.BondStereo.STEREOE,
                         Chem.rdchem.BondStereo.STEREOCIS,
                         Chem.rdchem.BondStereo.STEREOTRANS]
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)

class BaseBondFeaturizer(object):
    """An abstract class for bond featurizers.
    Loop over all bonds in a molecule and featurize them with the ``featurizer_funcs``.
    We assume the constructed ``DGLGraph`` is a bi-directed graph where the **i** th bond in the
    molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the **(2i)**-th and **(2i+1)**-th edges
    in the DGLGraph.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Bond) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.

    Examples
    --------

    >>> from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = BaseBondFeaturizer({'type': bond_type_one_hot, 'ring': bond_is_in_ring})
    >>> bond_featurizer(mol)
    {'type': tensor([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.]]),
     'ring': tensor([[0.], [0.], [0.], [0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    4
    >>> bond_featurizer.feat_size('ring')
    1

    See Also
    --------
    CanonicalBondFeaturizer
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            bond = Chem.MolFromSmiles('CO').GetBondWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](bond))

        return self._feat_sizes[feat_name]

    def __call__(self, mol):
        """Featurize all bonds in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    """A default featurizer for bonds.

    The bond features include:
    * **One hot encoding of the bond type**. The supported bond types include
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``,
      ``STEREOCIS``, ``STEREOTRANS``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Examples
    --------
    >>> from dgllife.utils import CanonicalBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    12

    See Also
    --------
    BaseBondFeaturizer
    """
    def __init__(self, bond_data_field='e'):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 bond_stereo_one_hot]
            )})

# pylint: disable=E1102
class WeaveEdgeFeaturizer(object):
    """Edge featurizer in Weave.

    The edge featurization is introduced in `Molecular Graph Convolutions:
    Moving Beyond Fingerprints <https://arxiv.org/abs/1603.00856>`__.

    This featurization is performed for a complete graph of atoms with self loops added,
    which considers:

    * Number of bonds between each pairs of atoms
    * One-hot encoding of bond type if a bond exists between a pair of atoms
    * Whether a pair of atoms belongs to a same ring

    Parameters
    ----------
    edge_data_field : str
        Name for storing edge features in DGLGraphs, default to ``'e'``.
    max_distance : int
        Maximum number of bonds to consider between each pair of atoms.
        Default to 7.
    bond_types : list of Chem.rdchem.BondType or None
        Bond types to consider for one hot encoding. If None, we consider by
        default single, double, triple and aromatic bonds.
    """
    def __init__(self, edge_data_field='e', max_distance=7, bond_types=None):
        super(WeaveEdgeFeaturizer, self).__init__()

        self._edge_data_field = edge_data_field
        self._max_distance = max_distance
        if bond_types is None:
            bond_types = [Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC]
        self._bond_types = bond_types

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._edge_data_field]

        return feats.shape[-1]

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping self._edge_data_field to a float32 tensor of shape (N, M), where
            N is the number of atom pairs and M is the feature size.
        """
        # Part 1 based on number of bonds between each pair of atoms
        distance_matrix = torch.from_numpy(Chem.GetDistanceMatrix(mol))
        # Change shape from (V, V, 1) to (V^2, 1)
        distance_matrix = distance_matrix.float().reshape(-1, 1)
        # Elementwise compare if distance is bigger than 0, 1, ..., max_distance - 1
        distance_indicators = (distance_matrix >
                               torch.arange(0, self._max_distance).float()).float()

        # Part 2 for one hot encoding of bond type.
        num_atoms = mol.GetNumAtoms()
        bond_indicators = torch.zeros(num_atoms, num_atoms, len(self._bond_types))
        for bond in mol.GetBonds():
            bond_type_encoding = torch.tensor(
                bond_type_one_hot(bond, allowable_set=self._bond_types)).float()
            begin_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_indicators[begin_atom_idx, end_atom_idx] = bond_type_encoding
            bond_indicators[end_atom_idx, begin_atom_idx] = bond_type_encoding
        # Reshape from (V, V, num_bond_types) to (V^2, num_bond_types)
        bond_indicators = bond_indicators.reshape(-1, len(self._bond_types))

        # Part 3 for whether a pair of atoms belongs to a same ring.
        sssr = Chem.GetSymmSSSR(mol)
        ring_mate_indicators = torch.zeros(num_atoms, num_atoms, 1)
        for ring in sssr:
            ring = list(ring)
            num_atoms_in_ring = len(ring)
            for i in range(num_atoms_in_ring):
                ring_mate_indicators[ring[i], torch.tensor(ring)] = 1
        ring_mate_indicators = ring_mate_indicators.reshape(-1, 1)

        return {self._edge_data_field: torch.cat([distance_indicators,
                                                  bond_indicators,
                                                  ring_mate_indicators], dim=1)}
