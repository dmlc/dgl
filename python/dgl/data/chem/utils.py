import dgl.backend as F
import itertools
import numpy as np
from functools import partial

from collections import defaultdict
from dgl import DGLGraph

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

__all__ = ['one_hot_encoding', 'atom_type_one_hot', 'atomic_number_one_hot', 'atomic_number',
           'atom_degree_one_hot', 'atom_degree', 'atom_total_degree_one_hot', 'atom_total_degree',
           'atom_implicit_valence_one_hot', 'atom_implicit_valence', 'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot', 'atom_total_num_H', 'atom_formal_charge_one_hot',
           'atom_formal_charge', 'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons', 'atom_is_aromatic_one_hot', 'atom_is_aromatic',
           'atom_chiral_tag_one_hot', 'atom_mass', 'ConcatFeaturizer', 'BaseAtomFeaturizer',
           'CanonicalAtomFeaturizer', 'mol_to_graph', 'smiles_to_bigraph',
           'mol_to_bigraph', 'smiles_to_complete_graph', 'mol_to_complete_graph',
           'bond_type_one_hot', 'bond_is_conjugated_one_hot', 'bond_is_conjugated',
           'bond_is_in_ring_one_hot', 'bond_is_in_ring', 'bond_stereo_one_hot',
           'BaseBondFeaturizer', 'CanonicalBondFeaturizer']

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
    atom_total_degree
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
    atom_degree_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)

def atom_total_degree(atom):
    """
    See Also
    --------
    atom_degree

    Returns
    -------
    list
        List containing one int only.
    """
    return [atom.GetTotalDegree()]

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valences of an atom.

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
    """
    return [atom.GetImplicitValence()]

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
    """
    return [atom.GetFormalCharge()]

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
    """
    return [atom.GetIsAromatic()]

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

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

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

    >>> from dgl.data.chem import BaseAtomFeaturizer, atom_mass, atom_degree_one_hot
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
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name):
        """Get the feature size for ``feat_name``.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``.
        """
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
        Name for storing atom features in DGLGraphs, default to be 'h'.
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

    >>> from dgl.data.chem import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = BaseBondFeaturizer({'bond_type': bond_type_one_hot, 'in_ring': bond_is_in_ring})
    >>> bond_featurizer(mol)
    {'bond_type': tensor([[1., 0., 0., 0.],
                          [1., 0., 0., 0.],
                          [1., 0., 0., 0.],
                          [1., 0., 0., 0.]]),
     'in_ring': tensor([[0.], [0.], [0.], [0.]])}
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name):
        """Get the feature size for ``feat_name``.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``.
        """
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
    """
    def __init__(self, bond_data_field='e'):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 bond_stereo_one_hot]
            )})

#################################################################
# DGLGraph Construction
#################################################################

def mol_to_graph(mol, graph_constructor, atom_featurizer, bond_featurizer):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Takes an RDKit molecule as input and returns a DGLGraph
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Converted DGLGraph for the molecule
    """
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if atom_featurizer is not None:
        g.ndata.update(atom_featurizer(mol))

    if bond_featurizer is not None:
        g.edata.update(bond_featurizer(mol))

    return g

def construct_bigraph_from_mol(mol, add_self_loop=False):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

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
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
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

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   atom_featurizer=None,
                   bond_featurizer=None):
    """Convert an RDKit molecule object into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)

def smiles_to_bigraph(smiles, add_self_loop=False,
                      atom_featurizer=None,
                      bond_featurizer=None):
    """Convert a SMILES into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_bigraph(mol, add_self_loop, atom_featurizer, bond_featurizer)

def construct_complete_graph_from_mol(mol, add_self_loop=False):
    """Construct a complete graph with topology only for the molecule

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    If self loops are not created, we will not have (0, 0), (1, 1), ...

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty complete graph topology of the molecule
    """
    g = DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    if add_self_loop:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms)],
            [j for i in range(num_atoms) for j in range(num_atoms)])
    else:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                j for i in range(num_atoms)
                for j in range(num_atoms) if i != j
            ])

    return g

def mol_to_complete_graph(mol, add_self_loop=False,
                          atom_featurizer=None,
                          bond_featurizer=None):
    """Convert an RDKit molecule into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_complete_graph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)

def smiles_to_complete_graph(smiles, add_self_loop=False,
                             atom_featurizer=None,
                             bond_featurizer=None):
    """Convert a SMILES into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_complete_graph(mol, add_self_loop, atom_featurizer, bond_featurizer)
