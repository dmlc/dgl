import dgl.backend as F
import numpy as np

from dgl import DGLGraph

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

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
        List of boolean values where at most one value is True.
        If the i-th value is True, then we must have
        x == allowable_set[i].
    """
    return list(map(lambda s: x == s, allowable_set))

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers

    All atom featurizers that map a molecule to atom features should subclass it.
    All subclasses should overwrite ``_featurize_atom``, which featurizes a single
    atom and ``__call__``, which featurizes all atoms in a molecule.
    """

    def _featurize_atom(self, atom):
        return NotImplementedError

    def __call__(self, mol):
        return NotImplementedError

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

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to be 'h'.
    """

    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__()
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

def smile2graph(smile, add_self_loop=False, atom_featurizer=CanonicalAtomFeaturizer(), bond_featurizer=None):
    """Convert SMILES into a DGLGraph.

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
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to CanonicalAtomFeaturizer().
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    """
    mol = Chem.MolFromSmiles(smile)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
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
