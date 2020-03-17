import dgl.backend as F
import numpy as np
import os.path as osp

from collections import defaultdict
from dgllife.utils.featurizers import *
from functools import partial
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures



# Constants
atom_set = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'unknown']
chiral_set = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
hybridization_set = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3]


def atom_type_one_hot(atom, allowable_set):
    """One hot encoding for the type of an atom.
    Maps inputs not in the allowable set to the last element:'unknown'.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Atom types to consider.
        Default: 'H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'P', 'I', 'unknown'.
        Size: 11
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    x = atom.GetAtomicNum()
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_partial_charge(atom):
    """Partial charge of an atom calculated from the whole molecule.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one int indicating calculated partial charge of an atom.
    """
    return [float(atom.GetProp('_GasteigerCharge'))]


class AtomFeaturizer(BaseAtomFeaturizer):
    """A default featurizer for atoms.
    The atom features include:
    * **One hot encoding of the atom type**. The supported atom types include
      'H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'P', 'I', 'unknown'. size: 11.
    * **One hot encoding of the atom chirality**. The supported chirality types include
      'R' and 'S'. size: 2.
    * **Formal charge of the atom**. size:1.
    * **Partial charge of the atom**. size: 1.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``. size: 3.
    * **Whether the atom is aromatic**. size: 1.
    * **One hot encoding of whether the atom is a hydrogen bond donor and/or acceptor**. size: 2.
    * **the number of rings that include this atom**. The supported ring size used include
      3–8. size: 6.
    **We assume the resulting DGLGraph will not contain any virtual nodes.**
    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to be 'h'.
    """

    def __init__(self, atom_data_field='h'):
        super(AtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=atom_set),
                 partial(atom_chiral_tag_one_hot, allowable_set=chiral_set, encode_unknown=False),
                 atom_formal_charge,
                 atom_partial_charge,
                 partial(atom_hybridization_one_hot, allowable_set=hybridization_set, encode_unknown=False),
                 atom_is_aromatic]
            )})

    def atom_is_H_bond_donor_or_acceptor(self, mol_feats):
        """One hot encoding of whether the atom is a hydrogen bond donor and/or acceptor.
        Parameters
        ----------
        mol_feats : tuple
            Features for molecules.
        Returns
        -------
        is_donor: defaultdict
            the atom id dict for H bond donor
        is_acceptor: : defaultdict
            the atom id dict for H bond acceptor
        """
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)
        # Get atom Ids including H bonds features
        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == 'Donor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == 'Acceptor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1
        return is_donor, is_acceptor

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
        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)
        # get molecular features
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        print(type(mol_featurizer))
        # get ring features
        sssr = Chem.GetSymmSSSR(mol)

        # Compute features for each atom
        is_donor, is_acceptor = self.atom_is_H_bond_donor_or_acceptor(mol_feats)

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))
            # add H bonds features
            if is_donor[i]:
                atom_features[feat_name][i].append(1)
            else:
                atom_features[feat_name][i].append(0)
            if is_acceptor[i]:
                atom_features[feat_name][i].append(1)
            else:
                atom_features[feat_name][i].append(0)
            # add rings features
            count = [0] * 6
            for j in range(len(sssr)):
                if i in sssr[j] and 3 <= len(sssr[j]) <= 8:
                    count[len(sssr[j])] += 1
            atom_features[feat_name][i] += count

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features


'''


class CanonicalBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_field='e'):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_in_ring]
            )})

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
        num_atoms = mol.GetNumAtoms()
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


'''
test_mol = Chem.MolFromSmiles('OC1C2C1CC2')
a = AtomFeaturizer()
b = a(test_mol)
print(b)
