import numpy as np
import torch

from functools import partial
from rdkit import Chem

from dgllife.data import USPTO
from dgllife.utils import ConcatFeaturizer, BaseAtomFeaturizer, atom_type_one_hot, \
    atom_degree_one_hot, atom_explicit_valence_one_hot, atom_implicit_valence_one_hot, \
    atom_is_aromatic, BaseBondFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring

# Atom types distinguished in featurization
atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
              'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
              'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
              'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi',
              'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']

def atom_pair_featurizer(reactants, data_field='atom_pair'):
    """Featurize each pair of atoms, which will be used in updating
    the edata of a complete DGLGraph.

    The features include the bond type between the atoms (if any) and whether
    they belong to the same molecule. It is used in the global attention mechanism.

    Parameters
    ----------
    reactants : str
        SMILES for reactants
    data_field : str
        Key for storing the features in DGLGraph.edata. Default to 'atom_pair'

    Returns
    -------
    dict
        Mapping data_field to a float32 tensor of shape (V^2, 10), which are
        features for each pair of atoms.
    """
    # Decide the reactant membership for each atom
    atom_to_reactant = dict()
    reactant_list = reactants.split('.')
    for id, s in enumerate(reactant_list):
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            atom_to_reactant[atom.GetIntProp('molAtomMapNumber') - 1] = id

    # Construct mapping from atom pair to RDKit bond object
    all_reactant_mol = Chem.MolFromSmiles(reactants)
    atom_pair_to_bond = dict()
    for bond in all_reactant_mol.GetBonds():
        atom1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        atom2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        atom_pair_to_bond[(atom1, atom2)] = bond
        atom_pair_to_bond[(atom2, atom1)] = bond

    def _featurize_a_bond(bond):
        return bond_type_one_hot(bond) + bond_is_conjugated(bond) + bond_is_in_ring(bond)

    features = []
    num_atoms = mol.GetNumAtoms()
    for j in range(num_atoms):
        for i in range(num_atoms):
            pair_feature = np.zeros(10)
            if i == j:
                features.append(pair_feature)
                continue

            bond = atom_pair_to_bond.get((i, j), None)
            if bond is not None:
                pair_feature[1:7] = _featurize_a_bond(bond)
            else:
                pair_feature[0] = 1.
            pair_feature[-4] = 1. if atom_to_reactant[i] != atom_to_reactant[j] else 0.
            pair_feature[-3] = 1. if atom_to_reactant[i] == atom_to_reactant[j] else 0.
            pair_feature[-2] = 1. if len(reactant_list) == 1 else 0.
            pair_feature[-1] = 1. if len(reactant_list) > 1 else 0.
            features.append(pair_feature)
    return {data_field: torch.from_numpy(np.stack(features, axis=0).astype(np.float32))}

def load_data():
    """Load and pre-process the dataset.

    Construct DGLGraphs and featurize their nodes/edges.

    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    atom_featurizer = BaseAtomFeaturizer({
        'hv': ConcatFeaturizer(
            [partial(atom_type_one_hot, allowable_set=atom_types, encode_unknown=True),
             partial(atom_degree_one_hot, allowable_set=list(range(6))),
             atom_explicit_valence_one_hot,
             partial(atom_implicit_valence_one_hot, allowable_set=list(range(6))),
             atom_is_aromatic]
        )
    })
    bond_featurizer = BaseBondFeaturizer({
        'he': ConcatFeaturizer([
            bond_type_one_hot, bond_is_conjugated, bond_is_in_ring]
        )
    })
    train_set = USPTO('train', node_featurizer=atom_featurizer,
                      edge_featurizer=bond_featurizer,
                      atom_pair_featurizer=atom_pair_featurizer)
    val_set = USPTO('val', node_featurizer=atom_featurizer,
                    edge_featurizer=bond_featurizer,
                    atom_pair_featurizer=atom_pair_featurizer)
    test_set = USPTO('test', node_featurizer=atom_featurizer,
                     edge_featurizer=bond_featurizer,
                     atom_pair_featurizer=atom_pair_featurizer)

    return train_set, val_set, test_set
