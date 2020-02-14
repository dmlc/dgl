from functools import partial

from dgllife.data import USPTO
from dgllife.utils import mol_to_bigraph, ConcatFeaturizer, BaseAtomFeaturizer, \
    atom_type_one_hot, atom_degree_one_hot, atom_explicit_valence_one_hot, \
    atom_implicit_valence_one_hot, atom_is_aromatic, BaseBondFeaturizer, bond_type_one_hot, \
    bond_is_conjugated, bond_is_in_ring

# Atom types distinguished in featurization
atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
              'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
              'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
              'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi',
              'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']

def load_data():
    """Load and pre-process the dataset.

    Construct DGLGraphs and featurize their nodes/edges.
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
    train_set = USPTO('train', mol_to_bigraph, atom_featurizer, bond_featurizer)
    val_set = USPTO('val', mol_to_bigraph, atom_featurizer, bond_featurizer)
    test_set = USPTO('test', mol_to_bigraph, atom_featurizer, bond_featurizer)

    return train_set, val_set, test_set
