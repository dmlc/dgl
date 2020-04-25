import torch

from dgllife.utils.featurizers import *
from rdkit import Chem

def test_one_hot_encoding():
    x = 1.
    allowable_set = [0., 1., 2.]
    assert one_hot_encoding(x, allowable_set) == [0, 1, 0]
    assert one_hot_encoding(x, allowable_set, encode_unknown=True) == [0, 1, 0, 0]

    assert one_hot_encoding(x, allowable_set) == [0, 1, 0, 0]
    assert one_hot_encoding(x, allowable_set, encode_unknown=True) == [0, 1, 0, 0]
    assert one_hot_encoding(-1, allowable_set, encode_unknown=True) == [0, 0, 0, 1]

def test_mol1():
    return Chem.MolFromSmiles('CCO')

def test_mol2():
    return Chem.MolFromSmiles('C1=CC2=CC=CC=CC2=C1')

def test_atom_type_one_hot():
    mol = test_mol1()
    assert atom_type_one_hot(mol.GetAtomWithIdx(0), ['C', 'O']) == [1, 0]
    assert atom_type_one_hot(mol.GetAtomWithIdx(2), ['C', 'O']) == [0, 1]

def test_atomic_number_one_hot():
    mol = test_mol1()
    assert atomic_number_one_hot(mol.GetAtomWithIdx(0), [6, 8]) == [1, 0]
    assert atomic_number_one_hot(mol.GetAtomWithIdx(2), [6, 8]) == [0, 1]

def test_atomic_number():
    mol = test_mol1()
    assert atomic_number(mol.GetAtomWithIdx(0)) == [6]
    assert atomic_number(mol.GetAtomWithIdx(2)) == [8]

def test_atom_degree_one_hot():
    mol = test_mol1()
    assert atom_degree_one_hot(mol.GetAtomWithIdx(0), [0, 1, 2]) == [0, 1, 0]
    assert atom_degree_one_hot(mol.GetAtomWithIdx(1), [0, 1, 2]) == [0, 0, 1]

def test_atom_degree():
    mol = test_mol1()
    assert atom_degree(mol.GetAtomWithIdx(0)) == [1]
    assert atom_degree(mol.GetAtomWithIdx(1)) == [2]

def test_atom_total_degree_one_hot():
    mol = test_mol1()
    assert atom_total_degree_one_hot(mol.GetAtomWithIdx(0), [0, 2, 4]) == [0, 0, 1]
    assert atom_total_degree_one_hot(mol.GetAtomWithIdx(2), [0, 2, 4]) == [0, 1, 0]

def test_atom_total_degree():
    mol = test_mol1()
    assert atom_total_degree(mol.GetAtomWithIdx(0)) == [4]
    assert atom_total_degree(mol.GetAtomWithIdx(2)) == [2]

def test_atom_explicit_valence_one_hot():
    mol = test_mol1()
    assert atom_implicit_valence_one_hot(mol.GetAtomWithIdx(0), [1, 2, 3]) == [1, 0, 0]
    assert atom_implicit_valence_one_hot(mol.GetAtomWithIdx(1), [1, 2, 3]) == [0, 1, 0]

def test_atom_explicit_valence():
    mol = test_mol1()
    assert atom_explicit_valence(mol.GetAtomWithIdx(0)) == [1]
    assert atom_explicit_valence(mol.GetAtomWithIdx(1)) == [2]

def test_atom_implicit_valence_one_hot():
    mol = test_mol1()
    assert atom_implicit_valence_one_hot(mol.GetAtomWithIdx(0), [1, 2, 3]) == [0, 0, 1]
    assert atom_implicit_valence_one_hot(mol.GetAtomWithIdx(1), [1, 2, 3]) == [0, 1, 0]

def test_atom_implicit_valence():
    mol = test_mol1()
    assert atom_implicit_valence(mol.GetAtomWithIdx(0)) == [3]
    assert atom_implicit_valence(mol.GetAtomWithIdx(1)) == [2]

def test_atom_hybridization_one_hot():
    mol = test_mol1()
    assert atom_hybridization_one_hot(mol.GetAtomWithIdx(0)) == [0, 0, 1, 0, 0]

def test_atom_total_num_H_one_hot():
    mol = test_mol1()
    assert atom_total_num_H_one_hot(mol.GetAtomWithIdx(0)) == [0, 0, 0, 1, 0]
    assert atom_total_num_H_one_hot(mol.GetAtomWithIdx(1)) == [0, 0, 1, 0, 0]

def test_atom_total_num_H():
    mol = test_mol1()
    assert atom_total_num_H(mol.GetAtomWithIdx(0)) == [3]
    assert atom_total_num_H(mol.GetAtomWithIdx(1)) == [2]

def test_atom_formal_charge_one_hot():
    mol = test_mol1()
    assert atom_formal_charge_one_hot(mol.GetAtomWithIdx(0)) == [0, 0, 1, 0, 0]

def test_atom_formal_charge():
    mol = test_mol1()
    assert atom_formal_charge(mol.GetAtomWithIdx(0)) == [0]

def test_atom_num_radical_electrons_one_hot():
    mol = test_mol1()
    assert atom_num_radical_electrons_one_hot(mol.GetAtomWithIdx(0)) == [1, 0, 0, 0, 0]

def test_atom_num_radical_electrons():
    mol = test_mol1()
    assert atom_num_radical_electrons(mol.GetAtomWithIdx(0)) == [0]

def test_atom_is_aromatic_one_hot():
    mol = test_mol1()
    assert atom_is_aromatic_one_hot(mol.GetAtomWithIdx(0)) == [1, 0]
    mol = test_mol2()
    assert atom_is_aromatic_one_hot(mol.GetAtomWithIdx(0)) == [0, 1]

def test_atom_is_aromatic():
    mol = test_mol1()
    assert atom_is_aromatic(mol.GetAtomWithIdx(0)) == [0]
    mol = test_mol2()
    assert atom_is_aromatic(mol.GetAtomWithIdx(0)) == [1]

def test_atom_is_in_ring_one_hot():
    mol = test_mol1()
    assert atom_is_in_ring_one_hot(mol.GetAtomWithIdx(0)) == [1, 0]
    mol = test_mol2()
    assert atom_is_in_ring_one_hot(mol.GetAtomWithIdx(0)) == [0, 1]

def test_atom_is_in_ring():
    mol = test_mol1()
    assert atom_is_in_ring(mol.GetAtomWithIdx(0)) == [0]
    mol = test_mol2()
    assert atom_is_in_ring(mol.GetAtomWithIdx(0)) == [1]

def test_atom_chiral_tag_one_hot():
    mol = test_mol1()
    assert atom_chiral_tag_one_hot(mol.GetAtomWithIdx(0)) == [1, 0, 0, 0]

def test_atom_mass():
    mol = test_mol1()
    atom = mol.GetAtomWithIdx(0)
    assert atom_mass(atom) == [atom.GetMass() * 0.01]
    atom = mol.GetAtomWithIdx(1)
    assert atom_mass(atom) == [atom.GetMass() * 0.01]

def test_concat_featurizer():
    test_featurizer = ConcatFeaturizer(
        [atom_is_aromatic_one_hot, atom_chiral_tag_one_hot]
    )
    mol = test_mol1()
    assert test_featurizer(mol.GetAtomWithIdx(0)) == [1, 0, 1, 0, 0, 0]
    mol = test_mol2()
    assert test_featurizer(mol.GetAtomWithIdx(0)) == [0, 1, 1, 0, 0, 0]

class TestAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self):
        super(TestAtomFeaturizer, self).__init__(
            featurizer_funcs={
                'h1': ConcatFeaturizer([atom_total_degree_one_hot,
                                        atom_formal_charge_one_hot]),
                'h2': ConcatFeaturizer([atom_num_radical_electrons_one_hot])
            }
        )

def test_base_atom_featurizer():
    test_featurizer = TestAtomFeaturizer()
    assert test_featurizer.feat_size('h1') == 11
    assert test_featurizer.feat_size('h2') == 5
    mol = test_mol1()
    feats = test_featurizer(mol)
    assert torch.allclose(feats['h1'],
                          torch.tensor([[0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
                                        [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.]]))
    assert torch.allclose(feats['h2'],
                          torch.tensor([[1., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0.]]))

def test_canonical_atom_featurizer():
    test_featurizer = CanonicalAtomFeaturizer()
    assert test_featurizer.feat_size() == 74
    assert test_featurizer.feat_size('h') == 74
    mol = test_mol1()
    feats = test_featurizer(mol)
    assert list(feats.keys()) == ['h']
    assert torch.allclose(feats['h'],
                          torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                         0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                         1., 0.],
                                        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                                         0., 0.],
                                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
                                         0., 0.]]))

def test_weave_atom_featurizer():
    featurizer = WeaveAtomFeaturizer()
    assert featurizer.feat_size() == 27
    mol = test_mol1()
    feats = featurizer(mol)
    assert list(feats.keys()) == ['h']
    assert torch.allclose(feats['h'],
                          torch.tensor([[0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, -0.0418, 0.0000, 0.0000, 0.0000,
                                         1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000],
                                        [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0402, 0.0000, 0.0000, 0.0000,
                                         1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000],
                                        [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, -0.3967, 0.0000, 0.0000, 0.0000,
                                         1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000]]), rtol=1e-3)

def test_bond_type_one_hot():
    mol = test_mol1()
    assert bond_type_one_hot(mol.GetBondWithIdx(0)) == [1, 0, 0, 0]
    mol = test_mol2()
    assert bond_type_one_hot(mol.GetBondWithIdx(0)) == [0, 0, 0, 1]

def test_bond_is_conjugated_one_hot():
    mol = test_mol1()
    assert bond_is_conjugated_one_hot(mol.GetBondWithIdx(0)) == [1, 0]
    mol = test_mol2()
    assert bond_is_conjugated_one_hot(mol.GetBondWithIdx(0)) == [0, 1]

def test_bond_is_conjugated():
    mol = test_mol1()
    assert bond_is_conjugated(mol.GetBondWithIdx(0)) == [0]
    mol = test_mol2()
    assert bond_is_conjugated(mol.GetBondWithIdx(0)) == [1]

def test_bond_is_in_ring_one_hot():
    mol = test_mol1()
    assert bond_is_in_ring_one_hot(mol.GetBondWithIdx(0)) == [1, 0]
    mol = test_mol2()
    assert bond_is_in_ring_one_hot(mol.GetBondWithIdx(0)) == [0, 1]

def test_bond_is_in_ring():
    mol = test_mol1()
    assert bond_is_in_ring(mol.GetBondWithIdx(0)) == [0]
    mol = test_mol2()
    assert bond_is_in_ring(mol.GetBondWithIdx(0)) == [1]

def test_bond_stereo_one_hot():
    mol = test_mol1()
    assert bond_stereo_one_hot(mol.GetBondWithIdx(0)) == [1, 0, 0, 0, 0, 0]

class TestBondFeaturizer(BaseBondFeaturizer):
    def __init__(self):
        super(TestBondFeaturizer, self).__init__(
            featurizer_funcs={
                'h1': ConcatFeaturizer([bond_is_in_ring, bond_is_conjugated]),
                'h2': ConcatFeaturizer([bond_stereo_one_hot])
            }
        )

def test_base_bond_featurizer():
    test_featurizer = TestBondFeaturizer()
    assert test_featurizer.feat_size('h1') == 2
    assert test_featurizer.feat_size('h2') == 6
    mol = test_mol1()
    feats = test_featurizer(mol)
    assert torch.allclose(feats['h1'], torch.tensor([[0., 0.], [0., 0.], [0., 0.], [0., 0.]]))
    assert torch.allclose(feats['h2'], torch.tensor([[1., 0., 0., 0., 0., 0.],
                                                     [1., 0., 0., 0., 0., 0.],
                                                     [1., 0., 0., 0., 0., 0.],
                                                     [1., 0., 0., 0., 0., 0.]]))

def test_canonical_bond_featurizer():
    test_featurizer = CanonicalBondFeaturizer()
    assert test_featurizer.feat_size() == 12
    assert test_featurizer.feat_size('e') == 12
    mol = test_mol1()
    feats = test_featurizer(mol)
    assert torch.allclose(feats['e'], torch.tensor(
        [[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]))

def test_weave_edge_featurizer():
    test_featurizer = WeaveEdgeFeaturizer()
    assert test_featurizer.feat_size() == 12
    mol = test_mol1()
    feats = test_featurizer(mol)
    assert torch.allclose(feats['e'],
                          torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                        [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))

if __name__ == '__main__':
    test_one_hot_encoding()
    test_atom_type_one_hot()
    test_atomic_number_one_hot()
    test_atomic_number()
    test_atom_degree_one_hot()
    test_atom_degree()
    test_atom_total_degree_one_hot()
    test_atom_total_degree()
    test_atom_explicit_valence()
    test_atom_implicit_valence_one_hot()
    test_atom_implicit_valence()
    test_atom_hybridization_one_hot()
    test_atom_total_num_H_one_hot()
    test_atom_total_num_H()
    test_atom_formal_charge_one_hot()
    test_atom_formal_charge()
    test_atom_num_radical_electrons_one_hot()
    test_atom_num_radical_electrons()
    test_atom_is_aromatic_one_hot()
    test_atom_is_aromatic()
    test_atom_is_in_ring_one_hot()
    test_atom_is_in_ring()
    test_atom_chiral_tag_one_hot()
    test_atom_mass()
    test_concat_featurizer()
    test_base_atom_featurizer()
    test_canonical_atom_featurizer()
    test_weave_atom_featurizer()
    test_bond_type_one_hot()
    test_bond_is_conjugated_one_hot()
    test_bond_is_conjugated()
    test_bond_is_in_ring_one_hot()
    test_bond_is_in_ring()
    test_bond_stereo_one_hot()
    test_base_bond_featurizer()
    test_canonical_bond_featurizer()
    test_weave_edge_featurizer()
