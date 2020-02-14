import numpy as np
import torch

from dgllife.utils.featurizers import *
from dgllife.utils.mol_to_graph import *
from rdkit import Chem

test_smiles1 = 'CCO'
test_smiles2 = 'Fc1ccccc1'

class TestAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self):
        super(TestAtomFeaturizer, self).__init__(
            featurizer_funcs={'hv': ConcatFeaturizer([atomic_number])})

class TestBondFeaturizer(BaseBondFeaturizer):
    def __init__(self):
        super(TestBondFeaturizer, self).__init__(
            featurizer_funcs={'he': ConcatFeaturizer([bond_is_in_ring])})

def test_smiles_to_bigraph():
    # Test the case with self loops added.
    g1 = smiles_to_bigraph(test_smiles1, add_self_loop=True)
    src, dst = g1.edges()
    assert torch.allclose(src, torch.LongTensor([0, 2, 2, 1, 0, 1, 2]))
    assert torch.allclose(dst, torch.LongTensor([2, 0, 1, 2, 0, 1, 2]))

    # Test the case without self loops.
    test_node_featurizer = TestAtomFeaturizer()
    test_edge_featurizer = TestBondFeaturizer()
    g2 = smiles_to_bigraph(test_smiles2, add_self_loop=False,
                           node_featurizer=test_node_featurizer,
                           edge_featurizer=test_edge_featurizer)
    assert torch.allclose(g2.ndata['hv'], torch.tensor([[9.], [6.], [6.], [6.],
                                                        [6.], [6.], [6.]]))
    assert torch.allclose(g2.edata['he'], torch.tensor([[0.], [0.], [1.], [1.], [1.],
                                                        [1.], [1.], [1.], [1.], [1.],
                                                        [1.], [1.], [1.], [1.]]))

def test_mol_to_bigraph():
    mol1 = Chem.MolFromSmiles(test_smiles1)
    g1 = mol_to_bigraph(mol1, add_self_loop=True)
    src, dst = g1.edges()
    assert torch.allclose(src, torch.LongTensor([0, 2, 2, 1, 0, 1, 2]))
    assert torch.allclose(dst, torch.LongTensor([2, 0, 1, 2, 0, 1, 2]))

    # Test the case without self loops.
    mol2 = Chem.MolFromSmiles(test_smiles2)
    test_node_featurizer = TestAtomFeaturizer()
    test_edge_featurizer = TestBondFeaturizer()
    g2 = mol_to_bigraph(mol2, add_self_loop=False,
                        node_featurizer=test_node_featurizer,
                        edge_featurizer=test_edge_featurizer)
    assert torch.allclose(g2.ndata['hv'], torch.tensor([[9.], [6.], [6.], [6.],
                                                        [6.], [6.], [6.]]))
    assert torch.allclose(g2.edata['he'], torch.tensor([[0.], [0.], [1.], [1.], [1.],
                                                        [1.], [1.], [1.], [1.], [1.],
                                                        [1.], [1.], [1.], [1.]]))

def test_smiles_to_complete_graph():
    test_node_featurizer = TestAtomFeaturizer()
    g = smiles_to_complete_graph(test_smiles1, add_self_loop=False,
                                 node_featurizer=test_node_featurizer)
    src, dst = g.edges()
    assert torch.allclose(src, torch.LongTensor([0, 0, 1, 1, 2, 2]))
    assert torch.allclose(dst, torch.LongTensor([1, 2, 0, 2, 0, 1]))
    assert torch.allclose(g.ndata['hv'], torch.tensor([[6.], [8.], [6.]]))

def test_mol_to_complete_graph():
    test_node_featurizer = TestAtomFeaturizer()
    mol1 = Chem.MolFromSmiles(test_smiles1)
    g = mol_to_complete_graph(mol1, add_self_loop=False,
                              node_featurizer=test_node_featurizer)
    src, dst = g.edges()
    assert torch.allclose(src, torch.LongTensor([0, 0, 1, 1, 2, 2]))
    assert torch.allclose(dst, torch.LongTensor([1, 2, 0, 2, 0, 1]))
    assert torch.allclose(g.ndata['hv'], torch.tensor([[6.], [8.], [6.]]))

def test_k_nearest_neighbors():
    coordinates = np.array([[0.1, 0.1, 0.1],
                            [0.2, 0.1, 0.1],
                            [0.15, 0.15, 0.1],
                            [0.1, 0.15, 0.16],
                            [1.2, 0.1, 0.1],
                            [1.3, 0.2, 0.1]])
    neighbor_cutoff = 1.
    max_num_neighbors = 2
    srcs, dsts, dists = k_nearest_neighbors(coordinates, neighbor_cutoff, max_num_neighbors)
    assert srcs == [2, 3, 2, 0, 0, 1, 0, 2, 5, 4]
    assert dsts == [0, 0, 1, 1, 2, 2, 3, 3, 4, 5]
    assert dists == [0.07071067811865474,
                     0.07810249675906654,
                     0.07071067811865477,
                     0.1,
                     0.07071067811865474,
                     0.07071067811865477,
                     0.07810249675906654,
                     0.07810249675906654,
                     0.14142135623730956,
                     0.14142135623730956]

if __name__ == '__main__':
    test_smiles_to_bigraph()
    test_mol_to_bigraph()
    test_smiles_to_complete_graph()
    test_mol_to_complete_graph()
    test_k_nearest_neighbors()
