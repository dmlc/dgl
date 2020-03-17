import numpy as np
import torch

from dgllife.utils.featurizers import *
from dgllife.utils.mol_to_graph import *
from rdkit import Chem
from rdkit.Chem import AllChem

test_smiles1 = 'CCO'
test_smiles2 = 'Fc1ccccc1'
test_smiles3 = '[CH2:1]([CH3:2])[N:3]1[CH2:4][CH2:5][C:6]([CH3:16])' \
               '([CH3:17])[c:7]2[cH:8][cH:9][c:10]([N+:13]([O-:14])=[O:15])' \
               '[cH:11][c:12]21.[CH3:18][CH2:19][O:20][C:21]([CH3:22])=[O:23]'

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

    # Test the case where atoms come with a default order and we do not
    # want to change the order, which is related to the application of
    # reaction center prediction.
    g3 = smiles_to_bigraph(test_smiles3, node_featurizer=test_node_featurizer,
                           canonical_atom_order=False)
    assert torch.allclose(g3.ndata['hv'], torch.tensor([[6.], [6.], [7.], [6.], [6.], [6.],
                                                        [6.], [6.], [6.], [6.], [6.], [6.],
                                                        [7.], [8.], [8.], [6.], [6.], [6.],
                                                        [6.], [8.], [6.], [6.], [8.]]))

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

    # Test the case where atoms come with a default order and we do not
    # want to change the order, which is related to the application of
    # reaction center prediction.
    mol3 = Chem.MolFromSmiles(test_smiles3)
    g3 = mol_to_bigraph(mol3, node_featurizer=test_node_featurizer,
                        canonical_atom_order=False)
    assert torch.allclose(g3.ndata['hv'], torch.tensor([[6.], [6.], [7.], [6.], [6.], [6.],
                                                        [6.], [6.], [6.], [6.], [6.], [6.],
                                                        [7.], [8.], [8.], [6.], [6.], [6.],
                                                        [6.], [8.], [6.], [6.], [8.]]))

def test_smiles_to_complete_graph():
    test_node_featurizer = TestAtomFeaturizer()
    g1 = smiles_to_complete_graph(test_smiles1, add_self_loop=False,
                                 node_featurizer=test_node_featurizer)
    src, dst = g1.edges()
    assert torch.allclose(src, torch.LongTensor([0, 0, 1, 1, 2, 2]))
    assert torch.allclose(dst, torch.LongTensor([1, 2, 0, 2, 0, 1]))
    assert torch.allclose(g1.ndata['hv'], torch.tensor([[6.], [8.], [6.]]))

    # Test the case where atoms come with a default order and we do not
    # want to change the order, which is related to the application of
    # reaction center prediction.
    g2 = smiles_to_complete_graph(test_smiles3, node_featurizer=test_node_featurizer,
                                  canonical_atom_order=False)
    assert torch.allclose(g2.ndata['hv'], torch.tensor([[6.], [6.], [7.], [6.], [6.], [6.],
                                                        [6.], [6.], [6.], [6.], [6.], [6.],
                                                        [7.], [8.], [8.], [6.], [6.], [6.],
                                                        [6.], [8.], [6.], [6.], [8.]]))

def test_mol_to_complete_graph():
    test_node_featurizer = TestAtomFeaturizer()
    mol1 = Chem.MolFromSmiles(test_smiles1)
    g1 = mol_to_complete_graph(mol1, add_self_loop=False,
                               node_featurizer=test_node_featurizer)
    src, dst = g1.edges()
    assert torch.allclose(src, torch.LongTensor([0, 0, 1, 1, 2, 2]))
    assert torch.allclose(dst, torch.LongTensor([1, 2, 0, 2, 0, 1]))
    assert torch.allclose(g1.ndata['hv'], torch.tensor([[6.], [8.], [6.]]))

    # Test the case where atoms come with a default order and we do not
    # want to change the order, which is related to the application of
    # reaction center prediction.
    mol2 = Chem.MolFromSmiles(test_smiles3)
    g2 = mol_to_complete_graph(mol2, node_featurizer=test_node_featurizer,
                               canonical_atom_order=False)
    assert torch.allclose(g2.ndata['hv'], torch.tensor([[6.], [6.], [7.], [6.], [6.], [6.],
                                                        [6.], [6.], [6.], [6.], [6.], [6.],
                                                        [7.], [8.], [8.], [6.], [6.], [6.],
                                                        [6.], [8.], [6.], [6.], [8.]]))

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
    assert srcs == [2, 3, 2, 0, 0, 1, 0, 2, 1, 5, 4]
    assert dsts == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert dists == [0.07071067811865478, 0.0781024967590666, 0.07071067811865483,
                     0.1, 0.07071067811865478, 0.07071067811865483, 0.0781024967590666,
                     0.0781024967590666, 1.0, 0.14142135623730956, 0.14142135623730956]

    # Test the case where self loops are included
    srcs, dsts, dists = k_nearest_neighbors(coordinates, neighbor_cutoff,
                                            max_num_neighbors, self_loops=True)
    assert srcs == [0, 2, 1, 2, 2, 0, 3, 0, 4, 5, 4, 5]
    assert dsts == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert dists == [0.0, 0.07071067811865478, 0.0, 0.07071067811865483, 0.0,
                     0.07071067811865478, 0.0, 0.0781024967590666, 0.0,
                     0.14142135623730956, 0.14142135623730956, 0.0]

    # Test the case where max_num_neighbors is not given
    srcs, dsts, dists = k_nearest_neighbors(coordinates, neighbor_cutoff=10.)
    assert srcs == [1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 4, 5,
                    0, 1, 2, 4, 5, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4]
    assert dsts == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    assert dists == [0.1, 0.07071067811865478, 0.0781024967590666, 1.1,
                     1.2041594578792296, 0.1, 0.07071067811865483,
                     0.12688577540449525, 1.0, 1.104536101718726,
                     0.07071067811865478, 0.07071067811865483,
                     0.0781024967590666, 1.0511898020814319, 1.151086443322134,
                     0.0781024967590666, 0.12688577540449525, 0.0781024967590666,
                     1.1027692415006867, 1.202538980657176, 1.1, 1.0,
                     1.0511898020814319, 1.1027692415006867, 0.14142135623730956,
                     1.2041594578792296, 1.104536101718726, 1.151086443322134,
                     1.202538980657176, 0.14142135623730956]

def test_smiles_to_nearest_neighbor_graph():
    mol = Chem.MolFromSmiles(test_smiles1)
    AllChem.EmbedMolecule(mol)
    coordinates = mol.GetConformers()[0].GetPositions()

    # Test node featurizer
    test_node_featurizer = TestAtomFeaturizer()
    g = smiles_to_nearest_neighbor_graph(test_smiles1, coordinates, neighbor_cutoff=10,
                                         node_featurizer=test_node_featurizer)
    assert torch.allclose(g.ndata['hv'], torch.tensor([[6.], [8.], [6.]]))
    assert g.number_of_edges() == 6
    assert 'dist' not in g.edata

    # Test self loops
    g = smiles_to_nearest_neighbor_graph(test_smiles1, coordinates, neighbor_cutoff=10,
                                         add_self_loop=True)
    assert g.number_of_edges() == 9

    # Test max_num_neighbors
    g = smiles_to_nearest_neighbor_graph(test_smiles1, coordinates, neighbor_cutoff=10,
                                         max_num_neighbors=1, add_self_loop=True)
    assert g.number_of_edges() == 3

    # Test pairwise distances
    g = smiles_to_nearest_neighbor_graph(test_smiles1, coordinates,
                                         neighbor_cutoff=10, keep_dists=True)
    assert 'dist' in g.edata
    coordinates = torch.from_numpy(coordinates)
    srcs, dsts = g.edges()
    dist = torch.norm(
        coordinates[srcs] - coordinates[dsts], dim=1, p=2).float().reshape(-1, 1)
    assert torch.allclose(dist, g.edata['dist'])

def test_mol_to_nearest_neighbor_graph():
    mol = Chem.MolFromSmiles(test_smiles1)
    AllChem.EmbedMolecule(mol)
    coordinates = mol.GetConformers()[0].GetPositions()

    # Test node featurizer
    test_node_featurizer = TestAtomFeaturizer()
    g = mol_to_nearest_neighbor_graph(mol, coordinates, neighbor_cutoff=10,
                                      node_featurizer=test_node_featurizer)
    assert torch.allclose(g.ndata['hv'], torch.tensor([[6.], [8.], [6.]]))
    assert g.number_of_edges() == 6
    assert 'dist' not in g.edata

    # Test self loops
    g = mol_to_nearest_neighbor_graph(mol, coordinates, neighbor_cutoff=10, add_self_loop=True)
    assert g.number_of_edges() == 9

    # Test max_num_neighbors
    g = mol_to_nearest_neighbor_graph(mol, coordinates, neighbor_cutoff=10,
                                      max_num_neighbors=1, add_self_loop=True)
    assert g.number_of_edges() == 3

    # Test pairwise distances
    g = mol_to_nearest_neighbor_graph(mol, coordinates, neighbor_cutoff=10, keep_dists=True)
    assert 'dist' in g.edata
    coordinates = torch.from_numpy(coordinates)
    srcs, dsts = g.edges()
    dist = torch.norm(
        coordinates[srcs] - coordinates[dsts], dim=1, p=2).float().reshape(-1, 1)
    assert torch.allclose(dist, g.edata['dist'])

if __name__ == '__main__':
    test_smiles_to_bigraph()
    test_mol_to_bigraph()
    test_smiles_to_complete_graph()
    test_mol_to_complete_graph()
    test_k_nearest_neighbors()
    test_smiles_to_nearest_neighbor_graph()
    test_mol_to_nearest_neighbor_graph()
