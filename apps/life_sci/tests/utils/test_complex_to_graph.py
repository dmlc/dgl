import os
import shutil
import torch

from dgl.data.utils import download, _get_dgl_url, extract_archive
from dgllife.utils.complex_to_graph import *
from dgllife.utils.rdkit_utils import load_molecule

def remove_dir(dir):
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
        except OSError:
            pass

def test_acnn_graph_construction_and_featurization():
    remove_dir('tmp1')
    remove_dir('tmp2')

    url = _get_dgl_url('dgllife/example_mols.tar.gz')
    local_path = 'tmp1/example_mols.tar.gz'
    download(url, path=local_path)
    extract_archive(local_path, 'tmp2')

    pocket_mol, pocket_coords = load_molecule(
        'tmp2/example_mols/example.pdb', remove_hs=True)
    ligand_mol, ligand_coords = load_molecule(
        'tmp2/example_mols/example.pdbqt', remove_hs=True)
    pocket_mol_with_h, pocket_coords_with_h = load_molecule(
        'tmp2/example_mols/example.pdb', remove_hs=False)

    remove_dir('tmp1')
    remove_dir('tmp2')

    # Test default case
    g = ACNN_graph_construction_and_featurization(ligand_mol,
                                                  pocket_mol,
                                                  ligand_coords,
                                                  pocket_coords)
    assert set(g.ntypes) == set(['protein_atom', 'ligand_atom'])
    assert set(g.etypes) == set(['protein', 'ligand', 'complex', 'complex', 'complex', 'complex'])
    assert g.number_of_nodes('protein_atom') == 286
    assert g.number_of_nodes('ligand_atom') == 21

    assert g.number_of_edges('protein') == 3432
    assert g.number_of_edges('ligand') == 252
    assert g.number_of_edges(('protein_atom', 'complex', 'protein_atom')) == 3349
    assert g.number_of_edges(('ligand_atom', 'complex', 'ligand_atom')) == 131
    assert g.number_of_edges(('protein_atom', 'complex', 'ligand_atom')) == 121
    assert g.number_of_edges(('ligand_atom', 'complex', 'protein_atom')) == 83

    assert 'atomic_number' in g.nodes['protein_atom'].data
    assert 'atomic_number' in g.nodes['ligand_atom'].data
    assert torch.allclose(g.nodes['protein_atom'].data['mask'],
                          torch.ones(g.number_of_nodes('protein_atom'), 1))
    assert torch.allclose(g.nodes['ligand_atom'].data['mask'],
                          torch.ones(g.number_of_nodes('ligand_atom'), 1))
    assert 'distance' in g.edges['protein'].data
    assert 'distance' in g.edges['ligand'].data
    assert 'distance' in g.edges[('protein_atom', 'complex', 'protein_atom')].data
    assert 'distance' in g.edges[('ligand_atom', 'complex', 'ligand_atom')].data
    assert 'distance' in g.edges[('protein_atom', 'complex', 'ligand_atom')].data
    assert 'distance' in g.edges[('ligand_atom', 'complex', 'protein_atom')].data

    # Test max_num_ligand_atoms and max_num_protein_atoms
    max_num_ligand_atoms = 30
    max_num_protein_atoms = 300
    g = ACNN_graph_construction_and_featurization(ligand_mol,
                                                  pocket_mol,
                                                  ligand_coords,
                                                  pocket_coords,
                                                  max_num_ligand_atoms=max_num_ligand_atoms,
                                                  max_num_protein_atoms=max_num_protein_atoms)
    assert g.number_of_nodes('ligand_atom') == max_num_ligand_atoms
    assert g.number_of_nodes('protein_atom') == max_num_protein_atoms
    ligand_mask = torch.zeros(max_num_ligand_atoms, 1)
    ligand_mask[:ligand_mol.GetNumAtoms(), :] = 1.
    assert torch.allclose(ligand_mask, g.nodes['ligand_atom'].data['mask'])
    protein_mask = torch.zeros(max_num_protein_atoms, 1)
    protein_mask[:pocket_mol.GetNumAtoms(), :] = 1.
    assert torch.allclose(protein_mask, g.nodes['protein_atom'].data['mask'])

    # Test neighbor_cutoff
    neighbor_cutoff = 6.
    g = ACNN_graph_construction_and_featurization(ligand_mol,
                                                  pocket_mol,
                                                  ligand_coords,
                                                  pocket_coords,
                                                  neighbor_cutoff=neighbor_cutoff)
    assert g.number_of_edges('protein') == 3405
    assert g.number_of_edges('ligand') == 193
    assert g.number_of_edges(('protein_atom', 'complex', 'protein_atom')) == 3331
    assert g.number_of_edges(('ligand_atom', 'complex', 'ligand_atom')) == 123
    assert g.number_of_edges(('protein_atom', 'complex', 'ligand_atom')) == 119
    assert g.number_of_edges(('ligand_atom', 'complex', 'protein_atom')) == 82

    # Test max_num_neighbors
    g = ACNN_graph_construction_and_featurization(ligand_mol,
                                                  pocket_mol,
                                                  ligand_coords,
                                                  pocket_coords,
                                                  max_num_neighbors=6)
    assert g.number_of_edges('protein') == 1716
    assert g.number_of_edges('ligand') == 126
    assert g.number_of_edges(('protein_atom', 'complex', 'protein_atom')) == 1691
    assert g.number_of_edges(('ligand_atom', 'complex', 'ligand_atom')) == 86
    assert g.number_of_edges(('protein_atom', 'complex', 'ligand_atom')) == 40
    assert g.number_of_edges(('ligand_atom', 'complex', 'protein_atom')) == 25

    # Test strip_hydrogens
    g = ACNN_graph_construction_and_featurization(pocket_mol_with_h,
                                                  pocket_mol_with_h,
                                                  pocket_coords_with_h,
                                                  pocket_coords_with_h,
                                                  strip_hydrogens=True)
    assert g.number_of_nodes('ligand_atom') != pocket_mol_with_h.GetNumAtoms()
    assert g.number_of_nodes('protein_atom') != pocket_mol_with_h.GetNumAtoms()
    non_h_atomic_numbers = []
    for i in range(pocket_mol_with_h.GetNumAtoms()):
        atom = pocket_mol_with_h.GetAtomWithIdx(i)
        if atom.GetSymbol() != 'H':
            non_h_atomic_numbers.append(atom.GetAtomicNum())
    non_h_atomic_numbers = torch.tensor(non_h_atomic_numbers).float().reshape(-1, 1)
    assert torch.allclose(non_h_atomic_numbers, g.nodes['ligand_atom'].data['atomic_number'])
    assert torch.allclose(non_h_atomic_numbers, g.nodes['protein_atom'].data['atomic_number'])

if __name__ == '__main__':
    test_acnn_graph_construction_and_featurization()
