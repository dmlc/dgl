import os
import shutil

from dgl.data.utils import download, _get_dgl_url, extract_archive
from dglchem.utils.complex_to_graph import *
from dglchem.utils.rdkit_utils import load_molecule

def remove_dir(dir):
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
        except OSError:
            pass

def test_acnn_graph_construction_and_featurization():
    remove_dir('tmp1')
    remove_dir('tmp2')

    url = _get_dgl_url('dglchem/example_mols.tar.gz')
    local_path = 'tmp1/example_mols.tar.gz'
    download(url, path=local_path)
    extract_archive(local_path, 'tmp2')

    pocket_mol, pocket_coords = load_molecule(
        'tmp2/example_mols/example.pdb', remove_hs=True)
    ligand_mol, ligand_coords = load_molecule(
        'tmp2/example_mols/example.pdbqt', remove_hs=True)

    remove_dir('tmp1')
    remove_dir('tmp2')

if __name__ == '__main__':
    test_acnn_graph_construction_and_featurization()
