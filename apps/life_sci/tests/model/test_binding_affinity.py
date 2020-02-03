import dgl
import os
import shutil
import torch

from dgl.data.utils import _get_dgl_url, download, extract_archive

from dgllife.model.model_zoo.acnn import ACNN
from dgllife.utils.complex_to_graph import ACNN_graph_construction_and_featurization
from dgllife.utils.rdkit_utils import load_molecule

def remove_dir(dir):
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
        except OSError:
            pass

def test_acnn():
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

    remove_dir('tmp1')
    remove_dir('tmp2')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g1 = ACNN_graph_construction_and_featurization(ligand_mol,
                                                   pocket_mol,
                                                   ligand_coords,
                                                   pocket_coords)

    model = ACNN()
    model.to(device)
    g1.to(device)
    assert model(g1).shape == torch.Size([1, 1])

    bg = dgl.batch_hetero([g1, g1])
    bg.to(device)
    assert model(bg).shape == torch.Size([2, 1])

    model = ACNN(hidden_sizes=[1, 2],
                 weight_init_stddevs=[1, 1],
                 dropouts=[0.1, 0.],
                 features_to_use=torch.tensor([6., 8.]),
                 radial=[[12.0], [0.0, 2.0], [4.0]])
    model.to(device)
    g1.to(device)
    assert model(g1).shape == torch.Size([1, 1])

    bg = dgl.batch_hetero([g1, g1])
    bg.to(device)
    assert model(bg).shape == torch.Size([2, 1])

if __name__ == '__main__':
    test_acnn()
