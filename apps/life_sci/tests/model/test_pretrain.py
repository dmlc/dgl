import dgl
import os
import torch

from dglls.model import load_pretrained
from dglls.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

def remove_file(fname):
    if os.path.isfile(fname):
        try:
            os.remove(fname)
        except OSError:
            pass

def run_dgmg_ChEMBL(model):
    assert model(
        actions=[(0, 2), (1, 3), (0, 0), (1, 0), (2, 0), (1, 3), (0, 7)],
        rdkit_mol=True) == 'CO'
    assert model(rdkit_mol=False) is None
    model.eval()
    assert model(rdkit_mol=True) is not None

def run_dgmg_ZINC(model):
    assert model(
        actions=[(0, 2), (1, 3), (0, 5), (1, 0), (2, 0), (1, 3), (0, 9)],
        rdkit_mol=True) == 'CO'
    assert model(rdkit_mol=False) is None
    model.eval()
    assert model(rdkit_mol=True) is not None

def test_dgmg():
    model = load_pretrained('DGMG_ZINC_canonical')
    run_dgmg_ZINC(model)
    model = load_pretrained('DGMG_ZINC_random')
    run_dgmg_ZINC(model)
    model = load_pretrained('DGMG_ChEMBL_canonical')
    run_dgmg_ChEMBL(model)
    model = load_pretrained('DGMG_ChEMBL_random')
    run_dgmg_ChEMBL(model)

    remove_file('DGMG_ChEMBL_canonical_pre_trained.pth')
    remove_file('DGMG_ChEMBL_random_pre_trained.pth')
    remove_file('DGMG_ZINC_canonical_pre_trained.pth')
    remove_file('DGMG_ZINC_random_pre_trained.pth')

def test_jtnn():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = load_pretrained('JTNN_ZINC').to(device)

    remove_file('JTNN_ZINC_pre_trained.pth')

def test_gcn_tox21():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    node_featurizer = CanonicalAtomFeaturizer()
    g1 = smiles_to_bigraph('CO', node_featurizer=node_featurizer)
    g2 = smiles_to_bigraph('CCO', node_featurizer=node_featurizer)
    bg = dgl.batch([g1, g2])
    model = load_pretrained('GCN_Tox21').to(device)
    model(bg.to(device), bg.ndata.pop('h').to(device))
    model.eval()
    model(g1.to(device), g1.ndata.pop('h').to(device))

    remove_file('GCN_Tox21_pre_trained.pth')

def test_gat_tox21():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    node_featurizer = CanonicalAtomFeaturizer()
    g1 = smiles_to_bigraph('CO', node_featurizer=node_featurizer)
    g2 = smiles_to_bigraph('CCO', node_featurizer=node_featurizer)
    bg = dgl.batch([g1, g2])
    model = load_pretrained('GAT_Tox21').to(device)
    model(bg.to(device), bg.ndata.pop('h').to(device))
    model.eval()
    model(g1.to(device), g1.ndata.pop('h').to(device))

    remove_file('GAT_Tox21_pre_trained.pth')

if __name__ == '__main__':
    test_dgmg()
    test_jtnn()
    test_gcn_tox21()
    test_gat_tox21()
