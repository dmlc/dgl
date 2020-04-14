import dgl
import os
import torch

from functools import partial

from dgllife.model import load_pretrained
from dgllife.utils import *

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

def test_weave_tox21():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    node_featurizer = WeaveAtomFeaturizer()
    edge_featurizer = WeaveEdgeFeaturizer(max_distance=2)
    g1 = smiles_to_complete_graph('CO', node_featurizer=node_featurizer,
                                  edge_featurizer=edge_featurizer, add_self_loop=True)
    g2 = smiles_to_complete_graph('CCO', node_featurizer=node_featurizer,
                                  edge_featurizer=edge_featurizer, add_self_loop=True)
    bg = dgl.batch([g1, g2])

    model = load_pretrained('Weave_Tox21').to(device)
    model(bg.to(device), bg.ndata.pop('h').to(device), bg.edata.pop('e').to(device))
    model.eval()
    model(g1.to(device), g1.ndata.pop('h').to(device), g1.edata.pop('e').to(device))

    remove_file('Weave_Tox21_pre_trained.pth')

def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

def test_attentivefp_aromaticity():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    node_featurizer = BaseAtomFeaturizer(
        featurizer_funcs={'hv': ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=[
                'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                    encode_unknown=True),
            partial(atom_degree_one_hot, allowable_set=list(range(6))),
            atom_formal_charge, atom_num_radical_electrons,
            partial(atom_hybridization_one_hot, encode_unknown=True),
            lambda atom: [0], # A placeholder for aromatic information,
            atom_total_num_H_one_hot, chirality
        ],
        )}
    )
    edge_featurizer = BaseBondFeaturizer({
        'he': lambda bond: [0 for _ in range(10)]
    })
    g1 = smiles_to_bigraph('CO', node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer)
    g2 = smiles_to_bigraph('CCO', node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer)
    bg = dgl.batch([g1, g2])

    model = load_pretrained('AttentiveFP_Aromaticity').to(device)
    model(bg.to(device), bg.ndata.pop('hv').to(device), bg.edata.pop('he').to(device))
    model.eval()
    model(g1.to(device), g1.ndata.pop('hv').to(device), g1.edata.pop('he').to(device))

    remove_file('AttentiveFP_Aromaticity_pre_trained.pth')

if __name__ == '__main__':
    test_dgmg()
    test_jtnn()
    test_gcn_tox21()
    test_gat_tox21()
    test_attentivefp_aromaticity()
