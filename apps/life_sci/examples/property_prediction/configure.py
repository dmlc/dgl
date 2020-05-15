from functools import partial

# graph construction
from dgllife.utils import smiles_to_bigraph, smiles_to_complete_graph
# general featurization
from dgllife.utils import ConcatFeaturizer
# node featurization
from dgllife.utils import CanonicalAtomFeaturizer, BaseAtomFeaturizer, WeaveAtomFeaturizer, \
    atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, atom_num_radical_electrons, \
    atom_hybridization_one_hot, atom_total_num_H_one_hot
# edge featurization
from dgllife.utils.featurizers import BaseBondFeaturizer, WeaveEdgeFeaturizer

from utils import chirality

GCN_Tox21 = {
    'random_seed': 2,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'node_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'classifier_hidden_feats': 64,
    'patience': 10,
    'smiles_to_graph': smiles_to_bigraph,
    'node_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc_score'
}

GAT_Tox21 = {
    'random_seed': 2,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'node_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'smiles_to_graph': smiles_to_bigraph,
    'node_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc_score'
}

Weave_Tox21 = {
    'random_seed': 2,
    'batch_size': 32,
    'lr': 1e-3,
    'num_epochs': 100,
    'node_data_field': 'h',
    'edge_data_field': 'e',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'num_gnn_layers': 2,
    'gnn_hidden_feats': 50,
    'graph_feats': 128,
    'patience': 10,
    'smiles_to_graph': partial(smiles_to_complete_graph, add_self_loop=True),
    'node_featurizer': WeaveAtomFeaturizer(),
    'edge_featurizer': WeaveEdgeFeaturizer(max_distance=2),
    'metric_name': 'roc_auc_score'
}

MPNN_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'num_epochs': 250,
    'node_in_feats': 15,
    'node_out_feats': 64,
    'edge_in_feats': 5,
    'edge_hidden_feats': 128,
    'n_tasks': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'mae',
    'weight_decay': 0
}

SchNet_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'num_epochs': 250,
    'node_feats': 64,
    'hidden_feats': [64, 64, 64],
    'classifier_hidden_feats': 64,
    'n_tasks': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'mae',
    'weight_decay': 0
}

MGCN_Alchemy = {
    'random_seed': 0,
    'batch_size': 16,
    'num_epochs': 250,
    'feats': 128,
    'n_layers': 3,
    'classifier_hidden_feats': 64,
    'n_tasks': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'mae',
    'weight_decay': 0
}

AttentiveFP_Aromaticity = {
    'random_seed': 8,
    'graph_feat_size': 200,
    'num_layers': 2,
    'num_timesteps': 2,
    'node_feat_size': 39,
    'edge_feat_size': 10,
    'n_tasks': 1,
    'dropout': 0.2,
    'weight_decay': 10 ** (-5.0),
    'lr': 10 ** (-2.5),
    'batch_size': 128,
    'num_epochs': 800,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'patience': 80,
    'metric_name': 'rmse',
    'smiles_to_graph': smiles_to_bigraph,
    # Follow the atom featurization in the original work
    'node_featurizer': BaseAtomFeaturizer(
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
    ),
    'edge_featurizer': BaseBondFeaturizer({
        'he': lambda bond: [0 for _ in range(10)]
    })
}

experiment_configures = {
    'GCN_Tox21': GCN_Tox21,
    'GAT_Tox21': GAT_Tox21,
    'Weave_Tox21': Weave_Tox21,
    'MPNN_Alchemy': MPNN_Alchemy,
    'SchNet_Alchemy': SchNet_Alchemy,
    'MGCN_Alchemy': MGCN_Alchemy,
    'AttentiveFP_Aromaticity': AttentiveFP_Aromaticity
}
def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
