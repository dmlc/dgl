from dgl.data.chem import CanonicalAtomFeaturizer

GCN_Tox21 = {
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'train_val_test_split': [0.8, 0.1, 0.1],
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'classifier_hidden_feats': 64,
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

GAT_Tox21 = {
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'train_val_test_split': [0.8, 0.1, 0.1],
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

MPNN_Alchemy = {
    'batch_size': 16,
    'num_epochs': 250,
    'node_in_feats': 15,
    'edge_in_feats': 5,
    'output_dim': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'l1'
}

SCHNET_Alchemy = {
    'batch_size': 16,
    'num_epochs': 250,
    'norm': True,
    'output_dim': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'l1'
}

MGCN_Alchemy = {
    'batch_size': 16,
    'num_epochs': 250,
    'norm': True,
    'output_dim': 12,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'l1'
}

experiment_configures = {
    'GCN_Tox21': GCN_Tox21,
    'GAT_Tox21': GAT_Tox21,
    'MPNN_Alchemy': MPNN_Alchemy,
    'SCHNET_Alchemy': SCHNET_Alchemy,
    'MGCN_Alchemy': MGCN_Alchemy
}

def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
