from dglls.utils.featurizers import CanonicalAtomFeaturizer

GCN_Tox21 = {
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gcn_hidden_feats': [64, 64],
    'classifier_hidden_feats': 64,
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

GAT_Tox21 = {
    'random_seed': 0,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
}

experiment_configures = {
    'GCN_Tox21': GCN_Tox21,
    'GAT_Tox21': GAT_Tox21,
}

def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
