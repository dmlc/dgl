import torch

ACNN_PDBBind_core_protein = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': False,
    'random_seed': 0,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'batch_size': 16,
    'hidden_sizes': [32, 32, 16],
    'atomic_numbers_considered': torch.tensor([
        6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                       7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
                      [0.0, 4.0, 8.0], [0.4]],
    'lr': 0.001,
    'patience': 10,
    'num_epochs': 10,
    'metric': 'r2'
}

ACNN_PDBBind_core_pocket = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 0,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'batch_size': 16,
    'hidden_sizes': [32, 32, 16],
    'atomic_numbers_considered': torch.tensor([
        6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                       7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
                      [0.0, 4.0, 8.0], [0.4]],
    'lr': 0.001,
    'patience': 10,
    'num_epochs': 10,
    'metric': 'r2'
}

ACNN_PDBBind_refined_protein = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': False,
    'random_seed': 0,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'batch_size': 16,
    'hidden_sizes': [32, 32, 16],
    'atomic_numbers_considered': torch.tensor([
        6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                       7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
                      [0.0, 4.0, 8.0], [0.4]],
    'lr': 0.001,
    'patience': 10,
    'num_epochs': 10,
    'metric': 'r2'
}

ACNN_PDBBind_refined_pocket = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 0,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'batch_size': 16,
    'hidden_sizes': [32, 32, 16],
    'atomic_numbers_considered': torch.tensor([
        6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                       7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
                      [0.0, 4.0, 8.0], [0.4]],
    'lr': 0.001,
    'patience': 10,
    'num_epochs': 10,
    'metric': 'r2'
}

experiment_configures = {
    'ACNN_PDBBind_core_protein': ACNN_PDBBind_core_protein,
    'ACNN_PDBBind_core_pocket': ACNN_PDBBind_core_pocket,
    'ACNN_PDBBind_refined_protein': ACNN_PDBBind_refined_protein,
    'ACNN_PDBBind_refined_pocket': ACNN_PDBBind_refined_pocket
}

def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
