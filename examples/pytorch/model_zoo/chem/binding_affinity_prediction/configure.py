import numpy as np
import torch

ACNN_PDBBind_core_pocket_random = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 120,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'random'
}

ACNN_PDBBind_core_pocket_scaffold = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 170,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'scaffold'
}

ACNN_PDBBind_core_pocket_stratified = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 110,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'stratified'
}

ACNN_PDBBind_core_pocket_temporal = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 80,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'temporal'
}

ACNN_PDBBind_refined_pocket_random = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 200,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'random'
}

ACNN_PDBBind_refined_pocket_scaffold = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 350,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'scaffold'
}

ACNN_PDBBind_refined_pocket_stratified = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 400,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'stratified'
}

ACNN_PDBBind_refined_pocket_temporal = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'num_epochs': 350,
    'metrics': ['pearson_r2', 'mae'],
    'split': 'temporal'
}

experiment_configures = {
    'ACNN_PDBBind_core_pocket_random': ACNN_PDBBind_core_pocket_random,
    'ACNN_PDBBind_core_pocket_scaffold': ACNN_PDBBind_core_pocket_scaffold,
    'ACNN_PDBBind_core_pocket_stratified': ACNN_PDBBind_core_pocket_stratified,
    'ACNN_PDBBind_core_pocket_temporal': ACNN_PDBBind_core_pocket_temporal,
    'ACNN_PDBBind_refined_pocket_random': ACNN_PDBBind_refined_pocket_random,
    'ACNN_PDBBind_refined_pocket_scaffold': ACNN_PDBBind_refined_pocket_scaffold,
    'ACNN_PDBBind_refined_pocket_stratified': ACNN_PDBBind_refined_pocket_stratified,
    'ACNN_PDBBind_refined_pocket_temporal': ACNN_PDBBind_refined_pocket_temporal
}

def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
