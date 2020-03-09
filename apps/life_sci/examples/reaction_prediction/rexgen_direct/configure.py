# Configuration for reaction center identification
reaction_center_config = {
    'batch_size': 20,
    'hidden_size': 300,
    'max_norm': 5.0,
    'node_in_feats': 82,
    'edge_in_feats': 6,
    'node_pair_in_feats': 10,
    'node_out_feats': 300,
    'n_layers': 3,
    'n_tasks': 5,
    'lr': 0.001,
    'num_epochs': 50,
    'print_every': 50,
    'decay_every': 2000,      # Learning rate decay
    'lr_decay_factor': 0.9,
    'top_ks': [10, 20]
}
