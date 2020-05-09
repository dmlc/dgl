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
    'num_epochs': 18,
    'print_every': 50,
    'decay_every': 10000,      # Learning rate decay
    'lr_decay_factor': 0.9,
    'top_ks_val': [12, 16, 20, 40, 80],
    'top_ks_test': [6, 8, 10],
    'max_k': 80
}

# Configuration for candidate ranking
candidate_ranking_config = {
    'batch_size': 4,
    'hidden_size': 500,
    'num_encode_gnn_layers': 3,
    'max_norm': 50.0,
    'node_in_feats': 89,
    'edge_in_feats': 5,
    'lr': 0.001,
    'num_epochs': 6,
    'print_every': 20,
    'decay_every': 100000,
    'lr_decay_factor': 0.9,
    'top_ks': [1, 2, 3, 5],
    'max_k': 10,
    'max_num_change_combos_per_reaction_train': 150,
    'max_num_change_combos_per_reaction_eval': 1500,
    'num_candidate_bond_changes': 16
}
candidate_ranking_config['max_norm'] = candidate_ranking_config['max_norm'] * \
                                       candidate_ranking_config['batch_size']
