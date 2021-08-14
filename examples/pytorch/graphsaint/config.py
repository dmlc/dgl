
CONFIG={
    'ppi_n':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0, 'edge_budget': 4000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 1000, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 6000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'node', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 0,
        'online': 'tot', 'num_subg_train': 0, 'num_subg_norm': 144, 'batch_size_norm': 144, 'num_workers': 0
    },

    'ppi_e':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0.1, 'edge_budget': 4000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 1000, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 6000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'edge', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 0,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'ppi_rw':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0.1, 'edge_budget': 4000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 1000, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 6000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'rw', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 0,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'flickr_n':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'edge_budget': 6000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 40, 'n_hidden': 256, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 25, 'num_roots': 6000, 'sampler': 'node', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'flickr_e':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'edge_budget': 6000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 15, 'n_hidden': 256, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 25, 'num_roots': 6000, 'sampler': 'edge', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'flickr_rw':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'edge_budget': 6000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 40, 'n_hidden': 256, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 25, 'num_roots': 6000, 'sampler': 'rw', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'reddit_n':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'edge_budget': 4000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 40, 'n_hidden': 128, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'node', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'reddit_e':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'edge_budget': 6000, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 40, 'n_hidden': 128, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'edge', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'reddit_rw':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'edge_budget': 6000, 'gpu': 0, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 30, 'n_hidden': 128, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'rw', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'yelp_n':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'edge_budget': 6000, 'gpu': 0, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 50, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'node', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'yelp_e':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'edge_budget': 2500, 'gpu': 0, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 100, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'edge', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'yelp_rw':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'edge_budget': 2500, 'gpu': 0, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 75, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 1250, 'sampler': 'rw', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'amazon_n':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'edge_budget': 2500, 'gpu': 0, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 30, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 4500,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'node', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'amazon_e':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'edge_budget': 200, 'gpu': 0,'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 30, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'edge', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    },

    'amazon_rw':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'edge_budget': 2500, 'gpu': 0,'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 30, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 1500, 'sampler': 'rw', 'use_val': False, 'val_every': 1, 'num_workers_sampler': 4,
        'online': True, 'num_subg_train': 0, 'num_subg_norm': 10000, 'batch_size_norm': 200, 'num_workers': 6
    }
}