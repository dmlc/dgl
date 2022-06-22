
CONFIG={
    'ppi_n':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0, 'edge_budget': 4000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 50, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 6000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'node', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 0,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'ppi_e':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0.1, 'edge_budget': 4000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 50, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 6000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'edge', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 0,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'ppi_rw':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0.1, 'edge_budget': 4000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 50, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 6000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'rw', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 0,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'flickr_n':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'edge_budget': 6000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 50, 'n_hidden': 256, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 25, 'num_roots': 6000, 'sampler': 'node', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 0,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': False
    },

    'flickr_e':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'edge_budget': 6000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 50, 'n_hidden': 256, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 25, 'num_roots': 6000, 'sampler': 'edge', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 0,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': False
    },

    'flickr_rw':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'edge_budget': 6000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 50, 'n_hidden': 256, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 25, 'num_roots': 6000, 'sampler': 'rw', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 0,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': False
    },

    'reddit_n':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'edge_budget': 4000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 20, 'n_hidden': 128, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'node', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 8,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'reddit_e':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'edge_budget': 6000, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 20, 'n_hidden': 128, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 50, 'num_roots': 3000, 'sampler': 'edge', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 8,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'reddit_rw':
    {
        'aggr': 'concat', 'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'edge_budget': 6000, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 10, 'n_hidden': 128, 'no_batch_norm': False, 'node_budget': 8000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'rw', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 8,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'yelp_n':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'edge_budget': 6000, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 10, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'node', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 8,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'yelp_e':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'edge_budget': 2500, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 10, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'edge', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 8,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'yelp_rw':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'edge_budget': 2500, 'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 10, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 1250, 'sampler': 'rw', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 8,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'amazon_n':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'edge_budget': 2500, 'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 5, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 4500,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'node', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 4,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    },

    'amazon_e':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'edge_budget': 2000, 'gpu': 0,'length': 4,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 10, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 200, 'sampler': 'edge', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 20,
        'num_subg_sampler': 5000, 'batch_size_sampler': 50, 'num_workers': 26, 'full': True
    },

    'amazon_rw':
    {
        'aggr': 'concat', 'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'edge_budget': 2500, 'gpu': 0,'length': 2,
        'log_dir': 'none', 'lr': 0.01, 'n_epochs': 5, 'n_hidden': 512, 'no_batch_norm': False, 'node_budget': 5000,
        'num_subg': 50, 'num_roots': 1500, 'sampler': 'rw', 'use_val': True, 'val_every': 1, 'num_workers_sampler': 4,
        'num_subg_sampler': 10000, 'batch_size_sampler': 200, 'num_workers': 8, 'full': True
    }
}
