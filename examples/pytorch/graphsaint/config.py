# arch: model architecture configuration, e.g., '1-1-0', means there're three layers, the first
# and the second layer employ message passing on the graph and aggregate the embeddings of each
# node and its neighbors. The last layer only updates each node's embedding.

# budget: configuration for SAINTSampler

CONFIG={
    'ppi_n':
    {
        'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0, 'budget': 6000, 'lr': 0.01,
        'n_epochs': 50, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'node'
    },

    'ppi_e':
    {
        'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0.1, 'budget': 4000, 'lr': 0.01,
        'n_epochs': 50, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'edge'
    },

    'ppi_rw':
    {
        'arch': '1-0-1-0', 'dataset': 'ppi', 'dropout': 0.1, 'budget': (3000, 2), 'lr': 0.01,
        'n_epochs': 50, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'walk'
    },

    'flickr_n':
    {
        'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'budget': 8000, 'lr': 0.01,
        'n_epochs': 50, 'n_hidden': 256, 'norm_ratio': 25, 'sampler': 'node'
    },

    'flickr_e':
    {
        'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'budget': 6000, 'lr': 0.01,
        'n_epochs': 50, 'n_hidden': 256, 'norm_ratio': 25, 'sampler': 'edge'
    },

    'flickr_rw':
    {
        'arch': '1-1-0', 'dataset': 'flickr', 'dropout': 0.2, 'budget': (6000, 2), 'lr': 0.01,
        'n_epochs': 50, 'n_hidden': 256, 'norm_ratio': 25, 'sampler': 'walk'
    },

    'reddit_n':
    {
        'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'budget': 8000, 'lr': 0.01,
        'n_epochs': 20, 'n_hidden': 128, 'norm_ratio': 50, 'sampler': 'node'
    },

    'reddit_e':
    {
        'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'budget': 6000, 'lr': 0.01,
        'n_epochs': 20, 'n_hidden': 128, 'norm_ratio': 50, 'sampler': 'edge'
    },

    'reddit_rw':
    {
        'arch': '1-0-1-0', 'dataset': 'reddit', 'dropout': 0.1, 'budget': (200, 4), 'lr': 0.01,
        'n_epochs': 10, 'n_hidden': 128, 'norm_ratio': 50, 'sampler': 'walk'
    },

    'yelp_n':
    {
        'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'budget': 5000, 'lr': 0.01,
        'n_epochs': 10, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'node'
    },

    'yelp_e':
    {
        'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'budget': 2500, 'lr': 0.01,
        'n_epochs': 10, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'edge'
    },

    'yelp_rw':
    {
        'arch': '1-1-0', 'dataset': 'yelp', 'dropout': 0.1, 'budget': (1250, 2), 'lr': 0.01,
        'n_epochs': 10, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'walk'
    },

    'amazon_n':
    {
        'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'budget': 4500, 'lr': 0.01,
        'n_epochs': 5, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'node'
    },

    'amazon_e':
    {
        'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'budget': 2000, 'lr': 0.01,
        'n_epochs': 10, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'edge'
    },

    'amazon_rw':
    {
        'arch': '1-1-0', 'dataset': 'amazon', 'dropout': 0.1, 'budget': (1500, 2), 'lr': 0.01,
        'n_epochs': 5, 'n_hidden': 512, 'norm_ratio': 50, 'sampler': 'walk'
    }
}
