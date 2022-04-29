import os
import pytest

@pytest.mark.parametrize('data', ['cora', 'citeseer', 'pubmed', 'csv', 'reddit',
                                  'co-buy-computer', 'ogbn-arxiv', 'ogbn-products'])
@pytest.mark.parametrize('model', ['gcn', 'gat', 'sage', 'sgc', 'gin'])
def test_nodepred(data, model):
    os.system('dgl configure nodepred --data {} --model {}'.format(data, model))
    assert os.path.exists('nodepred_{}_{}.yaml'.format(data, model))

    custom_config_file = 'custom_{}_{}.yaml'.format(data, model)
    os.system('dgl configure nodepred --data {} --model {} --cfg {}'.format(data, model,
                                                                            custom_config_file))
    assert os.path.exists(custom_config_file)
