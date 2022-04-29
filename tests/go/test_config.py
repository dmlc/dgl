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

@pytest.mark.parametrize('data', ['cora', 'citeseer', 'pubmed', 'csv', 'reddit',
                                  'co-buy-computer', 'ogbn-arxiv', 'ogbn-products'])
@pytest.mark.parametrize('model', ['gcn', 'gat', 'sage'])
def test_nodepred_ns(data, model):
    os.system('dgl configure nodepred-ns --data {} --model {}'.format(data, model))
    assert os.path.exists('nodepred-ns_{}_{}.yaml'.format(data, model))

    custom_config_file = 'custom_{}_{}.yaml'.format(data, model)
    os.system('dgl configure nodepred-ns --data {} --model {} --cfg {}'.format(data, model,
                                                                               custom_config_file))
    assert os.path.exists(custom_config_file)

@pytest.mark.parametrize('data', ['cora', 'citeseer', 'pubmed', 'csv', 'reddit',
                                  'co-buy-computer', 'ogbn-arxiv', 'ogbn-products', 'ogbl-collab',
                                  'ogbl-citation2'])
@pytest.mark.parametrize('node_model', ['gcn' ,'gat', 'sage', 'sgc', 'gin'])
@pytest.mark.parametrize('edge_model', ['ele', 'bilinear'])
@pytest.mark.parametrize('neg_sampler', ['global', 'persource'])
def test_linkpred(data, node_model, edge_model, neg_sampler):
    pass
