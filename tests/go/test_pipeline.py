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

    custom_script = '_'.join([data, model]) + '.py'
    os.system('dgl export --cfg {} --output {}'.format(custom_config_file, custom_script))
    assert os.path.exists(custom_script)

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

    custom_script = '_'.join([data, model]) + '.py'
    os.system('dgl export --cfg {} --output {}'.format(custom_config_file, custom_script))
    assert os.path.exists(custom_script)

@pytest.mark.parametrize('data', ['cora', 'citeseer', 'pubmed', 'csv', 'reddit',
                                  'co-buy-computer', 'ogbn-arxiv', 'ogbn-products', 'ogbl-collab',
                                  'ogbl-citation2'])
@pytest.mark.parametrize('node_model', ['gcn' ,'gat', 'sage', 'sgc', 'gin'])
@pytest.mark.parametrize('edge_model', ['ele', 'bilinear'])
@pytest.mark.parametrize('neg_sampler', ['global', 'persource'])
def test_linkpred(data, node_model, edge_model, neg_sampler):
    custom_config_file = '_'.join([data, node_model, edge_model, neg_sampler]) + '.yaml'
    os.system('dgl configure linkpred --data {} --node-model {} --edge-model {} --neg-sampler {} --cfg {}'.format(
        data, node_model, edge_model, neg_sampler, custom_config_file))
    assert os.path.exists(custom_config_file)

    custom_script = '_'.join([data, node_model, edge_model, neg_sampler]) + '.py'
    os.system('dgl export --cfg {} --output {}'.format(custom_config_file, custom_script))
    assert os.path.exists(custom_script)

@pytest.mark.parametrize('data', ['cora', 'citeseer', 'pubmed', 'csv', 'reddit',
                                  'co-buy-computer', 'ogbn-arxiv', 'ogbn-products', 'ogbl-collab',
                                  'ogbl-citation2'])
@pytest.mark.parametrize('node_model', ['gcn' ,'gat', 'sage', 'sgc', 'gin'])
@pytest.mark.parametrize('edge_model', ['ele', 'bilinear'])
def test_linkpred_default_neg_sampler(data, node_model, edge_model):
    custom_config_file = '_'.join([data, node_model, edge_model]) + '.yaml'
    os.system('dgl configure linkpred --data {} --node-model {} --edge-model {} --cfg {}'.format(
        data, node_model, edge_model, custom_config_file))
    assert os.path.exists(custom_config_file)

@pytest.mark.parametrize('data', ['csv', 'ogbg-molhiv', 'ogbg-molpcba'])
@pytest.mark.parametrize('model', ['gin', 'pna'])
def test_graphpred(data, model):
    os.system('dgl configure graphpred --data {} --model {}'.format(data, model))
    assert os.path.exists('graphpred_{}_{}.yaml'.format(data, model))

    custom_config_file = 'custom_{}_{}.yaml'.format(data, model)
    os.system('dgl configure graphpred --data {} --model {} --cfg {}'.format(data, model,
                                                                             custom_config_file))
    assert os.path.exists(custom_config_file)

    custom_script = '_'.join([data, model]) + '.py'
    os.system('dgl export --cfg {} --output {}'.format(custom_config_file, custom_script))
    assert os.path.exists(custom_script)

@pytest.mark.parametrize('recipe',
                         ['graphpred_hiv_gin.yaml',
                          'graphpred_hiv_pna.yaml',
                          'graphpred_pcba_gin.yaml',
                          'linkpred_cora_sage.yaml',
                          'linkpred_citation2_sage.yaml',
                          'linkpred_collab_sage.yaml',
                          'nodepred_citeseer_gat.yaml',
                          'nodepred_citeseer_gcn.yaml',
                          'nodepred_citeseer_sage.yaml',
                          'nodepred_cora_gat.yaml',
                          'nodepred_cora_gcn.yaml',
                          'nodepred_cora_sage.yaml',
                          'nodepred_pubmed_gat.yaml',
                          'nodepred_pubmed_gcn.yaml',
                          'nodepred_pubmed_sage.yaml',
                          'nodepred-ns_arxiv_gcn.yaml',
                          'nodepred-ns_product_sage.yaml'])
def test_recipe(recipe):
    # Remove all generated yaml files
    current_dir = os.listdir("./")
    for item in current_dir:
        if item.endswith(".yaml"):
            os.remove(item)

    os.system('dgl recipe get {}'.format(recipe))
    assert os.path.exists(recipe)

def test_node_cora():
    os.system('dgl configure nodepred --data cora --model gcn')
    os.system('dgl train --cfg nodepred_cora_gcn.yaml')
    assert os.path.exists('checkpoint.pth')
    assert os.path.exists('model.pth')
