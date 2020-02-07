import dgl
import torch

from dgl import DGLGraph

from dgllife.model.model_zoo import *

def test_graph1():
    """Graph with node features."""
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1)

def test_graph2():
    """Batched graph with node features."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1)

def test_graph3():
    """Graph with node features and edge features."""
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * g.number_of_edges()).float().reshape(-1, 2)

def test_graph4():
    """Batched graph with node features and edge features."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * bg.number_of_edges()).float().reshape(-1, 2)

def test_graph5():
    """Graph with node types and edge distances."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g1, torch.LongTensor([0, 1, 0]), torch.randn(3, 1)

def test_graph6():
    """Batched graph with node types and edge distances."""
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.LongTensor([0, 1, 0, 2, 0, 3, 4, 4]), torch.randn(7, 1)

def test_mlp_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g_feats = torch.tensor([[1.], [2.]]).to(device)
    mlp_predictor = MLPPredictor(in_feats=1, hidden_feats=1, n_tasks=2).to(device)
    assert mlp_predictor(g_feats).shape == torch.Size([2, 2])

def test_gcn_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gcn_predictor = GCNPredictor(in_feats=1)
    gcn_predictor.eval()
    assert gcn_predictor(g, node_feats).shape == torch.Size([1, 1])
    gcn_predictor.train()
    assert gcn_predictor(bg, batch_node_feats).shape == torch.Size([2, 1])

    # Test configured setting
    gcn_predictor = GCNPredictor(in_feats=1,
                                 hidden_feats=[1],
                                 activation=[F.relu],
                                 residual=[True],
                                 batchnorm=[True],
                                 dropout=[0.1],
                                 classifier_hidden_feats=1,
                                 classifier_dropout=0.1,
                                 n_tasks=2).to(device)
    gcn_predictor.eval()
    assert gcn_predictor(g, node_feats).shape == torch.Size([1, 2])
    gcn_predictor.train()
    assert gcn_predictor(bg, batch_node_feats).shape == torch.Size([2, 2])

def test_gat_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats = test_graph1()
    g, node_feats = g.to(device), node_feats.to(device)
    bg, batch_node_feats = test_graph2()
    bg, batch_node_feats = bg.to(device), batch_node_feats.to(device)

    # Test default setting
    gat_predictor = GATPredictor(in_feats=1).to(device)
    gat_predictor.eval()
    assert gat_predictor(g, node_feats).shape == torch.Size([1, 1])
    gat_predictor.train()
    assert gat_predictor(bg, batch_node_feats).shape == torch.Size([2, 1])

    # Test configured setting
    gat_predictor = GATPredictor(in_feats=1,
                                 hidden_feats=[1, 2],
                                 num_heads=[2, 3],
                                 feat_drops=[0.1, 0.1],
                                 attn_drops=[0.1, 0.1],
                                 alphas=[0.1, 0.1],
                                 residuals=[True, True],
                                 agg_modes=['mean', 'flatten'],
                                 activations=[None, F.elu]).to(device)
    gat_predictor.eval()
    assert gat_predictor(g, node_feats).shape == torch.Size([1, 1])
    gat_predictor.train()
    assert gat_predictor(bg, batch_node_feats).shape == torch.Size([2, 1])

def test_attentivefp_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)
    attentivefp_predictor = AttentiveFPPredictor(node_feat_size=1,
                                                 edge_feat_size=2,
                                                 num_layers=2,
                                                 num_timesteps=1,
                                                 graph_feat_size=1,
                                                 n_tasks=2).to(device)
    assert attentivefp_predictor(g, node_feats, edge_feats).shape == torch.Size([1, 2])
    assert attentivefp_predictor(bg, batch_node_feats, batch_edge_feats).shape == \
           torch.Size([2, 2])

def test_schnet_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_types, edge_dists = test_graph5()
    g, node_types, edge_dists = g.to(device), node_types.to(device), edge_dists.to(device)
    bg, batch_node_types, batch_edge_dists = test_graph6()
    bg, batch_node_types, batch_edge_dists = bg.to(device), batch_node_types.to(device), \
                                             batch_edge_dists.to(device)

    # Test default setting
    schnet_predictor = SchNetPredictor().to(device)
    assert schnet_predictor(g, node_types, edge_dists).shape == torch.Size([1, 1])
    assert schnet_predictor(bg, batch_node_types, batch_edge_dists).shape == \
           torch.Size([2, 1])

    # Test configured setting
    schnet_predictor = SchNetPredictor(node_feats=2,
                                       hidden_feats=[2, 2],
                                       classifier_hidden_feats=3,
                                       n_tasks=3,
                                       num_node_types=5,
                                       cutoff=0.3).to(device)
    assert schnet_predictor(g, node_types, edge_dists).shape == torch.Size([1, 3])
    assert schnet_predictor(bg, batch_node_types, batch_edge_dists).shape == \
           torch.Size([2, 3])

def test_mgcn_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_types, edge_dists = test_graph5()
    g, node_types, edge_dists = g.to(device), node_types.to(device), edge_dists.to(device)
    bg, batch_node_types, batch_edge_dists = test_graph6()
    bg, batch_node_types, batch_edge_dists = bg.to(device), batch_node_types.to(device), \
                                             batch_edge_dists.to(device)

    # Test default setting
    mgcn_predictor = MGCNPredictor().to(device)
    assert mgcn_predictor(g, node_types, edge_dists).shape == torch.Size([1, 1])
    assert mgcn_predictor(bg, batch_node_types, batch_edge_dists).shape == \
           torch.Size([2, 1])

    # Test configured setting
    mgcn_predictor = MGCNPredictor(feats=2,
                                   n_layers=2,
                                   classifier_hidden_feats=3,
                                   n_tasks=3,
                                   num_node_types=5,
                                   num_edge_types=150,
                                   cutoff=0.3).to(device)
    assert mgcn_predictor(g, node_types, edge_dists).shape == torch.Size([1, 3])
    assert mgcn_predictor(bg, batch_node_types, batch_edge_dists).shape == \
           torch.Size([2, 3])

def test_mpnn_predictor():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    g, node_feats, edge_feats = test_graph3()
    g, node_feats, edge_feats = g.to(device), node_feats.to(device), edge_feats.to(device)
    bg, batch_node_feats, batch_edge_feats = test_graph4()
    bg, batch_node_feats, batch_edge_feats = bg.to(device), batch_node_feats.to(device), \
                                             batch_edge_feats.to(device)

    # Test default setting
    mpnn_predictor = MPNNPredictor(node_in_feats=1,
                                   edge_in_feats=2)
    assert mpnn_predictor(g, node_feats, edge_feats).shape == torch.Size([1, 1])
    assert mpnn_predictor(bg, batch_node_feats, batch_edge_feats).shape == \
           torch.Size([2, 1])

    # Test configured setting
    mpnn_predictor = MPNNPredictor(node_in_feats=1,
                                   edge_in_feats=2,
                                   node_out_feats=2,
                                   edge_hidden_feats=2,
                                   n_tasks=2,
                                   num_step_message_passing=2,
                                   num_step_set2set=2,
                                   num_layer_set2set=2)
    assert mpnn_predictor(g, node_feats, edge_feats).shape == torch.Size([1, 2])
    assert mpnn_predictor(bg, batch_node_feats, batch_edge_feats).shape == \
           torch.Size([2, 2])

if __name__ == '__main__':
    test_mlp_predictor()
    test_gcn_predictor()
    test_gat_predictor()
    test_attentivefp_predictor()
    test_schnet_predictor()
    test_mgcn_predictor()
    test_mpnn_predictor()
