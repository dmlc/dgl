from dglls.model.model_zoo import *

def test_graph1():
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1)

def test_graph2():
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1)

def test_mlp_predictor():
    g_feats = torch.tensor([[1.], [2.]])
    mlp_predictor = MLPPredictor(in_feats=1, hidden_feats=1, n_tasks=2)
    assert mlp_predictor(g_feats).shape == torch.Size([2, 2])

def test_gcn_predictor():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()
    gcn_predictor = GCNPredictor(in_feats=1,
                                 hidden_feats=[1],
                                 activation=[F.relu],
                                 residual=[True],
                                 batchnorm=[True],
                                 dropout=[0.1],
                                 classifier_hidden_feats=1,
                                 classifier_dropout=0.1,
                                 n_tasks=2)
    gcn_predictor.eval()
    assert gcn_predictor(g, node_feats).shape == torch.Size([1, 2])
    gcn_predictor.train()
    assert gcn_predictor(bg, batch_node_feats).shape == torch.Size([2, 2])

def test_gat_predictor():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()
    gat_predictor = GATPredictor(in_feats=1,
                                 hidden_feats=[1, 2],
                                 num_heads=[2, 3],
                                 feat_drops=[0.1, 0.1],
                                 attn_drops=[0.1, 0.1],
                                 alphas=[0.1, 0.1],
                                 residuals=[True, True],
                                 agg_modes=['mean', 'flatten'],
                                 activations=[None, F.elu])
    gat_predictor.eval()
    assert gat_predictor(g, node_feats).shape == torch.Size([1, 1])
    gat_predictor.train()
    assert gat_predictor(bg, batch_node_feats).shape == torch.Size([2, 1])

if __name__ == '__main__':
    test_mlp_predictor()
    test_gcn_predictor()
    test_gat_predictor()
