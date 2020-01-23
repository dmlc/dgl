import dgl
import torch
import torch.nn.functional as F

from dgl import DGLGraph
from dglls.model.gnn import GCN, GAT, AttentiveFPGNN

def test_graph1():
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1)

def test_graph2():
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1)

def test_graph3():
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * g.number_of_edges()).float().reshape(-1, 2)

def test_graph4():
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1), \
           torch.arange(2 * bg.number_of_edges()).float().reshape(-1, 2)

def test_gcn():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()

    # Test GCN
    gnn = GCN(in_feats=1,
              hidden_feats=[1, 1],
              activation=[F.relu, F.relu],
              residual=[True, True],
              batchnorm=[True, True],
              dropout=[0.2, 0.2])
    assert gnn(g, node_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 1])

def test_gat():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()

    # Test GAT
    gnn = GAT(in_feats=1,
              hidden_feats=[1, 1],
              num_heads=[2, 3],
              feat_drops=[0.1, 0.1],
              attn_drops=[0.1, 0.1],
              alphas=[0.2, 0.2],
              residuals=[True, True],
              agg_modes=['flatten', 'mean'],
              activations=[None, F.elu])
    assert gnn(g, node_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 1])

    gnn = GAT(in_feats=1,
              hidden_feats=[1, 1],
              num_heads=[2, 3],
              feat_drops=[0.1, 0.1],
              attn_drops=[0.1, 0.1],
              alphas=[0.2, 0.2],
              residuals=[True, True],
              agg_modes=['mean', 'flatten'],
              activations=[None, F.elu])
    assert gnn(g, node_feats).shape == torch.Size([3, 3])
    assert gnn(bg, batch_node_feats).shape == torch.Size([8, 3])

def test_attentive_fp_gnn():
    g, node_feats, edge_feats = test_graph3()
    bg, batch_node_feats, batch_edge_feats = test_graph4()

    # Test AttentiveFPGNN

    gnn = AttentiveFPGNN(node_feat_size=1,
                         edge_feat_size=2,
                         num_layers=1,
                         graph_feat_size=1,
                         dropout=0.)
    assert gnn(g, node_feats, edge_feats).shape == torch.Size([3, 1])
    assert gnn(bg, batch_node_feats, batch_edge_feats).shape == torch.Size([8, 1])

if __name__ == '__main__':
    test_gcn()
    test_gat()
    test_attentive_fp_gnn()
