import dgl
import torch
import torch.nn.functional as F

from dgl import DGLGraph
from dglls.model.gnn import *

def test_graph1():
    g = DGLGraph([(0, 1), (0, 2), (1, 2)])
    return g, torch.arange(g.number_of_nodes()).float().reshape(-1, 1)

def test_graph2():
    g1 = DGLGraph([(0, 1), (0, 2), (1, 2)])
    g2 = DGLGraph([(0, 1), (1, 2), (1, 3), (1, 4)])
    bg = dgl.batch([g1, g2])
    return bg, torch.arange(bg.number_of_nodes()).float().reshape(-1, 1)

def test_gcn():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()

    # Test GCNLayer
    gcn_layer = GCNLayer(in_feats=1,
                         out_feats=1,
                         activation=F.relu,
                         residual=True,
                         batchnorm=True,
                         dropout=0.2)
    assert gcn_layer(g, node_feats).shape == torch.Size([3, 1])
    assert gcn_layer(bg, batch_node_feats).shape == torch.Size([8, 1])

    # Test GCN
    gcn = GCN(in_feats=1,
              hidden_feats=[1, 1],
              activation=[F.relu, F.relu],
              residual=[True, True],
              batchnorm=[True, True],
              dropout=[0.2, 0.2])
    assert gcn(g, node_feats).shape == torch.Size([3, 1])
    assert gcn(bg, batch_node_feats).shape == torch.Size([8, 1])

def test_gat():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()

    # Test GATLayer
    gat_layer = GATLayer(in_feats=1,
                         out_feats=1,
                         num_heads=2,
                         feat_drop=0.1,
                         attn_drop=0.1,
                         alpha=0.2,
                         residual=True,
                         agg_mode='flatten',
                         activation=None)
    assert gat_layer(g, node_feats).shape == torch.Size([3, 2])
    assert gat_layer(bg, batch_node_feats).shape == torch.Size([8, 2])

    # Test GATLayer
    gat_layer = GATLayer(in_feats=1,
                         out_feats=1,
                         num_heads=2,
                         feat_drop=0.1,
                         attn_drop=0.1,
                         alpha=0.2,
                         residual=True,
                         agg_mode='mean',
                         activation=F.relu)
    assert gat_layer(g, node_feats).shape == torch.Size([3, 1])
    assert gat_layer(bg, batch_node_feats).shape == torch.Size([8, 1])

    # Test GAT
    gat = GAT(in_feats=1,
              hidden_feats=[1, 1],
              num_heads=[2, 3],
              feat_drops=[0.1, 0.1],
              attn_drops=[0.1, 0.1],
              alphas=[0.2, 0.2],
              residuals=[True, True],
              agg_modes=['flatten', 'mean'],
              activations=[None, F.elu])
    assert gat(g, node_feats).shape == torch.Size([3, 1])
    assert gat(bg, batch_node_feats).shape == torch.Size([8, 1])

    gat = GAT(in_feats=1,
              hidden_feats=[1, 1],
              num_heads=[2, 3],
              feat_drops=[0.1, 0.1],
              attn_drops=[0.1, 0.1],
              alphas=[0.2, 0.2],
              residuals=[True, True],
              agg_modes=['mean', 'flatten'],
              activations=[None, F.elu])
    assert gat(g, node_feats).shape == torch.Size([3, 3])
    assert gat(bg, batch_node_feats).shape == torch.Size([8, 3])

if __name__ == '__main__':
    test_gcn()
    test_gat()
