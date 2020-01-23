from dgl import DGLGraph
from dglls.model.readout import *

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

def test_weighted_sum_and_max():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()
    model = WeightedSumAndMax(in_feats=1)
    assert model(g, node_feats).shape == torch.Size([1, 2])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 2])

def test_attentive_fp_readout():
    g, node_feats = test_graph1()
    bg, batch_node_feats = test_graph2()
    model = AttentiveFPReadout(num_timesteps=1,
                               feat_size=1)
    assert model(g, node_feats).shape == torch.Size([1, 1])
    assert model(bg, batch_node_feats).shape == torch.Size([2, 1])

if __name__ == '__main__':
    test_weighted_sum_and_max()
    test_attentive_fp_readout()
