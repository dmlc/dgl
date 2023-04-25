import dgl
import pytest
import torch
from utils.graph_cases import get_cases
from dglgo.model import *


@pytest.mark.parametrize("g", get_cases(["has_scalar_e_feature"]))
def test_gcn(g):
    data_info = {"num_nodes": g.num_nodes(), "out_size": 7}
    node_feat = None
    edge_feat = g.edata["scalar_w"]

    # node embedding + not use_edge_weight
    model = GCN(data_info, embed_size=10, use_edge_weight=False)
    model(g, node_feat)

    # node embedding + use_edge_weight
    model = GCN(data_info, embed_size=10, use_edge_weight=True)
    model(g, node_feat, edge_feat)

    data_info["in_size"] = g.ndata["h"].shape[-1]
    node_feat = g.ndata["h"]

    # node feat + not use_edge_weight
    model = GCN(data_info, embed_size=-1, use_edge_weight=False)
    model(g, node_feat)

    # node feat + use_edge_weight
    model = GCN(data_info, embed_size=-1, use_edge_weight=True)
    model(g, node_feat, edge_feat)


@pytest.mark.parametrize("g", get_cases(["block-bipartite"]))
def test_gcn_block(g):
    data_info = {"in_size": 10, "out_size": 7}

    blocks = [g]
    node_feat = torch.randn(g.num_src_nodes(), data_info["in_size"])
    edge_feat = torch.abs(torch.randn(g.num_edges()))
    # not use_edge_weight
    model = GCN(data_info, use_edge_weight=False)
    model.forward_block(blocks, node_feat)

    # use_edge_weight
    model = GCN(data_info, use_edge_weight=True)
    model.forward_block(blocks, node_feat, edge_feat)


@pytest.mark.parametrize("g", get_cases(["has_scalar_e_feature"]))
def test_gat(g):
    data_info = {"num_nodes": g.num_nodes(), "out_size": 7}
    node_feat = None

    # node embedding
    model = GAT(data_info, embed_size=10)
    model(g, node_feat)

    # node feat
    data_info["in_size"] = g.ndata["h"].shape[-1]
    node_feat = g.ndata["h"]
    model = GAT(data_info, embed_size=-1)
    model(g, node_feat)


@pytest.mark.parametrize("g", get_cases(["block-bipartite"]))
def test_gat_block(g):
    data_info = {"in_size": 10, "out_size": 7}

    blocks = [g]
    node_feat = torch.randn(g.num_src_nodes(), data_info["in_size"])
    model = GAT(data_info, num_layers=1, heads=[8])
    model.forward_block(blocks, node_feat)


@pytest.mark.parametrize("g", get_cases(["has_scalar_e_feature"]))
def test_gin(g):
    data_info = {"num_nodes": g.num_nodes(), "out_size": 7}
    node_feat = None

    # node embedding
    model = GIN(data_info, embed_size=10)
    model(g, node_feat)

    # node feat
    data_info["in_size"] = g.ndata["h"].shape[-1]
    node_feat = g.ndata["h"]
    model = GIN(data_info, embed_size=-1)
    model(g, node_feat)


@pytest.mark.parametrize("g", get_cases(["has_scalar_e_feature"]))
def test_sage(g):
    data_info = {"num_nodes": g.num_nodes(), "out_size": 7}
    node_feat = None
    edge_feat = g.edata["scalar_w"]

    # node embedding
    model = GraphSAGE(data_info, embed_size=10)
    model(g, node_feat)
    model(g, node_feat, edge_feat)

    # node feat
    data_info["in_size"] = g.ndata["h"].shape[-1]
    node_feat = g.ndata["h"]
    model = GraphSAGE(data_info, embed_size=-1)
    model(g, node_feat)
    model(g, node_feat, edge_feat)


@pytest.mark.parametrize("g", get_cases(["block-bipartite"]))
def test_sage_block(g):
    data_info = {"in_size": 10, "out_size": 7}

    blocks = [g]
    node_feat = torch.randn(g.num_src_nodes(), data_info["in_size"])
    edge_feat = torch.abs(torch.randn(g.num_edges()))
    model = GraphSAGE(data_info, embed_size=-1)
    model.forward_block(blocks, node_feat)
    model.forward_block(blocks, node_feat, edge_feat)


@pytest.mark.parametrize("g", get_cases(["has_scalar_e_feature"]))
def test_sgc(g):
    data_info = {"num_nodes": g.num_nodes(), "out_size": 7}
    node_feat = None

    # node embedding
    model = SGC(data_info, embed_size=10)
    model(g, node_feat)

    # node feat
    data_info["in_size"] = g.ndata["h"].shape[-1]
    node_feat = g.ndata["h"]
    model = SGC(data_info, embed_size=-1)
    model(g, node_feat)


def test_bilinear():
    data_info = {"in_size": 10, "out_size": 1}
    model = BilinearPredictor(data_info)
    num_pairs = 10
    h_src = torch.randn(num_pairs, data_info["in_size"])
    h_dst = torch.randn(num_pairs, data_info["in_size"])
    model(h_src, h_dst)


def test_ele():
    data_info = {"in_size": 10, "out_size": 1}
    model = ElementWiseProductPredictor(data_info)
    num_pairs = 10
    h_src = torch.randn(num_pairs, data_info["in_size"])
    h_dst = torch.randn(num_pairs, data_info["in_size"])
    model(h_src, h_dst)


@pytest.mark.parametrize("virtual_node", [True, False])
def test_ogbg_gin(virtual_node):
    # Test for ogbg-mol datasets
    data_info = {"name": "ogbg-molhiv", "out_size": 1}
    model = OGBGGIN(
        data_info, embed_size=10, num_layers=2, virtual_node=virtual_node
    )
    num_nodes = 5
    num_edges = 15
    g1 = dgl.rand_graph(num_nodes, num_edges)
    g2 = dgl.rand_graph(num_nodes, num_edges)
    g = dgl.batch([g1, g2])
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    nfeat = torch.zeros(num_nodes, 9).long()
    efeat = torch.zeros(num_edges, 3).long()
    model(g, nfeat, efeat)

    # Test for non-ogbg-mol datasets
    data_info = {
        "name": "a_dataset",
        "out_size": 1,
        "node_feat_size": 15,
        "edge_feat_size": 5,
    }
    model = OGBGGIN(
        data_info, embed_size=10, num_layers=2, virtual_node=virtual_node
    )
    nfeat = torch.randn(num_nodes, data_info["node_feat_size"])
    efeat = torch.randn(num_edges, data_info["edge_feat_size"])
    model(g, nfeat, efeat)


def test_pna():
    # Test for ogbg-mol datasets
    data_info = {"name": "ogbg-molhiv", "delta": 1, "out_size": 1}
    model = PNA(data_info, embed_size=10, num_layers=2)
    num_nodes = 5
    num_edges = 15
    g = dgl.rand_graph(num_nodes, num_edges)
    nfeat = torch.zeros(num_nodes, 9).long()
    model(g, nfeat)

    # Test for non-ogbg-mol datasets
    data_info = {
        "name": "a_dataset",
        "node_feat_size": 15,
        "delta": 1,
        "out_size": 1,
    }
    model = PNA(data_info, embed_size=10, num_layers=2)
    nfeat = torch.randn(num_nodes, data_info["node_feat_size"])
    model(g, nfeat)
