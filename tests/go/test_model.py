import pytest
from test_utils.graph_cases import get_cases

@pytest.mark.parametrize('g', get_cases(['has_scalar_e_feature']))
def test_gcn(g):
    from dglgo.model import GCN

    data_info = {
        'num_nodes': g.num_nodes(),
        'out_size': 7
    }
    node_feat = None
    edge_feat = g.edata['scalar_w']

    # node embedding + not use_edge_weight
    model = GCN(data_info, embed_size=10, use_edge_weight=False)
    model(g, node_feat)

    # node embedding + use_edge_weight
    model = GCN(data_info, embed_size=10, use_edge_weight=True)
    model(g, node_feat, edge_feat)

    data_info['in_size'] = g.ndata['h'].shape[-1]
    node_feat = g.ndata['h']

    # node feat + not use_edge_weight
    model = GCN(data_info, embed_size=-1, use_edge_weight=False)
    model(g, node_feat)

    # node feat + use_edge_weight
    model = GCN(data_info, embed_size=-1, use_edge_weight=True)
    model(g, node_feat, edge_feat)
