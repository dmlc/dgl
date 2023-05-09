import io

import backend as F

import dgl.nn.pytorch as nn
import pytest
from utils import parametrize_idtype
from utils.graph_cases import get_cases

tmp_buffer = io.BytesIO()


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo"], exclude=["zero-degree"]))
def test_gatedgcn_conv(g, idtype):
    ctx = F.ctx()
    g = g.astype(idtype).to(ctx)
    gatedgcnconv = nn.GatedGCNConv(10, 10, 5)
    feat = F.randn((g.num_nodes(), 10))
    efeat = F.randn((g.num_edges(), 10))
    gatedgcnconv = gatedgcnconv.to(ctx)

    h, edge_h = gatedgcnconv(g, feat, efeat)
    # current we only do shape check
    assert h.shape == (g.number_of_dst_nodes(), 5)
    assert edge_h.shape == (g.number_of_edges(), 5)
