import dgl
from dgl.contrib.cugraph.nn import RelGraphConv as RelGraphConvOps
from dgl.nn import RelGraphConv
import torch
import pytest

device = 'cuda'

def generate_graph():
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    g = dgl.graph((u, v))
    num_rels = 3
    g.edata[dgl.ETYPE] = torch.randint(num_rels, (g.num_edges(),))
    return g

def test_full_graph():
    in_feat, out_feat, num_rels, num_bases = 10, 2, 3, 2
    kwargs = {'num_bases': num_bases, 'regularizer': 'basis', 'bias': False, 'self_loop': False}
    g = generate_graph().to(device)
    feat = torch.ones(g.num_nodes(), in_feat).to(device)

    conv = RelGraphConv(in_feat, out_feat, num_rels, **kwargs).to(device)
    res = conv(g, feat, g.edata[dgl.ETYPE])

    fanout = g.in_degrees().max().item()
    conv_ops = RelGraphConvOps(in_feat, out_feat, num_rels, fanout, **kwargs)
    conv_ops.W.data = conv.linear_r.W.data
    conv_ops.coeff.data = conv.linear_r.coeff.data

    _, _, edge_ids = g.adj_sparse('csc')
    etypes = g.edata[dgl.ETYPE][edge_ids].type(torch.int32)
    res_ops = conv_ops(g, feat, etypes)

    assert torch.allclose(res, res_ops, rtol=1e-03)

def test_mfg():
    in_feat, out_feat, num_rels, num_bases = 10, 2, 3, 2
    kwargs = {'num_bases': num_bases, 'regularizer': 'basis', 'bias': False, 'self_loop': False}
    g = generate_graph().to(device)
    block = dgl.to_block(g)
    feat = torch.ones(g.num_nodes(), in_feat).to(device)

    conv = RelGraphConv(in_feat, out_feat, num_rels, **kwargs).to(device)
    res = conv(block, feat[block.srcdata[dgl.NID]], block.edata[dgl.ETYPE])

    fanout = block.in_degrees().max().item()
    conv_ops = RelGraphConvOps(in_feat, out_feat, num_rels, fanout, **kwargs)
    conv_ops.W.data = conv.linear_r.W.data
    conv_ops.coeff.data = conv.linear_r.coeff.data

    _, _, edge_ids = block.adj_sparse('csc')
    etypes = block.edata[dgl.ETYPE][edge_ids].type(torch.int32)
    res_ops = conv_ops(block, feat[block.srcdata[dgl.NID]], etypes)

    assert torch.allclose(res, res_ops, rtol=1e-03)
