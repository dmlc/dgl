import pytest
import torch
import dgl
from dgl.nn import CuGraphRelGraphConv
from dgl.nn import RelGraphConv

# TODO(tingyu66): Re-enable the following tests after updating cuGraph CI image.
use_longs = [False, True]
max_in_degrees = [None, 8]
regularizers = [None, "basis"]
device = "cuda"


def generate_graph():
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    g = dgl.graph((u, v))
    num_rels = 3
    g.edata[dgl.ETYPE] = torch.randint(num_rels, (g.num_edges(),))
    return g

@pytest.mark.skip()
@pytest.mark.parametrize('use_long', use_longs)
@pytest.mark.parametrize('max_in_degree', max_in_degrees)
@pytest.mark.parametrize("regularizer", regularizers)
def test_full_graph(use_long, max_in_degree, regularizer):
    in_feat, out_feat, num_rels, num_bases = 10, 2, 3, 2
    kwargs = {
        "num_bases": num_bases,
        "regularizer": regularizer,
        "bias": False,
        "self_loop": False,
    }
    g = generate_graph().to(device)
    if use_long:
        g = g.long()
    else:
        g = g.int()
    feat = torch.ones(g.num_nodes(), in_feat).to(device)

    torch.manual_seed(0)
    conv1 = RelGraphConv(in_feat, out_feat, num_rels, **kwargs).to(device)

    torch.manual_seed(0)
    conv2 = CuGraphRelGraphConv(
        in_feat, out_feat, num_rels, max_in_degree=max_in_degree, **kwargs
    ).to(device)

    out1 = conv1(g, feat, g.edata[dgl.ETYPE])
    out2 = conv2(g, feat, g.edata[dgl.ETYPE])

    assert torch.allclose(out1, out2, atol=1e-06)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert torch.allclose(conv1.linear_r.W.grad, conv2.W.grad, atol=1e-6)
    if regularizer is not None:
        assert torch.allclose(
            conv1.linear_r.coeff.grad, conv2.coeff.grad, atol=1e-6
        )

@pytest.mark.skip()
@pytest.mark.parametrize('max_in_degree', max_in_degrees)
@pytest.mark.parametrize("regularizer", regularizers)
def test_mfg(max_in_degree, regularizer):
    in_feat, out_feat, num_rels, num_bases = 10, 2, 3, 2
    kwargs = {
        "num_bases": num_bases,
        "regularizer": regularizer,
        "bias": False,
        "self_loop": False,
    }
    g = generate_graph().to(device)
    block = dgl.to_block(g)
    feat = torch.ones(g.num_nodes(), in_feat).to(device)

    torch.manual_seed(0)
    conv1 = RelGraphConv(in_feat, out_feat, num_rels, **kwargs).to(device)

    torch.manual_seed(0)
    conv2 = CuGraphRelGraphConv(
        in_feat, out_feat, num_rels, max_in_degree=max_in_degree, **kwargs
    ).to(device)

    out1 = conv1(block, feat[block.srcdata[dgl.NID]], block.edata[dgl.ETYPE])
    out2 = conv2(block, feat[block.srcdata[dgl.NID]], block.edata[dgl.ETYPE])

    assert torch.allclose(out1, out2, atol=1e-06)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert torch.allclose(conv1.linear_r.W.grad, conv2.W.grad, atol=1e-6)
    if regularizer is not None:
        assert torch.allclose(
            conv1.linear_r.coeff.grad, conv2.coeff.grad, atol=1e-6
        )
