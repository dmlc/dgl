# pylint: disable=too-many-arguments, too-many-locals
from collections import OrderedDict
from itertools import product

import dgl
import pytest
import torch
from dgl.nn import CuGraphGATConv, GATConv

options = OrderedDict(
    {
        "idtype_int": [False, True],
        "max_in_degree": [None, 8],
        "num_heads": [1, 3],
        "to_block": [False, True],
    }
)


def generate_graph():
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    g = dgl.graph((u, v))
    return g


@pytest.mark.parametrize(",".join(options.keys()), product(*options.values()))
def test_gatconv_equality(idtype_int, max_in_degree, num_heads, to_block):
    device = "cuda:0"
    in_feat, out_feat = 10, 2
    args = (in_feat, out_feat, num_heads)
    kwargs = {"bias": False}
    g = generate_graph().to(device)
    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)
    feat = torch.rand(g.num_src_nodes(), in_feat).to(device)

    torch.manual_seed(0)
    conv1 = GATConv(*args, **kwargs, allow_zero_in_degree=True).to(device)
    out1 = conv1(g, feat)

    torch.manual_seed(0)
    conv2 = CuGraphGATConv(*args, **kwargs).to(device)
    dim = num_heads * out_feat
    with torch.no_grad():
        conv2.attn_weights.data[:dim] = conv1.attn_l.data.flatten()
        conv2.attn_weights.data[dim:] = conv1.attn_r.data.flatten()
        conv2.fc.weight.data[:] = conv1.fc.weight.data
    out2 = conv2(g, feat, max_in_degree=max_in_degree)
    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out1 = torch.rand_like(out1)
    grad_out2 = grad_out1.clone().detach()
    out1.backward(grad_out1)
    out2.backward(grad_out2)

    assert torch.allclose(conv1.fc.weight.grad, conv2.fc.weight.grad, atol=1e-6)
    assert torch.allclose(
        torch.cat((conv1.attn_l.grad, conv1.attn_r.grad), dim=0),
        conv2.attn_weights.grad.view(2, num_heads, out_feat),
        atol=1e-6,
    )
