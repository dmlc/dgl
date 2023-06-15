# pylint: disable=too-many-arguments, too-many-locals
from collections import OrderedDict
from itertools import product

import dgl
import pytest
import torch
from dgl.nn import CuGraphSAGEConv, SAGEConv

options = OrderedDict(
    {
        "idtype_int": [False, True],
        "max_in_degree": [None, 8],
        "to_block": [False, True],
    }
)


def generate_graph():
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    g = dgl.graph((u, v))
    return g


@pytest.mark.parametrize(",".join(options.keys()), product(*options.values()))
def test_SAGEConv_equality(idtype_int, max_in_degree, to_block):
    device = "cuda:0"
    in_feat, out_feat = 5, 2
    kwargs = {"aggregator_type": "mean"}
    g = generate_graph().to(device)
    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)
    feat = torch.rand(g.num_src_nodes(), in_feat).to(device)

    torch.manual_seed(0)
    conv1 = SAGEConv(in_feat, out_feat, **kwargs).to(device)

    torch.manual_seed(0)
    conv2 = CuGraphSAGEConv(in_feat, out_feat, **kwargs).to(device)

    with torch.no_grad():
        conv2.linear.weight.data[:, :in_feat] = conv1.fc_neigh.weight.data
        conv2.linear.weight.data[:, in_feat:] = conv1.fc_self.weight.data
        conv2.linear.bias.data[:] = conv1.fc_self.bias.data

    out1 = conv1(g, feat)
    out2 = conv2(g, feat, max_in_degree=max_in_degree)
    assert torch.allclose(out1, out2, atol=1e-06)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert torch.allclose(
        conv1.fc_neigh.weight.grad,
        conv2.linear.weight.grad[:, :in_feat],
        atol=1e-6,
    )
    assert torch.allclose(
        conv1.fc_self.weight.grad,
        conv2.linear.weight.grad[:, in_feat:],
        atol=1e-6,
    )
    assert torch.allclose(
        conv1.fc_self.bias.grad, conv2.linear.bias.grad, atol=1e-6
    )
