# pylint: disable=too-many-arguments, too-many-locals
from collections import OrderedDict
from itertools import product

import dgl
import pytest
import torch
from dgl.nn import CuGraphRelGraphConv, RelGraphConv

# TODO(tingyu66): Re-enable the following tests after updating cuGraph CI image.
options = OrderedDict(
    {
        "idtype_int": [False, True],
        "max_in_degree": [None, 8],
        "num_bases": [1, 2, 5],
        "regularizer": [None, "basis"],
        "self_loop": [False, True],
        "to_block": [False, True],
    }
)


def generate_graph():
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    g = dgl.graph((u, v))
    return g


@pytest.mark.parametrize(",".join(options.keys()), product(*options.values()))
def test_relgraphconv_equality(
    idtype_int, max_in_degree, num_bases, regularizer, self_loop, to_block
):
    device = "cuda:0"
    in_feat, out_feat, num_rels = 10, 2, 3
    args = (in_feat, out_feat, num_rels)
    kwargs = {
        "num_bases": num_bases,
        "regularizer": regularizer,
        "bias": False,
        "self_loop": self_loop,
    }
    g = generate_graph().to(device)
    g.edata[dgl.ETYPE] = torch.randint(num_rels, (g.num_edges(),)).to(device)
    if idtype_int:
        g = g.int()
    if to_block:
        g = dgl.to_block(g)
    feat = torch.rand(g.num_src_nodes(), in_feat).to(device)

    torch.manual_seed(0)
    conv1 = RelGraphConv(*args, **kwargs).to(device)

    torch.manual_seed(0)
    kwargs["apply_norm"] = False
    conv2 = CuGraphRelGraphConv(*args, **kwargs).to(device)

    out1 = conv1(g, feat, g.edata[dgl.ETYPE])
    out2 = conv2(g, feat, g.edata[dgl.ETYPE], max_in_degree=max_in_degree)
    assert torch.allclose(out1, out2, atol=1e-06)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    end = -1 if self_loop else None
    assert torch.allclose(conv1.linear_r.W.grad, conv2.W.grad[:end], atol=1e-6)

    if self_loop:
        assert torch.allclose(
            conv1.loop_weight.grad, conv2.W.grad[-1], atol=1e-6
        )

    if regularizer is not None:
        assert torch.allclose(
            conv1.linear_r.coeff.grad, conv2.coeff.grad, atol=1e-6
        )
