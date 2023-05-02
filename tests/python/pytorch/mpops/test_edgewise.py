import random

import backend as F

import dgl
import numpy as np
import pytest
import torch
from utils import parametrize_idtype

random.seed(42)
np.random.seed(42)
dgl.seed(42)
torch.random.manual_seed(42)


@parametrize_idtype
@pytest.mark.parametrize("feat_size", [(5,), ()])
def test_copy_u(idtype, feat_size):
    ctx = F.ctx()
    g = dgl.rand_graph(30, 100)
    g = g.astype(idtype).to(ctx)
    x = torch.randn(
        (g.num_nodes(),) + feat_size, requires_grad=True, device=ctx
    )

    y = dgl.copy_u(g, x)
    y.sum().backward()
    x_grad = x.grad

    x.grad.zero_()
    u, v = g.edges()
    y_true = x[u.long()]
    y_true.sum().backward()
    x_grad_true = x.grad

    assert torch.allclose(y, y_true)
    assert torch.allclose(x_grad, x_grad_true)


@parametrize_idtype
@pytest.mark.parametrize("feat_size", [(5,), ()])
def test_copy_u_hetero(idtype, feat_size):
    ctx = F.ctx()
    hg = dgl.heterograph(
        {
            ("user", "follow", "user"): ([0, 1, 2], [2, 3, 4]),
            ("user", "like", "movie"): ([3, 3, 1, 2], [0, 0, 1, 1]),
        }
    )

    hg = hg.astype(idtype).to(ctx)
    x = torch.randn(
        (hg.num_nodes("user"),) + feat_size, requires_grad=True, device=ctx
    )

    y = dgl.copy_u(hg, x, etype="like")
    y.sum().backward()
    x_grad = x.grad

    x.grad.zero_()
    u, v = hg.edges(etype="like")
    y_true = x[u.long()]
    y_true.sum().backward()
    x_grad_true = x.grad

    assert torch.allclose(y, y_true)
    assert torch.allclose(x_grad, x_grad_true)


@parametrize_idtype
@pytest.mark.parametrize("feat_size", [(5,), ()])
def test_copy_v(idtype, feat_size):
    ctx = F.ctx()
    g = dgl.rand_graph(30, 100)
    g = g.astype(idtype).to(ctx)
    x = torch.randn(
        (g.num_nodes(),) + feat_size, requires_grad=True, device=ctx
    )

    y = dgl.copy_v(g, x)
    y.sum().backward()
    x_grad = x.grad

    x.grad.zero_()
    u, v = g.edges()
    y_true = x[v.long()]
    y_true.sum().backward()
    x_grad_true = x.grad

    assert torch.allclose(y, y_true)
    assert torch.allclose(x_grad, x_grad_true)


@parametrize_idtype
@pytest.mark.parametrize("feat_size", [(5,), ()])
def test_copy_v_hetero(idtype, feat_size):
    ctx = F.ctx()
    hg = dgl.heterograph(
        {
            ("user", "follow", "user"): ([0, 1, 2], [2, 3, 4]),
            ("user", "like", "movie"): ([3, 3, 1, 2], [0, 0, 1, 1]),
        }
    )

    hg = hg.astype(idtype).to(ctx)
    x = torch.randn(
        (hg.num_nodes("movie"),) + feat_size, requires_grad=True, device=ctx
    )

    y = dgl.copy_v(hg, x, etype="like")
    y.sum().backward()
    x_grad = x.grad

    x.grad.zero_()
    u, v = hg.edges(etype="like")
    y_true = x[v.long()]
    y_true.sum().backward()
    x_grad_true = x.grad

    assert torch.allclose(y, y_true)
    assert torch.allclose(x_grad, x_grad_true)


binary_arg_sizes = [
    ((5,), (5,)),
    ((5,), ()),
    ((), (5,)),
    ((1, 3, 3), (4, 1, 3)),
    ((3, 3), (4, 1, 3)),
    ((4, 1, 3), (3, 3)),
]

dot_arg_sizes = [
    ((5,), (5,)),
    ((1, 3, 3), (4, 1, 3)),
    ((3, 3), (4, 1, 3)),
    ((4, 1, 3), (3, 3)),
]

ops = ["add", "sub", "mul", "div"]


def pad_shape(x, y, x_size, y_size):
    xy_size = torch.broadcast_shapes(x_size, y_size)
    new_x_size = (1,) * (len(xy_size) - len(x_size)) + x_size
    new_y_size = (1,) * (len(xy_size) - len(y_size)) + y_size
    new_x = x.view(-1, *new_x_size)
    new_y = y.view(-1, *new_y_size)
    return new_x, new_y


@parametrize_idtype
@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("x_size,y_size", binary_arg_sizes)
def test_u_op_v(idtype, op, x_size, y_size):
    ctx = F.ctx()
    g = dgl.rand_graph(30, 100)
    g = g.astype(idtype).to(ctx)
    x = torch.randn((g.num_nodes(),) + x_size, requires_grad=True, device=ctx)
    y = torch.randn((g.num_nodes(),) + y_size, requires_grad=True, device=ctx)

    f_dgl = getattr(dgl, f"u_{op}_v")
    z = f_dgl(g, x, y)
    z.sum().backward()
    x_grad = x.grad
    y_grad = y.grad

    x_grad.zero_()
    y_grad.zero_()
    u, v = g.edges()
    f_torch = getattr(torch, op)
    x_u, y_v = pad_shape(x[u.long()], y[v.long()], x_size, y_size)
    z_true = f_torch(x_u, y_v)
    z_true.sum().backward()
    x_grad_true = x.grad
    y_grad_true = y.grad

    assert torch.allclose(z, z_true)
    assert torch.allclose(x_grad, x_grad_true)
    assert torch.allclose(y_grad, y_grad_true)


@parametrize_idtype
@pytest.mark.parametrize("x_size,y_size", dot_arg_sizes)
def test_u_dot_v(idtype, x_size, y_size):
    ctx = F.ctx()
    g = dgl.rand_graph(30, 100)
    g = g.astype(idtype).to(ctx)
    x = torch.randn((g.num_nodes(),) + x_size, requires_grad=True, device=ctx)
    y = torch.randn((g.num_nodes(),) + y_size, requires_grad=True, device=ctx)

    z = dgl.u_dot_v(g, x, y)
    z.sum().backward()
    x_grad = x.grad
    y_grad = y.grad

    x_grad.zero_()
    y_grad.zero_()
    u, v = g.edges()
    x_u, y_v = pad_shape(x[u.long()], y[v.long()], x_size, y_size)
    z_true = (x_u * y_v).sum(-1).unsqueeze(-1)
    z_true.sum().backward()
    x_grad_true = x.grad
    y_grad_true = y.grad

    assert torch.allclose(z, z_true, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x_grad, x_grad_true)
    assert torch.allclose(y_grad, y_grad_true)
