import random
import unittest

import backend as F

import dgl
import numpy as np
import pytest
import torch
from dgl.ops import gather_mm, gsddmm, gspmm, segment_reduce
from utils import parametrize_idtype
from utils.graph_cases import get_cases

# Set seeds to make tests fully reproducible.
SEED = 12345  # random.randint(1, 99999)
random.seed(SEED)
np.random.seed(SEED)
dgl.seed(SEED)
F.seed(SEED)

udf_msg = {
    "add": lambda edges: {"m": edges.src["x"] + edges.data["w"]},
    "sub": lambda edges: {"m": edges.src["x"] - edges.data["w"]},
    "mul": lambda edges: {"m": edges.src["x"] * edges.data["w"]},
    "div": lambda edges: {"m": edges.src["x"] / edges.data["w"]},
    "copy_lhs": lambda edges: {"m": edges.src["x"]},
    "copy_rhs": lambda edges: {"m": edges.data["w"]},
}


def select(target, src, edge, dst):
    if target == "u":
        return src
    elif target == "v":
        return dst
    elif target == "e":
        return edge


def binary_op(msg, x, y):
    if msg == "add":
        return x + y
    elif msg == "sub":
        return x - y
    elif msg == "mul":
        return x * y
    elif msg == "div":
        return x / y
    elif msg == "dot":
        return F.sum(x * y, -1, keepdims=True)
    elif msg == "copy_lhs":
        return x
    elif msg == "copy_rhs":
        return y


def edge_func(lhs_target, rhs_target, msg):
    def foo(edges):
        return {
            "m": binary_op(
                msg,
                select(lhs_target, edges.src, edges.data, edges.dst)["x"],
                select(rhs_target, edges.src, edges.data, edges.dst)["y"],
            )
        }

    return foo


udf_apply_edges = {
    lhs_target
    + "_"
    + msg
    + "_"
    + rhs_target: edge_func(lhs_target, rhs_target, msg)
    for lhs_target in ["u", "v", "e"]
    for rhs_target in ["u", "v", "e"]
    for msg in ["add", "sub", "mul", "div", "dot", "copy_lhs", "copy_rhs"]
}

udf_reduce = {
    "sum": lambda nodes: {"v": F.sum(nodes.mailbox["m"], 1)},
    "min": lambda nodes: {"v": F.min(nodes.mailbox["m"], 1)},
    "max": lambda nodes: {"v": F.max(nodes.mailbox["m"], 1)},
}

graphs = [
    #    dgl.rand_graph(30, 0),
    dgl.rand_graph(30, 100),
    dgl.rand_bipartite("_U", "_E", "_V", 30, 40, 300),
]

spmm_shapes = [
    ((1, 2, 1, 3, 1), (4, 1, 3, 1, 1)),
    ((3, 3), (1, 3)),
    ((1,), (3,)),
    ((3,), (1,)),
    ((1,), (1,)),
    ((), ()),
]

sddmm_shapes = [
    ((1, 2, 1, 3, 1), (4, 1, 3, 1, 1)),
    ((5, 3, 1, 7), (1, 3, 7, 7)),
    ((1, 3, 3), (4, 1, 3)),
    ((3,), (3,)),
    ((1,), (1,)),
]


@pytest.mark.parametrize("g", graphs)
@pytest.mark.parametrize("shp", spmm_shapes)
@pytest.mark.parametrize(
    "msg", ["add", "sub", "mul", "div", "copy_lhs", "copy_rhs"]
)
@pytest.mark.parametrize("reducer", ["sum", "min", "max"])
@parametrize_idtype
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spmm(idtype, dtype, g, shp, msg, reducer):
    g = g.astype(idtype).to(F.ctx())
    print(g)
    print(g.idtype)

    hu = F.tensor(
        np.random.rand(*((g.number_of_src_nodes(),) + shp[0])).astype(dtype) + 1
    )
    he = F.tensor(
        np.random.rand(*((g.num_edges(),) + shp[1])).astype(dtype) + 1
    )
    print("u shape: {}, e shape: {}".format(F.shape(hu), F.shape(he)))

    g.srcdata["x"] = F.attach_grad(F.clone(hu))
    g.edata["w"] = F.attach_grad(F.clone(he))
    print("SpMM(message func: {}, reduce func: {})".format(msg, reducer))

    u = F.attach_grad(F.clone(hu))
    e = F.attach_grad(F.clone(he))
    with F.record_grad():
        v = gspmm(g, msg, reducer, u, e)
        if reducer in ["max", "min"]:
            v = F.replace_inf_with_zero(v)
        if g.num_edges() > 0:
            F.backward(F.reduce_sum(v))
            if msg != "copy_rhs":
                grad_u = F.grad(u)
            if msg != "copy_lhs":
                grad_e = F.grad(e)

    with F.record_grad():
        g.update_all(udf_msg[msg], udf_reduce[reducer])
        if g.num_edges() > 0:
            v1 = g.dstdata["v"]
            assert F.allclose(v, v1)
            print("forward passed")

            F.backward(F.reduce_sum(v1))
            if msg != "copy_rhs":
                if reducer in [
                    "min",
                    "max",
                ]:  # there might be some numerical errors
                    rate = F.reduce_sum(
                        F.abs(F.grad(g.srcdata["x"]) - grad_u)
                    ) / F.reduce_sum(F.abs(grad_u))
                    assert F.as_scalar(rate) < 1e-2, rate
                else:
                    assert F.allclose(F.grad(g.srcdata["x"]), grad_u)
            if msg != "copy_lhs":
                if reducer in ["min", "max"]:
                    rate = F.reduce_sum(
                        F.abs(F.grad(g.edata["w"]) - grad_e)
                    ) / F.reduce_sum(F.abs(grad_e))
                    assert F.as_scalar(rate) < 1e-2, rate
                else:
                    assert F.allclose(F.grad(g.edata["w"]), grad_e)
            print("backward passed")

    g.srcdata.pop("x")
    g.edata.pop("w")
    if "v" in g.dstdata:
        g.dstdata.pop("v")


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="Only support PyTorch for now.",
)
@parametrize_idtype
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [(torch.float16, 1e-3, 0.5), (torch.bfloat16, 4e-3, 2.0)],
)
def test_half_spmm(idtype, dtype, rtol, atol):
    if F._default_context_str == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU.")
    if (
        F._default_context_str == "gpu"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("BF16 is not supported.")

    # make sure the spmm result is < 512 to match the rtol/atol we set.
    g = dgl.graph(
        (torch.arange(900), torch.tensor([0] * 900)),
        idtype=idtype,
        device=F.ctx(),
    )
    feat_fp32 = torch.rand((g.num_src_nodes(), 32)).to(F.ctx())
    feat_half = feat_fp32.to(dtype)

    # test SpMMCSR
    g = g.formats(["csc"])
    res_fp32 = dgl.ops.copy_u_sum(g, feat_fp32)[0]
    res_half = dgl.ops.copy_u_sum(g, feat_half)[0].float()
    assert torch.allclose(res_fp32, res_half, rtol=rtol, atol=atol)

    # test SpMMCOO
    # TODO(Xin): half-precision SpMMCoo is temporally disabled.
    # g = g.formats(['coo'])
    # res_fp32 = dgl.ops.copy_u_sum(g, feat_fp32)[0]
    # res_half = dgl.ops.copy_u_sum(g, feat_half)[0].float()
    # assert torch.allclose(res_fp32, res_half, rtol=rtol, atol=atol)


@pytest.mark.parametrize("g", graphs)
@pytest.mark.parametrize("shp", sddmm_shapes)
@pytest.mark.parametrize("lhs_target", ["u", "v", "e"])
@pytest.mark.parametrize("rhs_target", ["u", "v", "e"])
@pytest.mark.parametrize(
    "msg", ["add", "sub", "mul", "div", "dot", "copy_lhs", "copy_rhs"]
)
@parametrize_idtype
def test_sddmm(g, shp, lhs_target, rhs_target, msg, idtype):
    if lhs_target == rhs_target:
        return
    g = g.astype(idtype).to(F.ctx())
    if dgl.backend.backend_name == "mxnet" and g.num_edges() == 0:
        pytest.skip()  # mxnet do not support zero shape tensor
    print(g)
    print(g.idtype)

    len_lhs = select(
        lhs_target,
        g.number_of_src_nodes(),
        g.num_edges(),
        g.number_of_dst_nodes(),
    )
    lhs_shp = (len_lhs,) + shp[0]
    len_rhs = select(
        rhs_target,
        g.number_of_src_nodes(),
        g.num_edges(),
        g.number_of_dst_nodes(),
    )
    rhs_shp = (len_rhs,) + shp[1]
    feat_lhs = F.tensor(np.random.rand(*lhs_shp) + 1)
    feat_rhs = F.tensor(np.random.rand(*rhs_shp) + 1)
    print(
        "lhs shape: {}, rhs shape: {}".format(
            F.shape(feat_lhs), F.shape(feat_rhs)
        )
    )

    lhs_frame = select(lhs_target, g.srcdata, g.edata, g.dstdata)
    rhs_frame = select(rhs_target, g.srcdata, g.edata, g.dstdata)
    lhs_frame["x"] = F.attach_grad(F.clone(feat_lhs))
    rhs_frame["y"] = F.attach_grad(F.clone(feat_rhs))
    msg_func = lhs_target + "_" + msg + "_" + rhs_target
    print("SDDMM(message func: {})".format(msg_func))

    lhs = F.attach_grad(F.clone(feat_lhs))
    rhs = F.attach_grad(F.clone(feat_rhs))
    with F.record_grad():
        e = gsddmm(
            g, msg, lhs, rhs, lhs_target=lhs_target, rhs_target=rhs_target
        )
        F.backward(F.reduce_sum(e))
        grad_lhs = F.grad(lhs)
        grad_rhs = F.grad(rhs)

    with F.record_grad():
        g.apply_edges(udf_apply_edges[msg_func])
        if g.num_edges() > 0:
            e1 = g.edata["m"]
            assert F.allclose(e, e1)
            print("forward passed")

            F.backward(F.reduce_sum(e1))
            if msg != "copy_rhs":
                assert F.allclose(F.grad(lhs_frame["x"]), grad_lhs)
            if msg != "copy_lhs":
                assert F.allclose(F.grad(rhs_frame["y"]), grad_rhs)
            print("backward passed")

    lhs_frame.pop("x")
    rhs_frame.pop("y")
    if "m" in g.edata:
        g.edata.pop("m")


@pytest.mark.parametrize("reducer", ["sum", "max", "min", "mean"])
def test_segment_reduce(reducer):
    ctx = F.ctx()
    value = F.tensor(np.random.rand(10, 5))
    v1 = F.attach_grad(F.clone(value))
    v2 = F.attach_grad(F.clone(value))
    seglen = F.tensor([2, 3, 0, 4, 1, 0, 0])
    u = F.copy_to(F.arange(0, F.shape(value)[0], F.int32), ctx)
    v = F.repeat(
        F.copy_to(F.arange(0, len(seglen), F.int32), ctx), seglen, dim=0
    )

    num_nodes = {"_U": len(u), "_V": len(seglen)}
    g = dgl.convert.heterograph(
        {("_U", "_E", "_V"): (u, v)}, num_nodes_dict=num_nodes
    )
    with F.record_grad():
        rst1 = gspmm(g, "copy_lhs", reducer, v1, None)
        if reducer in ["max", "min"]:
            rst1 = F.replace_inf_with_zero(rst1)
        F.backward(F.reduce_sum(rst1))
        grad1 = F.grad(v1)

    with F.record_grad():
        rst2 = segment_reduce(seglen, v2, reducer=reducer)
        F.backward(F.reduce_sum(rst2))
        assert F.allclose(rst1, rst2)
        print("forward passed")

        grad2 = F.grad(v2)
        assert F.allclose(grad1, grad2)
        print("backward passed")


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
@pytest.mark.parametrize("feat_size", [1, 8, 16, 64, 256])
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-2),
        (torch.float32, 3e-3),
        (torch.float64, 1e-4),
    ],
)
def test_segment_mm(idtype, feat_size, dtype, tol):
    if F._default_context_str == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU.")
    if (
        F._default_context_str == "gpu"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("BF16 is not supported.")
    dev = F.ctx()
    # input
    a = torch.tensor(np.random.rand(100, feat_size)).to(dev).to(dtype)
    a.requires_grad_()
    b = (
        torch.tensor(np.random.rand(10, feat_size, feat_size + 1))
        .to(dev)
        .to(dtype)
    )
    b.requires_grad_()
    seglen_a = torch.tensor([10, 15, 8, 0, 1, 9, 18, 24, 15, 0]).to(idtype)
    dc = torch.tensor(np.random.rand(100, feat_size + 1)).to(dev).to(dtype)
    # compute
    c = dgl.ops.segment_mm(a, b, seglen_a)
    c.backward(dc)
    da = a.grad.clone()
    db = b.grad.clone()
    # ground truth
    c_t = []
    off = 0
    for i, l in enumerate(seglen_a):
        c_t.append(a[off : off + l] @ b[i])
        off += l
    c_t = torch.cat(c_t).to(dtype)
    a.grad.zero_()
    b.grad.zero_()
    c_t.backward(dc)
    da_t = a.grad
    db_t = b.grad

    assert torch.allclose(c, c_t, atol=tol, rtol=tol)
    assert torch.allclose(da, da_t, atol=tol, rtol=tol)
    assert torch.allclose(db, db_t, atol=tol, rtol=tol)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@pytest.mark.parametrize("feat_size", [1, 8, 16, 64, 256])
@pytest.mark.parametrize(
    "dtype, tol",
    [
        (torch.float16, 1e-2),
        (torch.bfloat16, 2e-2),
        (torch.float32, 3e-3),
        (torch.float64, 1e-4),
    ],
)
def test_gather_mm_idx_b(feat_size, dtype, tol):
    if F._default_context_str == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU.")

    if F._default_context_str == "gpu":
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 is not supported.")

        if (
            dtype == torch.float16
            and torch.cuda.get_device_capability() < (7, 0)
        ) or (
            dtype == torch.bfloat16
            and torch.cuda.get_device_capability() < (8, 0)
        ):
            pytest.skip(
                f"{dtype} is not supported for atomic operations on GPU with "
                f"cuda capability ({torch.cuda.get_device_capability()})."
            )

    dev = F.ctx()
    # input
    a = torch.tensor(np.random.rand(100, feat_size)).to(dev).to(dtype)
    a.requires_grad_()
    b = (
        torch.tensor(np.random.rand(10, feat_size, feat_size + 1))
        .to(dev)
        .to(dtype)
    )
    b.requires_grad_()
    idx = torch.tensor(np.random.randint(0, 10, 100)).to(dev).long()
    dc = torch.tensor(np.random.rand(100, feat_size + 1)).to(dev).to(dtype)
    # compute
    c = gather_mm(a, b, idx_b=idx)
    c.backward(dc)
    da = a.grad.clone()
    db = b.grad.clone()
    # ground truth
    c_t = torch.bmm(a.unsqueeze(1), b[idx]).squeeze(1)
    a.grad.zero_()
    b.grad.zero_()
    c_t.backward(dc)
    da_t = a.grad
    db_t = b.grad

    assert torch.allclose(c, c_t, atol=tol, rtol=tol)
    assert torch.allclose(da, da_t, atol=tol, rtol=tol)
    assert torch.allclose(db, db_t, atol=tol, rtol=tol)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@parametrize_idtype
@pytest.mark.parametrize("feat_size", [1, 8, 16, 64, 256])
def _test_gather_mm_idx_a(idtype, feat_size):
    # TODO(minjie): currently disabled due to bugs in the CUDA kernel. Need to fix it later.
    import torch

    dev = F.ctx()
    # input
    a = torch.tensor(np.random.rand(10, feat_size)).to(dev)
    a.requires_grad_()
    b = torch.tensor(np.random.rand(100, feat_size, feat_size + 1)).to(dev)
    b.requires_grad_()
    idx = torch.tensor(np.random.randint(0, 10, 100)).to(dev)
    dc = torch.tensor(np.random.rand(100, feat_size + 1)).to(dev)
    # compute
    c = gather_mm(a, b, idx_a=idx)
    c.backward(dc)
    da = a.grad.clone()
    db = b.grad.clone()
    # ground truth
    c_t = torch.bmm(a[idx].unsqueeze(1), b).squeeze(1)
    a.grad.zero_()
    b.grad.zero_()
    c_t.backward(dc)
    da_t = a.grad
    db_t = b.grad

    assert torch.allclose(c, c_t, atol=1e-4, rtol=1e-4)
    assert torch.allclose(da, da_t, atol=1e-4, rtol=1e-4)
    assert torch.allclose(db, db_t, atol=1e-4, rtol=1e-4)


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="Only support PyTorch for now"
)
@unittest.skipIf(
    F._default_context_str == "gpu", reason="Libxsmm only fit in CPU."
)
def test_use_libxsmm_switch():
    import torch

    g = dgl.graph(([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]))
    x = torch.ones(3, 2, requires_grad=True)
    y = torch.arange(1, 13).float().view(6, 2).requires_grad_()

    dgl.use_libxsmm(False)
    assert ~dgl.is_libxsmm_enabled()
    dgl.ops.u_mul_e_sum(g, x, y)
    dgl.use_libxsmm(True)
    assert dgl.is_libxsmm_enabled()
    dgl.ops.u_mul_e_sum(g, x, y)
