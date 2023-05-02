import unittest

import backend as F

import dgl
import networkx as nx
import numpy as np
import pytest
from utils import parametrize_idtype
from utils.graph_cases import get_cases


@parametrize_idtype
def test_sum_case1(idtype):
    # NOTE: If you want to update this test case, remember to update the docstring
    #  example too!!!
    g1 = dgl.graph(([0, 1], [1, 0]), idtype=idtype, device=F.ctx())
    g1.ndata["h"] = F.tensor([1.0, 2.0])
    g2 = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g2.ndata["h"] = F.tensor([1.0, 2.0, 3.0])
    bg = dgl.batch([g1, g2])
    bg.ndata["w"] = F.tensor([0.1, 0.2, 0.1, 0.5, 0.2])
    assert F.allclose(F.tensor([3.0]), dgl.sum_nodes(g1, "h"))
    assert F.allclose(F.tensor([3.0, 6.0]), dgl.sum_nodes(bg, "h"))
    assert F.allclose(F.tensor([0.5, 1.7]), dgl.sum_nodes(bg, "h", "w"))


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo"], exclude=["dglgraph"]))
@pytest.mark.parametrize("reducer", ["sum", "max", "mean"])
def test_reduce_readout(g, idtype, reducer):
    g = g.astype(idtype).to(F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 3))
    g.edata["h"] = F.randn((g.num_edges(), 2))

    # Test.1: node readout
    x = dgl.readout_nodes(g, "h", op=reducer)
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = dgl.readout_nodes(sg, "h", op=reducer)
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    x = getattr(dgl, "{}_nodes".format(reducer))(g, "h")
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, "{}_nodes".format(reducer))(sg, "h")
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    # Test.2: edge readout
    x = dgl.readout_edges(g, "h", op=reducer)
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = dgl.readout_edges(sg, "h", op=reducer)
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    x = getattr(dgl, "{}_edges".format(reducer))(g, "h")
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, "{}_edges".format(reducer))(sg, "h")
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo"], exclude=["dglgraph"]))
@pytest.mark.parametrize("reducer", ["sum", "max", "mean"])
def test_weighted_reduce_readout(g, idtype, reducer):
    g = g.astype(idtype).to(F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 3))
    g.ndata["w"] = F.randn((g.num_nodes(), 1))
    g.edata["h"] = F.randn((g.num_edges(), 2))
    g.edata["w"] = F.randn((g.num_edges(), 1))

    # Test.1: node readout
    x = dgl.readout_nodes(g, "h", "w", op=reducer)
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = dgl.readout_nodes(sg, "h", "w", op=reducer)
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    x = getattr(dgl, "{}_nodes".format(reducer))(g, "h", "w")
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, "{}_nodes".format(reducer))(sg, "h", "w")
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    # Test.2: edge readout
    x = dgl.readout_edges(g, "h", "w", op=reducer)
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = dgl.readout_edges(sg, "h", "w", op=reducer)
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))

    x = getattr(dgl, "{}_edges".format(reducer))(g, "h", "w")
    # check correctness
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        sx = getattr(dgl, "{}_edges".format(reducer))(sg, "h", "w")
        subx.append(sx)
    assert F.allclose(x, F.cat(subx, dim=0))


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo"], exclude=["dglgraph"]))
@pytest.mark.parametrize("descending", [True, False])
def test_topk(g, idtype, descending):
    g = g.astype(idtype).to(F.ctx())
    g.ndata["x"] = F.randn((g.num_nodes(), 3))

    # Test.1: to test the case where k > number of nodes.
    dgl.topk_nodes(g, "x", 100, sortby=-1)

    # Test.2: test correctness
    min_nnodes = F.asnumpy(g.batch_num_nodes()).min()
    if min_nnodes <= 1:
        return
    k = min_nnodes - 1
    val, indices = dgl.topk_nodes(g, "x", k, descending=descending, sortby=-1)
    print(k)
    print(g.ndata["x"])
    print("val", val)
    print("indices", indices)
    subg = dgl.unbatch(g)
    subval, subidx = [], []
    for sg in subg:
        subx = F.asnumpy(sg.ndata["x"])
        ai = np.argsort(subx[:, -1:].flatten())
        if descending:
            ai = np.ascontiguousarray(ai[::-1])
        subx = np.expand_dims(subx[ai[:k]], 0)
        subval.append(F.tensor(subx))
        subidx.append(F.tensor(np.expand_dims(ai[:k], 0)))
    print(F.cat(subval, dim=0))
    assert F.allclose(val, F.cat(subval, dim=0))
    assert F.allclose(indices, F.cat(subidx, dim=0))

    # Test.3: sorby=None
    dgl.topk_nodes(g, "x", k, sortby=None)

    g.edata["x"] = F.randn((g.num_edges(), 3))

    # Test.4: topk edges where k > number of edges.
    dgl.topk_edges(g, "x", 100, sortby=-1)

    # Test.5: topk edges test correctness
    min_nedges = F.asnumpy(g.batch_num_edges()).min()
    if min_nedges <= 1:
        return
    k = min_nedges - 1
    val, indices = dgl.topk_edges(g, "x", k, descending=descending, sortby=-1)
    print(k)
    print(g.edata["x"])
    print("val", val)
    print("indices", indices)
    subg = dgl.unbatch(g)
    subval, subidx = [], []
    for sg in subg:
        subx = F.asnumpy(sg.edata["x"])
        ai = np.argsort(subx[:, -1:].flatten())
        if descending:
            ai = np.ascontiguousarray(ai[::-1])
        subx = np.expand_dims(subx[ai[:k]], 0)
        subval.append(F.tensor(subx))
        subidx.append(F.tensor(np.expand_dims(ai[:k], 0)))
    print(F.cat(subval, dim=0))
    assert F.allclose(val, F.cat(subval, dim=0))
    assert F.allclose(indices, F.cat(subidx, dim=0))


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo"], exclude=["dglgraph"]))
def test_softmax(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    g.ndata["h"] = F.randn((g.num_nodes(), 3))
    g.edata["h"] = F.randn((g.num_edges(), 2))

    # Test.1: node readout
    x = dgl.softmax_nodes(g, "h")
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        subx.append(F.softmax(sg.ndata["h"], dim=0))
    assert F.allclose(x, F.cat(subx, dim=0))

    # Test.2: edge readout
    x = dgl.softmax_edges(g, "h")
    subg = dgl.unbatch(g)
    subx = []
    for sg in subg:
        subx.append(F.softmax(sg.edata["h"], dim=0))
    assert F.allclose(x, F.cat(subx, dim=0))


@parametrize_idtype
@pytest.mark.parametrize("g", get_cases(["homo"], exclude=["dglgraph"]))
def test_broadcast(idtype, g):
    g = g.astype(idtype).to(F.ctx())
    gfeat = F.randn((g.batch_size, 3))

    # Test.0: broadcast_nodes
    g.ndata["h"] = dgl.broadcast_nodes(g, gfeat)
    subg = dgl.unbatch(g)
    for i, sg in enumerate(subg):
        assert F.allclose(
            sg.ndata["h"],
            F.repeat(F.reshape(gfeat[i], (1, 3)), sg.num_nodes(), dim=0),
        )

    # Test.1: broadcast_edges
    g.edata["h"] = dgl.broadcast_edges(g, gfeat)
    subg = dgl.unbatch(g)
    for i, sg in enumerate(subg):
        assert F.allclose(
            sg.edata["h"],
            F.repeat(F.reshape(gfeat[i], (1, 3)), sg.num_edges(), dim=0),
        )
